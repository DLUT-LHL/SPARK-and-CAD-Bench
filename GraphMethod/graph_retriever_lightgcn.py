import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Configuration
CURRENT_DIR = Path(__file__).parent.resolve()

class LightGCNAdapter(nn.Module):
    """
    Lightweight adapter to project BGE text features into a task-specific vector space.
    Following LightGCN principles, no non-linear activation is used after projection.
    """
    def __init__(self, input_dim, hidden_dim=768):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim, bias=False)
        nn.init.eye_(self.projection.weight)

    def forward(self, x):
        return self.projection(x)

class HypergraphLightGCNPlanner:
    def __init__(self, json_data, model_name='BAAI/bge-m3',
                 k_hops=3, device=None, alpha_weights=None):
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[LightGCN-Planner] Initializing on {self.device}")

        # Initialize text encoder
        model_path_obj = Path(model_name)
        
        if model_path_obj.exists() and any(model_path_obj.iterdir()):
            self.encoder = SentenceTransformer(str(model_path_obj), device=self.device)
        else:
            default_hf_model = 'BAAI/bge-m3'
            print(f"[LightGCN-Planner] Local model not found, downloading {default_hf_model} from HuggingFace...")
            self.encoder = SentenceTransformer(default_hf_model, device=self.device)
            self.encoder.save(str(model_path_obj))

        self.embed_dim = self.encoder.get_sentence_embedding_dimension()
        self.k_hops = k_hops  
        
        if alpha_weights is None:
            self.alpha_weights = [1.0 / (self.k_hops + 1)] * (self.k_hops + 1)
        else:
            self.alpha_weights = alpha_weights

        # Data parsing
        if isinstance(json_data, str):
            self.intel_sets = json.loads(json_data)
        else:
            self.intel_sets = json_data

        self.tools = self._extract_unique_tools(self.intel_sets)
        self.tool_name_to_idx = {t['tool_name']: i for i, t in enumerate(self.tools)}

        self.num_tools = len(self.tools)
        self.num_sets = len(self.intel_sets)
        self.total_nodes = self.num_tools + self.num_sets

        self._prepare_initial_features()
        self._build_symmetric_norm_adjacency()

        # Initialize adapter and optimizer
        self.adapter = LightGCNAdapter(self.embed_dim, hidden_dim=self.embed_dim).to(self.device)
        self.optimizer = optim.AdamW(self.adapter.parameters(), lr=1e-4, weight_decay=1e-4)

        # Ensure adapter is trained and features are exported during initialization
        self.train_adapter(epochs=50)

    def _extract_unique_tools(self, intel_sets):
        unique_tools = {}
        for item in intel_sets:
            for tool in item.get('tools', []):
                name = tool.get('tool_name')
                if name and name not in unique_tools:
                    unique_tools[name] = tool
        return list(unique_tools.values())

    def _prepare_initial_features(self):
        cache_dir = CURRENT_DIR / "cache"
        cache_dir.mkdir(exist_ok=True)
        emb_cache_file = cache_dir / "lightgcn_initial_features.pt"

        if emb_cache_file.exists():
            print(f"[{self.__class__.__name__}] Loading node features from cache...")
            self.X_initial = torch.load(emb_cache_file, map_location=self.device)
        else:
            print(f"[{self.__class__.__name__}] Encoding node features...")
            tool_texts = [t.get('description', '') for t in self.tools]
            set_texts = [item.get('description', '') for item in self.intel_sets]

            X_tools = self.encoder.encode(tool_texts, convert_to_tensor=False)
            X_sets = self.encoder.encode(set_texts, convert_to_tensor=False)

            X_combined = np.vstack([X_tools, X_sets])
            self.X_initial = torch.tensor(X_combined, dtype=torch.float32).to(self.device)
            
            torch.save(self.X_initial, emb_cache_file)

    def _build_symmetric_norm_adjacency(self):
        sources, targets = [], []
        for set_idx, item in enumerate(self.intel_sets):
            for tool in item.get('tools', []):
                t_idx = self.tool_name_to_idx.get(tool.get('tool_name'))
                if t_idx is None: continue
                s_idx = self.num_tools + set_idx
                sources.extend([t_idx, s_idx])
                targets.extend([s_idx, t_idx])

        indices = torch.tensor([sources, targets], dtype=torch.long)
        values = torch.ones(len(sources), dtype=torch.float32)

        degrees = torch.bincount(indices[0], minlength=self.total_nodes).float()
        degrees = torch.clamp(degrees, min=1.0)
        d_inv_sqrt = 1.0 / torch.sqrt(degrees)
        
        norm_values = values * d_inv_sqrt[indices[0]] * d_inv_sqrt[indices[1]]

        self.A_tilde = torch.sparse_coo_tensor(
            indices, norm_values, (self.total_nodes, self.total_nodes)
        ).to(self.device)

    def _lightgcn_propagate(self, E_0):
        all_layer_embeddings = [E_0]
        E_k = E_0
        
        for k in range(self.k_hops):
            E_k = torch.sparse.mm(self.A_tilde, E_k)
            all_layer_embeddings.append(E_k)
            
        E_final = torch.zeros_like(E_0)
        for i, emb in enumerate(all_layer_embeddings):
            E_final += self.alpha_weights[i] * emb
            
        return E_final

    def _export_final_embeddings(self):
        """Exports precomputed graph embeddings for the projector."""
        self.adapter.eval()
        with torch.no_grad():
            E_0 = self.adapter(self.X_initial)
            E_final = self._lightgcn_propagate(E_0)
            
            set_emb_final = E_final[self.num_tools:]
            set_norm_final = F.normalize(set_emb_final, dim=1)
            
            cache_dir = CURRENT_DIR / "cache"
            save_path = cache_dir / "lightgcn_embeddings.npy"
            
            np.save(save_path, set_norm_final.cpu().numpy())
            print(f"[LightGCN-Planner] Final graph features exported to: {save_path}")

    def train_adapter(self, epochs=50):
        cache_dir = CURRENT_DIR / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / "lightgcn_weights.pth"
        
        if cache_file.exists():
            print(f"\n[LightGCN-Planner] Loading weights from cache: {cache_file}")
            self.adapter.load_state_dict(torch.load(cache_file, map_location=self.device))
            self.adapter.eval()
            self._export_final_embeddings()
            return

        print(f"\n[LightGCN-Planner] Training Adapter for {epochs} epochs...")
        self.adapter.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            E_0 = self.adapter(self.X_initial)
            E_final = self._lightgcn_propagate(E_0)
            
            tools_emb = E_final[:self.num_tools]
            sets_emb = E_final[self.num_tools:]
            
            indices = self.A_tilde._indices()
            values = self.A_tilde._values()
            
            mask = (indices[0] < self.num_tools) & (indices[1] >= self.num_tools)
            sub_indices = indices[:, mask].clone()
            sub_indices[1] -= self.num_tools 
            sub_values = values[mask]
            
            A_sub_sparse = torch.sparse_coo_tensor(
                sub_indices, sub_values, 
                (self.num_tools, self.num_sets),
                device=self.device
            )
            pooled_sets = torch.sparse.mm(A_sub_sparse, sets_emb)
            
            loss = 1.0 - F.cosine_similarity(tools_emb, pooled_sets, dim=1).mean()
            l2_reg = 1e-4 * sum(torch.norm(w)**2 for w in self.adapter.parameters())
            total_loss = loss + l2_reg
            
            total_loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:2d} | Loss: {loss.item():.4f}")

        self.adapter.eval()
        torch.save(self.adapter.state_dict(), cache_file)
        self._export_final_embeddings()

    def retrieve(self, query_text, top_k=50):
        self.adapter.eval()
        with torch.no_grad():
            query_np = self.encoder.encode([query_text], normalize_embeddings=True)
            query_vec = torch.tensor(query_np, dtype=torch.float32).to(self.device)
            query_adapted = self.adapter(query_vec)
            query_norm = F.normalize(query_adapted, dim=1)

            E_0 = self.adapter(self.X_initial)
            E_final = self._lightgcn_propagate(E_0)
            
            set_emb_final = E_final[self.num_tools:]
            set_norm_final = F.normalize(set_emb_final, dim=1)

            scores = torch.matmul(query_norm, set_norm_final.T).squeeze(0)

            topk_vals, topk_indices = torch.topk(scores, k=min(top_k, self.num_sets))
            results = []

            for val, local_idx in zip(topk_vals, topk_indices):
                item = self.intel_sets[local_idx.item()]
                results.append({
                    "id": item.get('id'),
                    "score": val.item(),
                    "content": item.get('description', ''),
                    "tools": [t['tool_name'] for t in item.get('tools', [])]
                })

            return results