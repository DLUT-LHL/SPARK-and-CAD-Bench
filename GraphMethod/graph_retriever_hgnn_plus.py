import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
CURRENT_DIR = Path(__file__).parent.resolve()

class HGNN_Conv_Plus(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_Conv_Plus, self).__init__()
        self.theta = nn.Linear(in_ch, out_ch)

    def forward(self, x, H, W, inv_Dv, inv_De):
        y = torch.sparse.mm(H.t(), x)
        y = inv_De.unsqueeze(1) * y
        y = W.unsqueeze(1) * y
        x_new = torch.sparse.mm(H, y)
        x_new = inv_Dv.unsqueeze(1) * x_new
        return F.gelu(self.theta(x_new))

class HypergraphHGNNPlusPlanner:
    def __init__(self, json_data, model_name='BAAI/bge-m3', knn_k=3, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[HGNN+ SOTA] Initializing on {self.device} with Semantic Fusion")

        # Initialize text encoder
        model_path_obj = Path(model_name)
        
        if model_path_obj.exists() and any(model_path_obj.iterdir()):
            self.encoder = SentenceTransformer(str(model_path_obj), device=self.device)
        else:
            default_hf_model = 'BAAI/bge-m3'
            print(f"[GPA-Planner] Local model not found, downloading {default_hf_model} from HuggingFace...")
            self.encoder = SentenceTransformer(default_hf_model, device=self.device)
            self.encoder.save(str(model_path_obj))
        
        self.embed_dim = self.encoder.get_sentence_embedding_dimension()
        self.intel_sets = json_data if not isinstance(json_data, str) else json.loads(json_data)
        self.knn_k = knn_k 
        
        self.tools = self._extract_unique_tools(self.intel_sets)
        self.tool_idx = {t['tool_name']: i for i, t in enumerate(self.tools)}
        self.num_v = len(self.tools)
        self.num_natural_e = len(self.intel_sets)
        
        self._prepare_features()
        self._build_hyperedge_groups()

        self.hgnn_layer = HGNN_Conv_Plus(self.embed_dim, self.embed_dim).to(self.device)
        self.group_weights = nn.Parameter(torch.zeros(2, device=self.device))
        
        self.optimizer = optim.AdamW([
            {'params': self.hgnn_layer.parameters(), 'lr': 1e-4},
            {'params': [self.group_weights], 'lr': 1e-2} 
        ], weight_decay=1e-4)

    def _precompute_graph_embeddings(self):
        """
        Computes, caches, and exports fused graph node vectors.
        """
        self.hgnn_layer.eval()
        with torch.no_grad():
            W = self._get_adaptive_W()
            X_refined = self.hgnn_layer(self.X_initial, self.H, W, self.inv_Dv, self.inv_De)
            
            E_topology_feat = torch.sparse.mm(self.H_natural_T, X_refined) * self.inv_De_natural.unsqueeze(1)
            
            alpha = 0.5 
            E_fused_feat = alpha * E_topology_feat + (1 - alpha) * self.E_initial
            
            self.final_graph_embeddings = F.normalize(E_fused_feat, p=2, dim=1).cpu().numpy()
            
            cache_dir = CURRENT_DIR / "cache"
            cache_dir.mkdir(exist_ok=True)
            export_path = cache_dir / "hgnn_plus_graph_features.npy"
            np.save(export_path, self.final_graph_embeddings)
            print(f"[HGNN+ SOTA] Graph features precomputed and exported to: {export_path}")

    def _extract_unique_tools(self, intel_sets):
        unique = {}
        for s in intel_sets:
            for t in s.get('tools', []):
                name = t.get('tool_name')
                if name not in unique: 
                    unique[name] = t
        return list(unique.values())

    def _prepare_features(self):
        cache_dir = CURRENT_DIR / "cache"
        cache_dir.mkdir(exist_ok=True)
        v_cache_file = cache_dir / "hgnn_v_feat.pt"
        e_cache_file = cache_dir / "hgnn_e_feat.pt"

        if v_cache_file.exists() and e_cache_file.exists():
            print("[HGNN+ SOTA] Loading tool and hyperedge tensors from cache...")
            self.X_initial = torch.load(v_cache_file, map_location=self.device)
            self.E_initial = torch.load(e_cache_file, map_location=self.device)
        else:
            print("[HGNN+ SOTA] Encoding tool and hyperedge features...")
            # 1. Encode tool features (nodes)
            texts = [t.get('description', '') for t in self.tools]
            v_feat = self.encoder.encode(texts, convert_to_tensor=True, device=self.device)
            self.X_initial = v_feat.float()
            
            # 2. Encode intelligence set features (natural hyperedges)
            set_texts = [item.get('description', '') for item in self.intel_sets]
            e_feat = self.encoder.encode(set_texts, convert_to_tensor=True, device=self.device)
            self.E_initial = e_feat.float()

            torch.save(self.X_initial, v_cache_file)
            torch.save(self.E_initial, e_cache_file)

    def _build_hyperedge_groups(self):
        rows_1, cols_1 = [], []
        for e_idx, s in enumerate(self.intel_sets):
            for t in s.get('tools', []):
                v_idx = self.tool_idx.get(t.get('tool_name'))
                if v_idx is not None:
                    rows_1.append(v_idx)
                    cols_1.append(e_idx)
        
        X_norm = F.normalize(self.X_initial, dim=1)
        sim_matrix = torch.matmul(X_norm, X_norm.t())
        _, topk_indices = torch.topk(sim_matrix, k=self.knn_k, dim=1)
        
        rows_2, cols_2 = [], []
        for i in range(self.num_v):
            for neighbor_idx in topk_indices[i]:
                rows_2.append(neighbor_idx.item())
                cols_2.append(i) 
            
        self.num_knn_e = self.num_v
        self.num_total_e = self.num_natural_e + self.num_knn_e

        cols_2 = [c + self.num_natural_e for c in cols_2]
        
        rows = rows_1 + rows_2
        cols = cols_1 + cols_2
        
        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.ones(len(rows), device=self.device)
        self.H = torch.sparse_coo_tensor(indices, values, (self.num_v, self.num_total_e)).coalesce()
        
        indices_natural = torch.tensor([cols_1, rows_1], dtype=torch.long, device=self.device)
        values_natural = torch.ones(len(rows_1), device=self.device)
        self.H_natural_T = torch.sparse_coo_tensor(indices_natural, values_natural, (self.num_natural_e, self.num_v)).coalesce()

        self.De = torch.clamp(torch.sparse.sum(self.H, dim=0).to_dense(), min=1.0)
        self.Dv = torch.clamp(torch.sparse.sum(self.H, dim=1).to_dense(), min=1.0)
        self.inv_De = 1.0 / self.De
        self.inv_Dv = 1.0 / self.Dv
        
        self.inv_De_natural = self.inv_De[:self.num_natural_e]

    def _get_adaptive_W(self):
        w_norm = torch.sigmoid(self.group_weights)
        w_group1 = w_norm[0].expand(self.num_natural_e)
        w_group2 = w_norm[1].expand(self.num_knn_e)
        W = torch.cat([w_group1, w_group2])
        return W

    def train_adapter(self, epochs=50):
        cache_dir = CURRENT_DIR / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / "hgnn_plus_weights.pth"
        
        if cache_file.exists():
            print(f"\n[HGNN+ SOTA] Loading weights from cache: {cache_file}")
            checkpoint = torch.load(cache_file, map_location=self.device)
            self.hgnn_layer.load_state_dict(checkpoint['hgnn_layer'])
            self.group_weights.data = checkpoint['group_weights']
            self.hgnn_layer.eval()
            self._precompute_graph_embeddings()
            return

        print(f"[HGNN+ SOTA] Training for {epochs} epochs...")
        self.hgnn_layer.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            W = self._get_adaptive_W()
            X_refined = self.hgnn_layer(self.X_initial, self.H, W, self.inv_Dv, self.inv_De)
            loss_anchor = 1.0 - F.cosine_similarity(X_refined, self.X_initial).mean()
            
            total_loss = loss_anchor
            total_loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:2d} | Anchor Loss: {loss_anchor.item():.4f}")

        torch.save({
            'hgnn_layer': self.hgnn_layer.state_dict(),
            'group_weights': self.group_weights.data
        }, cache_file)
        
        self._precompute_graph_embeddings()

    def retrieve(self, query_text, top_k=50):
        if not hasattr(self, 'final_graph_embeddings'):
            self._precompute_graph_embeddings()

        with torch.no_grad():
            q_np = self.encoder.encode([query_text], show_progress_bar=False, normalize_embeddings=True)
            q_vec = q_np.reshape(1, -1)
            
            scores = cosine_similarity(q_vec, self.final_graph_embeddings)[0]
            
            top_val_indices = np.argsort(scores)[::-1][:min(top_k, self.num_natural_e)]
            
            results = []
            for idx in top_val_indices:
                item = self.intel_sets[idx]
                results.append({
                    "id": item.get('id'),
                    "score": float(scores[idx]),
                    "content": item.get('description', ''),
                    "tools": [t['tool_name'] for t in item.get('tools', [])]
                })
        return results