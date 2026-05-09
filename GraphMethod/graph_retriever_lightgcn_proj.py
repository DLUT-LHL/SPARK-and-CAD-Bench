import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from .graph_retriever_lightgcn import HypergraphLightGCNPlanner

class QueryProjector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        # Residual connection
        x_proj = x + self.mlp(x)
        # L2 Normalization for cosine similarity
        return torch.nn.functional.normalize(x_proj, p=2, dim=1)

class HypergraphLightGCNProjPlanner(HypergraphLightGCNPlanner):
    """
    LightGCN retriever integrated with a projection head for enhanced alignment.
    """
    def __init__(self, json_data, model_name='BAAI/bge-m3', k_hops=3, device=None, alpha_weights=None, proj_weight_path=None):
        super().__init__(json_data, model_name=model_name, k_hops=k_hops, device=device, alpha_weights=alpha_weights)
        
        print(f"\n[LightGCN-Proj] === Initializing LightGCN + Query Projector ===")
        
        # 1. Load adapter and precompute graph embeddings
        self.train_adapter(epochs=50) 
        self.adapter.eval()
        
        with torch.no_grad():
            print("[LightGCN-Proj] Precomputing and caching final graph features...")
            E_0 = self.adapter(self.X_initial)
            E_final = self._lightgcn_propagate(E_0)
            
            # Cache normalized set node embeddings
            self.set_emb_final = E_final[self.num_tools:]
            self.set_norm_final = F.normalize(self.set_emb_final, dim=1)
        
        # 2. Load projection head
        if proj_weight_path is None:
            current_dir = Path(__file__).parent.resolve()
            proj_weight_path = current_dir / "cache" / "query_projector_lightgcn.pt"
            
        self.projector = QueryProjector(input_dim=self.embed_dim).to(self.device)
        
        if Path(proj_weight_path).exists():
            print(f"[LightGCN-Proj] Successfully loaded projector weights: {proj_weight_path}")
            self.projector.load_state_dict(torch.load(proj_weight_path, map_location=self.device))
            self.projector.eval()
        else:
            print(f"[LightGCN-Proj] Warning: Projector weights not found at {proj_weight_path}. Using random initialization.")
            self.projector.eval()

    def retrieve(self, query_text, top_k=50):
        with torch.no_grad():
            # 1. Extract raw query features
            query_np = self.encoder.encode([query_text], normalize_embeddings=True)
            query_vec = torch.tensor(query_np, dtype=torch.float32).to(self.device)
            
            # 2. Project query into the graph semantic space
            query_proj = self.projector(query_vec)
            
            # 3. Compute cosine similarity (1 x Num_Sets)
            scores = torch.matmul(query_proj, self.set_norm_final.T).squeeze(0)

            # 4. Get Top-K results
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