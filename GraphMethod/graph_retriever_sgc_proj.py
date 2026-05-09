import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from .graph_retriever_sgc import HypergraphSGCPlanner
import torch.nn as nn

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
        x_proj = x + self.mlp(x)
        return torch.nn.functional.normalize(x_proj, p=2, dim=1)

class HypergraphSGCProjPlanner(HypergraphSGCPlanner):
    """
    SGC Planner with a projection head to align text embeddings with graph space.
    """
    def __init__(self, json_data, model_name='BAAI/bge-m3', k_hops=2, alpha=2.5, device=None, proj_weight_path=None):
        super().__init__(json_data, model_name=model_name, k_hops=k_hops, alpha=alpha, device=device)
        
        print(f"\n[SGC-Proj] === Initializing SGC + Projection Head ===")
        
        if proj_weight_path is None:
            current_dir = Path(__file__).parent.resolve()
            proj_weight_path = current_dir / "cache" / "query_projector_sgc_1.pt"
            
        self.projector = QueryProjector().to(self.device)
        if Path(proj_weight_path).exists():
            print(f"[SGC-Proj] Loading projection head weights: {proj_weight_path}")
            self.projector.load_state_dict(torch.load(proj_weight_path, map_location=self.device))
            self.projector.eval()
        else:
            print(f"[SGC-Proj] Warning: Weights not found at {proj_weight_path}. Using random initialization.")

    def retrieve(self, query_text, top_k=50):
        # 1. Basic encoding
        query_vec = self.encoder.encode([query_text], show_progress_bar=False, normalize_embeddings=True)
        
        # 2. Feature mapping via projection head
        with torch.no_grad():
            query_tensor = torch.tensor(query_vec, dtype=torch.float32).to(self.device)
            proj_tensor = self.projector(query_tensor)
            proj_vec = proj_tensor.cpu().numpy()
            
        # 3. Calculate similarity with cached SGC embeddings
        similarities = cosine_similarity(proj_vec, self.final_set_embeddings)[0]
        
        # 4. Rank and return
        k = min(top_k, len(self.intel_sets))
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            item = self.intel_sets[idx]
            results.append({
                "id": item.get('id'),
                "score": float(similarities[idx]),
                "content": item.get('description', ''),
                "tools": [t['tool_name'] for t in item.get('tools', [])]
            })
        return results