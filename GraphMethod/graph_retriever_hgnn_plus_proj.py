import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from .graph_retriever_hgnn_plus import HypergraphHGNNPlusPlanner

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

class HypergraphHGNNPlusProjPlanner(HypergraphHGNNPlusPlanner):
    """
    HGNN+ Planner integrated with a Query Projector for enhanced alignment.
    """
    def __init__(self, json_data, model_name='BAAI/bge-m3', knn_k=3, device=None, proj_weight_path=None):
        super().__init__(json_data, model_name=model_name, knn_k=knn_k, device=device)
        
        print(f"\n[HGNN+ Proj] === Initializing HGNN+ with Query Projector ===")
        
        # 1. Precompute and cache global static graph embeddings
        self._precompute_graph_embeddings()
        
        # 2. Load projection head
        if proj_weight_path is None:
            current_dir = Path(__file__).parent.resolve()
            proj_weight_path = current_dir / "cache" / "query_projector_hgnn_plus.pt"
            
        self.projector = QueryProjector(input_dim=self.embed_dim).to(self.device)
        if Path(proj_weight_path).exists():
            print(f"[HGNN+ Proj] Successfully loaded projector weights: {proj_weight_path}")
            self.projector.load_state_dict(torch.load(proj_weight_path, map_location=self.device))
            self.projector.eval()
        else:
            print(f"[HGNN+ Proj] Warning: Projector weights not found at {proj_weight_path}. Using random initialization.")

    def _precompute_graph_embeddings(self):
        """
        Computes fused graph node vectors to optimize online retrieval efficiency.
        """
        print("[HGNN+ Proj] Precomputing and caching global static graph features...")
        self.hgnn_layer.eval()
        with torch.no_grad():
            W = self._get_adaptive_W()
            X_refined = self.hgnn_layer(self.X_initial, self.H, W, self.inv_Dv, self.inv_De)
            
            # Extract topological hyperedge features
            E_topology_feat = torch.sparse.mm(self.H_natural_T, X_refined) * self.inv_De_natural.unsqueeze(1)
            
            # Semantic fusion
            alpha = 0.5 
            E_fused_feat = alpha * E_topology_feat + (1 - alpha) * self.E_initial
            
            # L2 Normalization and conversion to NumPy for fast cosine similarity
            self.final_graph_embeddings = torch.nn.functional.normalize(E_fused_feat, p=2, dim=1).cpu().numpy()

    def export_embeddings(self, save_path):
        """
        Exports graph features to be used as a Target Space for contrastive learning.
        """
        np.save(save_path, self.final_graph_embeddings)
        print(f"[HGNN+ Proj] HGNN+ fused graph embeddings exported to: {save_path}")

    def retrieve(self, query_text, top_k=50):
        # 1. BGE Text Encoding -> 2. Projection Space Mapping
        query_vec_raw = self.encoder.encode([query_text], show_progress_bar=False, normalize_embeddings=True)
        with torch.no_grad():
            query_tensor = torch.tensor(query_vec_raw, dtype=torch.float32).to(self.device)
            query_vec_proj = self.projector(query_tensor).cpu().numpy()
            
        # 3. Compute similarity against cached graph features
        scores = cosine_similarity(query_vec_proj, self.final_graph_embeddings)[0]
        
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