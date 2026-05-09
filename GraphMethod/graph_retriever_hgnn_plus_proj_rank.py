import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import torch.nn as nn
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

class HypergraphHGNNPlusProjRankPlanner(HypergraphHGNNPlusPlanner):
    """
    HGNN+ Advanced Planner with Query Projection and Dynamic Reranking.
    """
    def __init__(self, json_data, model_name='BAAI/bge-m3', knn_k=3, device=None, proj_weight_path=None, recall_n=300):
        super().__init__(json_data, model_name=model_name, knn_k=knn_k, device=device)
        
        self.recall_n = recall_n
        print(f"\n[HGNN+ Proj-Rank] === Initializing HGNN+ with Projector & Dynamic Reranker (Recall Pool: {self.recall_n}) ===")
        
        # Load projection head
        if proj_weight_path is None:
            current_dir = Path(__file__).parent.resolve()
            proj_weight_path = current_dir / "cache" / "query_projector_hgnn_plus.pt"
            
        self.projector = QueryProjector(input_dim=self.embed_dim).to(self.device)
        if Path(proj_weight_path).exists():
            print(f"[HGNN+ Proj-Rank] Successfully loaded projector weights: {proj_weight_path}")
            self.projector.load_state_dict(torch.load(proj_weight_path, map_location=self.device))
            self.projector.eval()
        else:
            print(f"[HGNN+ Proj-Rank] Warning: Projector weights not found. Using random initialization.")

        # Precompute graph embeddings
        if not hasattr(self, 'final_graph_embeddings'):
            print("[HGNN+ Proj-Rank] Triggering global static graph embedding precomputation...")
            self._precompute_graph_embeddings()

        # Compute raw text features for fine-grained reranking
        print("[HGNN+ Proj-Rank] Encoding raw text features (descriptions + tools)...")
        texts = [item.get('description', '') + " " + " ".join([t['description'] for t in item.get('tools', [])]) for item in self.intel_sets]
        self.raw_text_embeddings = self.encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    def retrieve(self, query_text, top_k=50):
        # Prepare query embeddings
        query_vec_raw = self.encoder.encode([query_text], show_progress_bar=False, normalize_embeddings=True)
        
        with torch.no_grad():
            query_tensor = torch.tensor(query_vec_raw, dtype=torch.float32).to(self.device)
            query_vec_proj = self.projector(query_tensor).cpu().numpy()

        # Stage 1: Macro Recall via HGNN+ (Graph-fused features)
        hgnn_similarities = cosine_similarity(query_vec_proj, self.final_graph_embeddings)[0]
        top_n_indices = np.argsort(hgnn_similarities)[::-1][:self.recall_n]
        top_n_hgnn_scores = hgnn_similarities[top_n_indices]

        # Stage 2: Dynamic Weight Calculation (Addressing over-smoothing)
        score_std = np.std(top_n_hgnn_scores[:50]) 
        
        if score_std < 0.014682:
            dynamic_text_weight = 0.8
        elif score_std < 0.017831:
            dynamic_text_weight = 0.5
        else:
            dynamic_text_weight = 0.2

        # Stage 3: Micro Literal Disambiguation (BGE text features)
        candidate_raw_embeddings = self.raw_text_embeddings[top_n_indices]
        text_similarities = cosine_similarity(query_vec_raw, candidate_raw_embeddings)[0]

        # Stage 4: Fusion Scoring and Final Ranking
        # Min-Max Normalization to prevent scale imbalance
        hgnn_norm = (top_n_hgnn_scores - np.min(top_n_hgnn_scores)) / (np.max(top_n_hgnn_scores) - np.min(top_n_hgnn_scores) + 1e-8)
        text_norm = (text_similarities - np.min(text_similarities)) / (np.max(text_similarities) - np.min(text_similarities) + 1e-8)

        final_scores = (1.0 - dynamic_text_weight) * hgnn_norm + dynamic_text_weight * text_norm
        final_reranked_indices = np.argsort(final_scores)[::-1][:top_k]
        
        global_top_k_indices = top_n_indices[final_reranked_indices]
        global_top_k_scores = final_scores[final_reranked_indices]

        # Return results
        results = []
        for rank_idx, global_idx in enumerate(global_top_k_indices):
            item = self.intel_sets[global_idx]
            results.append({
                "id": item.get('id'),
                "score": float(global_top_k_scores[rank_idx]),
                "hgnn_raw_score": float(top_n_hgnn_scores[final_reranked_indices[rank_idx]]),
                "text_raw_score": float(text_similarities[final_reranked_indices[rank_idx]]),
                "content": item.get('description', ''),
                "tools": [t['tool_name'] for t in item.get('tools', [])]
            })
        return results