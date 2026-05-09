import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import torch.nn as nn
from .graph_retriever_sgc import HypergraphSGCPlanner

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

class HypergraphSGCProjRankPlanner(HypergraphSGCPlanner):
    """
    SGC Retriever with Projection Head and Dynamic Reranking.
    Stage 1: Macro Recall via SGC Graph Embeddings.
    Stage 2: Micro Literal Reranking using raw BGE-M3 semantics.
    """
    def __init__(self, json_data, model_name='BAAI/bge-m3', k_hops=1, alpha=2.5, device=None, proj_weight_path=None, recall_n=300):
        super().__init__(json_data, model_name=model_name, k_hops=k_hops, alpha=alpha, device=device)
        
        self.recall_n = recall_n
        print(f"\n[SGC-Proj-Rank] === Initializing SGC + Proj Head + Dynamic Reranker (Recall Pool: {self.recall_n}) ===")
        
        if proj_weight_path is None:
            current_dir = Path(__file__).parent.resolve()
            proj_weight_path = current_dir / "cache" / "query_projector_sgc_1.pt"
            
        self.projector = QueryProjector().to(self.device)
        if Path(proj_weight_path).exists():
            print(f"[SGC-Proj-Rank] Loading projector weights: {proj_weight_path}")
            self.projector.load_state_dict(torch.load(proj_weight_path, map_location=self.device))
            self.projector.eval()
        else:
            print(f"[SGC-Proj-Rank] Warning: Weight file not found at {proj_weight_path}. Using random initialization.")

        # Cache raw text embeddings for micro-reranking
        if hasattr(self, 'initial_set_embeddings'):
            print("[SGC-Proj-Rank] Reusing cached raw text embeddings...")
            self.raw_text_embeddings = self.initial_set_embeddings
        else:
            print("[SGC-Proj-Rank] Computing raw text embeddings for reranking...")
            texts = [
                item.get('description', '') + " " + " ".join([t['description'] for t in item.get('tools', [])]) 
                for item in self.intel_sets
            ]
            self.raw_text_embeddings = self.encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    def retrieve(self, query_text, top_k=50):
        # 1. Prepare query embeddings
        query_vec_raw = self.encoder.encode([query_text], show_progress_bar=False, normalize_embeddings=True)
        
        with torch.no_grad():
            query_tensor = torch.tensor(query_vec_raw, dtype=torch.float32).to(self.device)
            query_vec_proj = self.projector(query_tensor).cpu().numpy()

        # 2. Stage 1: SGC Macro Recall
        sgc_similarities = cosine_similarity(query_vec_proj, self.final_set_embeddings)[0]
        
        top_n_indices = np.argsort(sgc_similarities)[::-1][:self.recall_n]
        top_n_sgc_scores = sgc_similarities[top_n_indices]

        # 3. Stage 2: Dynamic Weight Calculation
        score_std = np.std(top_n_sgc_scores[:50]) 
        
        w_high = 0.7
        w_low = 0.3
        w_mid = (w_high + w_low) / 2
        s_high = 0.0132
        s_low = 0.0087
        
        if score_std < s_low:
            dynamic_text_weight = w_high
        elif score_std < s_high:
            dynamic_text_weight = w_mid
        else:
            dynamic_text_weight = w_low

        # 4. Stage 3: Micro Literal Disambiguation
        candidate_raw_embeddings = self.raw_text_embeddings[top_n_indices]
        text_similarities = cosine_similarity(query_vec_raw, candidate_raw_embeddings)[0]

        # 5. Stage 4: Fusion and Final Ranking
        # Min-Max Normalization
        sgc_norm = (top_n_sgc_scores - np.min(top_n_sgc_scores)) / (np.max(top_n_sgc_scores) - np.min(top_n_sgc_scores) + 1e-8)
        text_norm = (text_similarities - np.min(text_similarities)) / (np.max(text_similarities) - np.min(text_similarities) + 1e-8)

        final_scores = (1.0 - dynamic_text_weight) * sgc_norm + dynamic_text_weight * text_norm

        final_reranked_indices = np.argsort(final_scores)[::-1][:top_k]
        
        global_top_k_indices = top_n_indices[final_reranked_indices]
        global_top_k_scores = final_scores[final_reranked_indices]

        # 6. Format results
        results = []
        for rank_idx, global_idx in enumerate(global_top_k_indices):
            item = self.intel_sets[global_idx]
            results.append({
                "id": item.get('id'),
                "score": float(global_top_k_scores[rank_idx]),
                "sgc_raw_score": float(top_n_sgc_scores[final_reranked_indices[rank_idx]]),
                "text_raw_score": float(text_similarities[final_reranked_indices[rank_idx]]),
                "content": item.get('description', ''),
                "tools": [t['tool_name'] for t in item.get('tools', [])]
            })
        return results