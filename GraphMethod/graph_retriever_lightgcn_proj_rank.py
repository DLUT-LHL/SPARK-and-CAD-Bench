import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

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
        x_proj = x + self.mlp(x)
        return torch.nn.functional.normalize(x_proj, p=2, dim=1)

class HypergraphLightGCNProjRankPlanner(HypergraphLightGCNPlanner):
    """
    Advanced LightGCN retriever integrated with a projection head and dynamic reranking.
    """
    def __init__(self, json_data, model_name='BAAI/bge-m3', k_hops=3, device=None, alpha_weights=None, proj_weight_path=None, recall_n=300):
        super().__init__(json_data, model_name=model_name, k_hops=k_hops, device=device, alpha_weights=alpha_weights)
        
        self.recall_n = recall_n
        print(f"\n[LightGCN-Proj-Rank] === Initializing LightGCN + Proj Head + Dynamic Reranker (Recall Pool: {self.recall_n}) ===")
        
        # 1. Precompute graph features
        self.train_adapter(epochs=50) 
        self.adapter.eval()
        
        with torch.no_grad():
            print("[LightGCN-Proj-Rank] Precomputing and caching final graph features...")
            E_0 = self.adapter(self.X_initial)
            E_final = self._lightgcn_propagate(E_0)
            
            # Normalize and cache set node embeddings
            self.set_emb_final = E_final[self.num_tools:]
            self.set_norm_final_np = F.normalize(self.set_emb_final, dim=1).cpu().numpy()

        # 2. Cache raw text embeddings for reranking
        if hasattr(self, 'initial_set_embeddings'):
            print("[LightGCN-Proj-Rank] Reusing cached raw text embeddings...")
            self.raw_text_embeddings = self.initial_set_embeddings
        else:
            print("[LightGCN-Proj-Rank] Computing raw text embeddings...")
            texts = [item.get('description', '') + " " + " ".join([t['description'] for t in item.get('tools', [])]) for item in self.intel_sets]
            self.raw_text_embeddings = self.encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)

        # 3. Load projection head
        if proj_weight_path is None:
            current_dir = Path(__file__).parent.resolve()
            proj_weight_path = current_dir / "cache" / "query_projector_lightgcn.pt"
            
        self.projector = QueryProjector(input_dim=self.embed_dim).to(self.device)
        
        if Path(proj_weight_path).exists():
            print(f"[LightGCN-Proj-Rank] Successfully loaded projector weights: {proj_weight_path}")
            self.projector.load_state_dict(torch.load(proj_weight_path, map_location=self.device))
            self.projector.eval()
        else:
            print(f"[LightGCN-Proj-Rank] Warning: Projector weights not found at {proj_weight_path}. Using random initialization.")
            self.projector.eval()

    def retrieve(self, query_text, top_k=50):
        # Prepare query embeddings
        query_vec_raw = self.encoder.encode([query_text], show_progress_bar=False, normalize_embeddings=True)
        
        with torch.no_grad():
            query_tensor = torch.tensor(query_vec_raw, dtype=torch.float32).to(self.device)
            query_vec_proj = self.projector(query_tensor).cpu().numpy()

        # Stage 1: Macro Recall via LightGCN
        lightgcn_similarities = cosine_similarity(query_vec_proj, self.set_norm_final_np)[0]
        
        top_n_indices = np.argsort(lightgcn_similarities)[::-1][:self.recall_n]
        top_n_lightgcn_scores = lightgcn_similarities[top_n_indices]

        # Stage 2: Calculate dynamic weight
        score_std = np.std(top_n_lightgcn_scores[:50]) 
        
        if score_std < 0.003663:
            dynamic_text_weight = 0.8
        elif score_std < 0.004495:
            dynamic_text_weight = 0.5
        else:
            dynamic_text_weight = 0.2

        # Stage 3: Micro Literal Disambiguation (Original BGE-M3 semantics)
        candidate_raw_embeddings = self.raw_text_embeddings[top_n_indices]
        text_similarities = cosine_similarity(query_vec_raw, candidate_raw_embeddings)[0]

        # Stage 4: Normalization and Fusion
        lightgcn_ptp = np.max(top_n_lightgcn_scores) - np.min(top_n_lightgcn_scores) + 1e-8
        lightgcn_norm = (top_n_lightgcn_scores - np.min(top_n_lightgcn_scores)) / lightgcn_ptp
        
        text_ptp = np.max(text_similarities) - np.min(text_similarities) + 1e-8
        text_norm = (text_similarities - np.min(text_similarities)) / text_ptp

        final_scores = (1.0 - dynamic_text_weight) * lightgcn_norm + dynamic_text_weight * text_norm

        # Final sorting
        final_reranked_indices = np.argsort(final_scores)[::-1][:top_k]
        global_top_k_indices = top_n_indices[final_reranked_indices]
        global_top_k_scores = final_scores[final_reranked_indices]

        # Format results
        results = []
        for rank_idx, global_idx in enumerate(global_top_k_indices):
            item = self.intel_sets[global_idx]
            results.append({
                "id": item.get('id'),
                "score": float(global_top_k_scores[rank_idx]),
                "lightgcn_raw_score": float(top_n_lightgcn_scores[final_reranked_indices[rank_idx]]),
                "text_raw_score": float(text_similarities[final_reranked_indices[rank_idx]]),
                "content": item.get('description', ''),
                "tools": [t['tool_name'] for t in item.get('tools', [])]
            })
        return results