import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from .graph_retriever_sgc import HypergraphSGCPlanner

class HypergraphSGCRankPlanner(HypergraphSGCPlanner):
    """
    Ablation study version: SGC macro recall + dynamic BGE reranking (No Projector).
    Purpose: To verify the independent contribution of the projection head.
    """
    def __init__(self, json_data, model_name='BAAI/bge-m3', k_hops=1, alpha=2.5, device=None, recall_n=300):
        super().__init__(json_data, model_name=model_name, k_hops=k_hops, alpha=alpha, device=device)
        
        self.recall_n = recall_n
        print(f"\n[SGC-Rank-Ablation] === Initializing SGC + Dynamic Reranker (NO Projector) (Recall Pool: {self.recall_n}) ===")
        
        # Cache raw text embeddings for micro-reranking
        if hasattr(self, 'initial_set_embeddings'):
            print("[SGC-Rank-Ablation] Reusing cached raw text embeddings...")
            self.raw_text_embeddings = self.initial_set_embeddings
        else:
            print("[SGC-Rank-Ablation] Computing raw text embeddings for reranking...")
            texts = [
                item.get('description', '') + " " + " ".join([t['description'] for t in item.get('tools', [])]) 
                for item in self.intel_sets
            ]
            self.raw_text_embeddings = self.encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    def retrieve(self, query_text, top_k=50):
        # Stage 1: SGC Macro Recall (Direct use of raw query vector)
        query_vec_raw = self.encoder.encode([query_text], show_progress_bar=False, normalize_embeddings=True)
        sgc_similarities = cosine_similarity(query_vec_raw, self.final_set_embeddings)[0]
        
        top_n_indices = np.argsort(sgc_similarities)[::-1][:self.recall_n]
        top_n_sgc_scores = sgc_similarities[top_n_indices]

        # Stage 2: Calculate dynamic weight
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

        # Stage 3: Micro Literal Disambiguation
        candidate_raw_embeddings = self.raw_text_embeddings[top_n_indices]
        text_similarities = cosine_similarity(query_vec_raw, candidate_raw_embeddings)[0]

        # Stage 4: Fusion and Final Ranking
        sgc_norm = (top_n_sgc_scores - np.min(top_n_sgc_scores)) / (np.max(top_n_sgc_scores) - np.min(top_n_sgc_scores) + 1e-8)
        text_norm = (text_similarities - np.min(text_similarities)) / (np.max(text_similarities) - np.min(text_similarities) + 1e-8)

        final_scores = (1.0 - dynamic_text_weight) * sgc_norm + dynamic_text_weight * text_norm

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
                "sgc_raw_score": float(top_n_sgc_scores[final_reranked_indices[rank_idx]]),
                "text_raw_score": float(text_similarities[final_reranked_indices[rank_idx]]),
                "content": item.get('description', ''),
                "tools": [t['tool_name'] for t in item.get('tools', [])]
            })
        return results