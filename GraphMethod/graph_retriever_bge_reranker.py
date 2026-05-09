import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from FlagEmbedding import FlagReranker
from .graph_retriever_bge import TextWeightedBGE

class BGERerankerPlanner(TextWeightedBGE):
    """
    Two-stage retrieval pipeline:
    Stage 1: BGE-M3 Vector Retrieval (Recall Top-N)
    Stage 2: BGE-Reranker-v2-m3 Cross-Encoder (Reranking)
    """
    def __init__(self, json_data, model_name=None, reranker_model_name=None, recall_n=100, device=None, **kwargs):
        super().__init__(json_data, model_name=model_name, device=device, **kwargs)
        
        self.recall_n = recall_n
        
        # Determine Reranker path
        if reranker_model_name is None:
            current_dir = Path(__file__).parent.parent.resolve()
            reranker_path = current_dir / "bge-reranker-v2-m3"
        else:
            reranker_path = Path(reranker_model_name)

        print(f"\n[BGE-Reranker] === Initializing BGE-M3 + Cross-Encoder Reranker ===")
        print(f"[BGE-Reranker] Recall N: {self.recall_n}")
        
        # Load Reranker model
        if reranker_path.exists():
            print(f"[BGE-Reranker] Loading Reranker from: {reranker_path}")
            self.reranker = FlagReranker(str(reranker_path), use_fp16=True, device=self.device)
        else:
            print(f"[BGE-Reranker] Warning: {reranker_path} not found. Attempting to download from HuggingFace...")
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, device=self.device)

        # Prepare text pool for reranking (intelligence set descriptions + tool descriptions)
        self.doc_texts = [
            item.get('description', '') + " " + " ".join([t.get('description', '') for t in item.get('tools', [])]) 
            for item in self.intel_sets
        ]

    def retrieve(self, query_text, top_k=50):
        # Stage 1: Vector-based Recall
        initial_candidates = super().retrieve(query_text, top_k=self.recall_n)
        
        if not initial_candidates:
            return []

        candidate_docs = []
        candidate_indices = []
        for cand in initial_candidates:
            for idx, item in enumerate(self.intel_sets):
                if item.get('id') == cand['id']:
                    candidate_docs.append(self.doc_texts[idx])
                    candidate_indices.append(idx)
                    break

        # Stage 2: Cross-Encoder Scoring
        pairs = [[query_text, doc] for doc in candidate_docs]
        
        with torch.no_grad():
            rerank_scores = self.reranker.compute_score(pairs, batch_size=32)
        
        if isinstance(rerank_scores, float):
            rerank_scores = [rerank_scores]

        # Stage 3: Final Ranking
        scored_results = []
        for i, score in enumerate(rerank_scores):
            original_item = self.intel_sets[candidate_indices[i]]
            scored_results.append({
                "id": original_item.get('id'),
                "score": float(score),
                "content": original_item.get('description', ''),
                "tools": [t['tool_name'] for t in original_item.get('tools', [])]
            })

        scored_results.sort(key=lambda x: x['score'], reverse=True)

        return scored_results[:top_k]