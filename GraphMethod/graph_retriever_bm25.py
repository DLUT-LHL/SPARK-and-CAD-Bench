import numpy as np
import jieba
import bm25s

class BM25Retriever:
    """
    BM25 retriever based on sparse matrix acceleration.
    Optimized for high performance with long queries.
    """
    def __init__(self, json_data, **kwargs):
        print("\n[BM25] === Initializing High-Speed BM25 Retriever ===")
        self.intel_sets = json_data
        
        self.corpus_texts = []
        for item in self.intel_sets:
            desc = item.get('description', '')
            tools_desc = " ".join([
                t.get('description', t.get('tool_name', '')) 
                for t in item.get('tools', [])
            ])
            self.corpus_texts.append(f"{desc} {tools_desc}")
            
        print("[BM25] Tokenizing corpus with Jieba...")
        self.corpus_tokens = [jieba.lcut(doc) for doc in self.corpus_texts]
        
        print("[BM25] Building BM25 sparse matrix index...")
        self.retriever = bm25s.BM25()
        self.retriever.index(self.corpus_tokens)
        print(f"[BM25] Initialization complete. Indexed {len(self.corpus_tokens)} intelligence sets.")

    def retrieve(self, query_text, top_k=50):
        # 1. Tokenize query
        query_tokens = jieba.lcut(query_text)
        
        # 2. Fast retrieval
        docs, scores = self.retriever.retrieve([query_tokens], corpus=np.arange(len(self.intel_sets)), k=top_k)
        
        top_k_indices = docs[0]
        top_k_scores = scores[0]
        
        # 3. Format results
        results = []
        for rank_idx, global_idx in enumerate(top_k_indices):
            item = self.intel_sets[global_idx]
            results.append({
                "id": item.get('id', item.get('chain_id')),
                "score": float(top_k_scores[rank_idx]),
                "content": item.get('description', ''),
                "tools": [t.get('tool_name', '') for t in item.get('tools', [])]
            })
            
        return results