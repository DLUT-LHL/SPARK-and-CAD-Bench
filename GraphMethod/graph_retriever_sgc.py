import json
import numpy as np
import os
import time
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from tqdm import tqdm
import torch

# Configuration
CURRENT_DIR = Path(__file__).parent.resolve()
FILE_DATA = CURRENT_DIR.parent / "dataset" / "intelligenceset.json"
CACHE_DIR = CURRENT_DIR / "cache"

class HypergraphSGCPlanner:
    def __init__(self, json_data, model_name='BAAI/bge-m3', k_hops=2, alpha=2.5, device=None):  
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[SGC] === Initializing SGC Retriever (Hops={k_hops}, Alpha={alpha}) on {self.device} ===")
        
        t_start = time.time()
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Initialize text encoder
        model_path_obj = Path(model_name)
        if model_path_obj.exists() and any(model_path_obj.iterdir()):
            self.encoder = SentenceTransformer(str(model_path_obj), device=self.device)
        else:
            default_hf_model = 'BAAI/bge-m3'
            print(f"[GPA-Planner] Local model not found, downloading {default_hf_model} from HuggingFace...")
            self.encoder = SentenceTransformer(default_hf_model, device=self.device)
            self.encoder.save(str(model_path_obj))

        # Data processing
        if isinstance(json_data, str):
            self.intel_sets = json.loads(json_data)
        else:
            self.intel_sets = json_data
            
        self.tools = self._extract_unique_tools(self.intel_sets)
        self.tool_name_to_idx = {t['tool_name']: i for i, t in enumerate(self.tools)}
        
        self.num_tools = len(self.tools)
        self.num_sets = len(self.intel_sets)
        self.total_nodes = self.num_tools + self.num_sets
                
        self.k_hops = k_hops
        self.alpha = alpha
        
        # Build SGC with cache logic
        self._load_or_compute_sgc()
        print(f"[SGC] Init done in {time.time() - t_start:.2f}s")

    def _extract_unique_tools(self, intel_sets):
        unique_tools = {}
        for item in intel_sets:
            for tool in item.get('tools', []):
                t_key = tool.get('tool_name')
                if t_key and t_key not in unique_tools:
                    unique_tools[t_key] = tool
        return list(unique_tools.values())

    def _load_or_compute_sgc(self):
        cache_file = CACHE_DIR / f"sgc_embeddings_k{self.k_hops}_a{self.alpha}.npy"
        
        if cache_file.exists():
            print(f"[SGC] Found cached embeddings at {cache_file}, loading...")
            self.final_set_embeddings = np.load(cache_file)
        else:
            self._precompute_sgc_embeddings()
            print(f"[SGC] Saving embeddings to {cache_file}...")
            np.save(cache_file, self.final_set_embeddings)

    def _precompute_sgc_embeddings(self):
        # 1. Encode all nodes
        print("[SGC] [1/4] Encoding Node Features...")
        tool_texts = [t.get('description', '') for t in self.tools]
        set_texts = [c.get('description', '') for c in self.intel_sets]
        
        X_tools = self.encoder.encode(tool_texts, show_progress_bar=True, normalize_embeddings=True)
        X_sets = self.encoder.encode(set_texts, show_progress_bar=True, normalize_embeddings=True)
        
        self.X_all = np.vstack([X_tools, X_sets])
        
        # 2. Build sparse adjacency matrix
        print("[SGC] [2/4] Building Sparse Adjacency Matrix...")
        rows = []
        cols = []
        
        for j, item in enumerate(self.intel_sets):
            for tool in item.get('tools', []):
                t_key = tool.get('tool_name')
                if t_key in self.tool_name_to_idx:
                    t_idx = self.tool_name_to_idx[t_key]
                    s_idx = self.num_tools + j
                    
                    rows.append(t_idx)
                    cols.append(s_idx)
                    rows.append(s_idx)
                    cols.append(t_idx)
        
        data = np.ones(len(rows), dtype=np.float32)
        A = sp.coo_matrix((data, (rows, cols)), shape=(self.total_nodes, self.total_nodes)).tocsr()
        
        # 3. Normalize S = D^-1 * (A + alpha * I)
        print("[SGC] [3/4] Normalizing Sparse Graph...")
        I = sp.eye(self.total_nodes, format='csr')
        A_tilde = A + (self.alpha * I)
        
        degrees = np.array(A_tilde.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1.0
        
        D_inv = sp.diags(1.0 / degrees, format='csr')
        S = D_inv.dot(A_tilde)
        
        # 4. Feature smoothing propagation
        print(f"[SGC] [4/4] Propagating Features (K={self.k_hops})...")
        current_X = self.X_all
        
        for step in range(self.k_hops):
            current_X = S.dot(current_X)
            
        self.final_set_embeddings = current_X[self.num_tools:, :]

    def retrieve(self, query_text, top_k=50):
        query_vec = self.encoder.encode([query_text], show_progress_bar=False, normalize_embeddings=True)
        similarities = cosine_similarity(query_vec, self.final_set_embeddings)[0]
        
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