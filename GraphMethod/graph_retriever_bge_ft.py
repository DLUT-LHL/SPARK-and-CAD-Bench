import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from tqdm import tqdm

# Configuration
CURRENT_DIR = Path(__file__).parent.resolve()
FILE_DATA = CURRENT_DIR.parent / "dataset" / "intelligenceset.json"
CACHE_DIR = CURRENT_DIR / "cache"

class TextWeightedBGEFT:
    def __init__(self, json_data, model_name='BAAI/bge-m3', alpha=0.6, device='cpu'):
        self.device = device
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
        
        self.embed_dim = self.encoder.get_sentence_embedding_dimension()

        # Data processing
        if isinstance(json_data, str):
            self.intel_sets = json.loads(json_data)
        else:
            self.intel_sets = json_data
            
        self.tools = self._extract_unique_tools(self.intel_sets)
        self.tool_name_to_idx = {t['tool_name']: i for i, t in enumerate(self.tools)}
        self.alpha = alpha
        
        # Precompute or load embeddings and matrices
        self._load_or_compute_data()

    def _extract_unique_tools(self, intel_sets):
        unique_tools = {}
        for item in intel_sets:
            for tool in item.get('tools', []):
                t_key = tool.get('tool_name')
                if t_key and t_key not in unique_tools:
                    unique_tools[t_key] = tool
        return list(unique_tools.values())

    def _load_or_compute_data(self):
        cache_emb_tools = CACHE_DIR / "baseline_emb_tools.npy"
        cache_emb_sets = CACHE_DIR / "baseline_emb_sets.npy"
        cache_matrix_H = CACHE_DIR / "baseline_matrix_H.npy"
        cache_set_lens = CACHE_DIR / "baseline_set_lengths.npy"

        if (cache_emb_tools.exists() and cache_emb_sets.exists() and 
            cache_matrix_H.exists() and cache_set_lens.exists()):
            print("[Baseline] Found cached data in 'cache/', loading...")
            self.emb_tools = np.load(cache_emb_tools)
            self.emb_sets = np.load(cache_emb_sets)
            self.H = np.load(cache_matrix_H)
            self.set_lengths = np.load(cache_set_lens)
        else:
            self._precompute_embeddings()
            self._build_incidence_matrix()
            
            print("[Baseline] Saving data to 'cache/'...")
            np.save(cache_emb_tools, self.emb_tools)
            np.save(cache_emb_sets, self.emb_sets)
            np.save(cache_matrix_H, self.H)
            np.save(cache_set_lens, self.set_lengths)

    def _precompute_embeddings(self):
        print(f"[Baseline] Encoding features for {len(self.tools)} tools and {len(self.intel_sets)} intel sets...")
        
        tool_texts = [t.get('description', '') for t in self.tools]
        set_texts = [item.get('description', '') for item in self.intel_sets]
        
        self.emb_tools = self.encoder.encode(tool_texts, show_progress_bar=True, normalize_embeddings=True)
        self.emb_sets = self.encoder.encode(set_texts, show_progress_bar=True, normalize_embeddings=True)

    def _build_incidence_matrix(self):
        num_tools = len(self.tools)
        num_sets = len(self.intel_sets)
        
        self.H = np.zeros((num_sets, num_tools), dtype=np.float32)
        self.set_lengths = np.ones(num_sets, dtype=np.float32) # Avoid division by zero
        
        print("[Baseline] Building Incidence Matrix...")
        for j, item in enumerate(tqdm(self.intel_sets, desc="Building Matrix")):
            count = 0
            for tool in item.get('tools', []):
                t_key = tool.get('tool_name')
                if t_key in self.tool_name_to_idx:
                    i = self.tool_name_to_idx[t_key]
                    self.H[j, i] = 1.0
                    count += 1
            if count > 0:
                self.set_lengths[j] = count

    def retrieve(self, query_text, top_k=50):
        # 1. Encode query
        emb_query = self.encoder.encode([query_text], show_progress_bar=False, normalize_embeddings=True)
        
        # 2. Calculate similarity scores
        sim_sets = cosine_similarity(emb_query, self.emb_sets)[0]
        sim_tools = cosine_similarity(emb_query, self.emb_tools)[0]
        
        # 3. Weighted fusion
        sum_tool_scores = np.dot(self.H, sim_tools)
        avg_tool_scores = sum_tool_scores / self.set_lengths
        
        final_scores = (self.alpha * sim_sets) + ((1 - self.alpha) * avg_tool_scores)
        
        # 4. Rank and return results
        k = min(top_k, len(self.intel_sets))
        top_indices = np.argsort(final_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            item = self.intel_sets[idx]
            results.append({
                "id": item.get('id'),
                "score": float(final_scores[idx]),
                "content": item.get('description', ''),
                "tools": [t['tool_name'] for t in item.get('tools', [])]
            })
        return results

if __name__ == "__main__":
    if FILE_DATA.exists():
        with open(FILE_DATA, 'r') as f: 
            data = json.load(f)
        engine = TextWeightedBGEFT(data)
        print("Test Load Success")