import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

# ================= Global Path Configuration =================
CURRENT_DIR = Path(__file__).parent.resolve()
FILE_DATASET = CURRENT_DIR / "dataset" / "intelligenceset.json"
FILE_TRAIN_CHALLENGE = CURRENT_DIR / "dataset" / "testset_challenge_check_train_valid.json"
FILE_TRAIN_GENERAL = CURRENT_DIR / "dataset" / "testset_general_check_train_valid.json"
EMBEDDING_MODEL = str(CURRENT_DIR / "BGE_m3")

# ================= Network Definition =================
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

# ================= Dataset Definition =================
class RetrievalTrainDataset(Dataset):
    def __init__(self, queries, target_indices):
        self.queries = torch.tensor(queries, dtype=torch.float32)
        self.target_indices = torch.tensor(target_indices, dtype=torch.long)
        
    def __len__(self): return len(self.queries)
    
    def __getitem__(self, idx): return self.queries[idx], self.target_indices[idx]

# ================= Helper Functions =================
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def extract_valid_pairs(data_list, intel_id_to_idx):
    queries, indices = [], []
    for item in data_list:
        t_id = item.get("target_intel_id")
        q_text = item.get("query")
        if t_id in intel_id_to_idx and q_text:
            queries.append(q_text)
            indices.append(intel_id_to_idx[t_id])
    return queries, indices

def encode_and_cache_queries(queries, cache_path, encoder):
    if os.path.exists(cache_path):
        print(f"Cache hit! Loading Query features: {cache_path}")
        return np.load(cache_path)
    
    print(f"Cache miss. Starting BGE-M3 encoding ({len(queries)} items)...")
    embeddings = encoder.encode(queries, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings

# ================= Refactored Core Training Logic =================
def run_projector_training(model_type, graph_cache_path, proj_save_path, query_cache_dir="cache", batch_size=128, epochs=500, lr=1e-4, tau=0.05, patience=5):
    """
    Main training logic encapsulated as a function so it can be called programmatically by main.py
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n=== Start Training Query Projection Head ===")
    print(f"[{model_type}] Target Space | Device: {DEVICE}")
    
    save_dir = Path(proj_save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    intel_data = load_json(FILE_DATASET)
    intel_id_to_idx = {item['id']: idx for idx, item in enumerate(intel_data)}
    
    graph_embeddings = np.load(graph_cache_path)
    graph_tensor = torch.tensor(graph_embeddings, dtype=torch.float32).to(DEVICE)
    
    train_data_challenge = load_json(FILE_TRAIN_CHALLENGE)
    train_data_general = load_json(FILE_TRAIN_GENERAL) 
    
    train_raw = train_data_challenge[:-1000] + train_data_general[:-1000]
    val_raw = train_data_challenge[-1000:] + train_data_general[-1000:]
    
    train_queries_text, train_targets = extract_valid_pairs(train_raw, intel_id_to_idx)
    val_queries_text, val_targets = extract_valid_pairs(val_raw, intel_id_to_idx)
    print(f"Data split complete - Train: {len(train_queries_text)} | Validation: {len(val_queries_text)}")
    
    train_cache_file = os.path.join(query_cache_dir, "train_queries_bge_cache.npy")
    val_cache_file = os.path.join(query_cache_dir, "val_queries_bge_cache.npy")
    
    if os.path.exists(train_cache_file) and os.path.exists(val_cache_file):
        train_queries_emb = encode_and_cache_queries(train_queries_text, train_cache_file, None)
        val_queries_emb = encode_and_cache_queries(val_queries_text, val_cache_file, None)
    else:
        print("\n[System] Initializing BGE-M3 encoder (loaded only once globally)...")
        encoder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        train_queries_emb = encode_and_cache_queries(train_queries_text, train_cache_file, encoder)
        val_queries_emb = encode_and_cache_queries(val_queries_text, val_cache_file, encoder)
        del encoder
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    train_dataset = RetrievalTrainDataset(train_queries_emb, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = RetrievalTrainDataset(val_queries_emb, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = QueryProjector(input_dim=train_queries_emb.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nStarting iterative training (Max Epochs: {epochs}, Patience: {patience})...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_q, batch_idx in train_loader:
            batch_q, batch_idx = batch_q.to(DEVICE), batch_idx.to(DEVICE)
            optimizer.zero_grad()
            
            proj_q = model(batch_q)
            logits = torch.matmul(proj_q, graph_tensor.T) / tau
            loss = criterion(logits, batch_idx)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_q, batch_idx in val_loader:
                batch_q, batch_idx = batch_q.to(DEVICE), batch_idx.to(DEVICE)
                proj_q = model(batch_q)
                logits = torch.matmul(proj_q, graph_tensor.T) / tau
                loss = criterion(logits, batch_idx)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), proj_save_path)
        else:
            patience_counter += 1
            print(f"  [-] Val Loss did not improve, early stopping counter: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\n[!] Early stopping triggered ({patience} consecutive epochs without Val Loss improvement).")
            break
            
    print(f"\nTraining complete! [{model_type}] Best projection head weights saved to: {proj_save_path}")


# ================= Dynamic Argument Parsing for Standalone Exec =================
def parse_args():
    parser = argparse.ArgumentParser(description="Query Projection Head Training Script")
    parser.add_argument("--model_type", type=str, default="SGC", choices=["SGC", "LightGCN", "HGNN_Plus"])
    parser.add_argument("--graph_cache_path", type=str, required=True, help="Graph network feature cache file path (.npy)")
    parser.add_argument("--proj_save_path", type=str, required=True, help="Projection head weights save path (.pt)")
    parser.add_argument("--query_cache_dir", type=str, default="cache", help="Query feature cache directory")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_projector_training(
        model_type=args.model_type,
        graph_cache_path=args.graph_cache_path,
        proj_save_path=args.proj_save_path,
        query_cache_dir=args.query_cache_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        tau=args.tau,
        patience=args.patience
    )