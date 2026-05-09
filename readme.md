# 🚀 Tool Retrieval Pipeline (Provisional Title)

This repository contains the official implementation of the paper **"SPARK: Bridging the Colloquial-Structural Gap in LLM Tool Retrieval via Hypergraph Alignment and Adaptive Reranking"** (Under Review).

Our framework introduces an end-to-end pipeline for **Conversational Tool Retrieval and LLM Selection**, tackling the challenge of accurately aligning complex user queries with hypergraph-structured intelligence sets. The pipeline integrates a customizable Projection Head and dynamic reranking strategies to enhance retrieval precision before LLM reasoning.

> **Note:** The full experimental results, charts, and detailed theoretical analysis will be updated here upon the paper's acceptance.

## 📂 Project Structure

```text
├── BGE_m3/                      # SOTA model weights (download required via ModelScope or Hugging Face)
├── BGE-reranker-v2-m3/          # SOTA model weights (download required via ModelScope or Hugging Face)
├── dataset/                     # Intelligence sets and Evaluation datasets (Challenge & General)
├── prompt/                      # LLM Prompts (e.g., LLMselector_prompt.txt)
├── eval_results/                # Generated reports from the Graph Retrieval stage
├── llm_selection_results/       # Final metrics and logs from the LLM Selector stage
├── cache/                       # Cached node embeddings and projection head weights (.pt)
├── GraphMethod/                 # Collection of Graph-based Retrieval Algorithms (e.g., SGC, LightGCN, HGNN+)
├── main.py                      # Main entry point for the end-to-end pipeline
├── train_projector.py           # Standalone script for training the Query Projection Head
├── requirements.txt             # Environment dependencies
└── .env                         # Environment variables (Dashscope API Keys)
```

## 🛠️ Environment Setup

We recommend using Python 3.9+. Please follow the steps below to set up your environment:

1. **Install PyTorch**
   Install PyTorch according to your system's CUDA version. Please refer to the [official PyTorch website](https://pytorch.org/) for the specific command.
   *Example (for Linux, CUDA 12.1):*

   ```
   pip install torch torchvision torchaudio
   ```
2. **Install Dependencies**

   ```
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables**
   Create a `.env` file in the root directory and add your LLM API Key:

   ```
   dashscope.api_key=your_api_key_here
   ```

## 🚀 Quick Start

The pipeline is fully automated. By specifying a single graph retrieval algorithm via the `--single_test` argument, the script will automatically orchestrate the extraction, training (if needed), retrieval, and LLM reasoning.

### 1. Full Pipeline Execution (with automated Proj-Head training)

To run the full evaluation pipeline using a specific algorithm (e.g., `SGC_Proj_Rank`) and enable the LLM Selector (`qwen3-14b`):

```
python main.py \
    --single_test SGC_Proj_Rank \
    --run_llm \
    --llm_model qwen3-14b
```

*If the projection head weights for `SGC_Proj_Rank` are missing, the script will automatically invoke `train_projector.py` to train and cache them before proceeding to the evaluation.*

### 2. Standalone Projection Head Training

If you wish to train the projection head manually with custom hyper-parameters:

```
python train_projector.py \
    --model_type SGC \
    --graph_cache_path cache/SGC_graph_embeddings.npy \
    --proj_save_path cache/query_projector_SGC_Proj_Rank.pt \
    --epochs 500 \
    --lr 1e-4
```

### 3. Debug Mode

To quickly verify that the pipeline is functioning correctly without iterating over the entire dataset, use the `--debug` flag:

```
python main.py --single_test SGC_Proj_Rank --run_llm --debug --test_num 10
```

## 📊 Evaluation & Metrics

The pipeline automatically generates evaluation reports in two directories:

* `eval_results/`: Contains Top-K retrieval metrics (Recall, MRR, NDCG) in JSON format.
* `llm_selection_results/`: Contains detailed reasoning logs, LLM token usage, and the final Selector Accuracy.

*(Detailed baseline comparisons and visualization dashboards will be released here later.)*

## 📝 License

This project is released under the [MIT License](https://www.google.com/search?q=LICENSE).
