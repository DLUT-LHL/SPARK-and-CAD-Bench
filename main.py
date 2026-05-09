import os
import sys

IS_CPU_RUNNING = '--cpu_running' in sys.argv
if IS_CPU_RUNNING:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import numpy as np
import time
import torch
import argparse
import gc
import logging
import re
import dashscope
from dashscope import Generation 
from pathlib import Path
from http import HTTPStatus
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# ================= Memory & Threading Config =================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= Directory & Path Configuration =================
CURRENT_DIR = Path(__file__).parent.resolve()

FILE_DATASET = CURRENT_DIR / "dataset" / "intelligenceset.json"
FILE_TEST_CHALLENGE = CURRENT_DIR / "dataset" / "testset_challenge_check_test.json"
FILE_TEST_GENERAL = CURRENT_DIR / "dataset" / "testset_general_check_test.json"

DIR_EVAL_RESULTS = CURRENT_DIR / "eval_results"
DIR_LLM_RESULTS = CURRENT_DIR / "llm_selection_results"
DIR_CACHE = CURRENT_DIR / "cache"  
DIR_EVAL_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_LLM_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_CACHE.mkdir(parents=True, exist_ok=True)

PROMPT_FILE = CURRENT_DIR / "prompt" / "LLMselector_prompt.txt"
ENV_FILE = CURRENT_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE)
dashscope.api_key = os.getenv("dashscope.api_key")

EMBEDDING_MODEL = str(CURRENT_DIR / "BGE_m3")
RETRIEVAL_TOP_K_DEFAULT = 50 

LLM_MODELS_MAPPING = {
    "qwen3-1.7b": "qwen3-1.7b",
    "qwen3-4b": "qwen3-4b",
    "qwen3-8b": "qwen3-8b",
    "qwen3-14b": "qwen3-14b"
}
LLM_MAX_RETRIES = 3
LLM_SAVE_BATCH_SIZE = 10

# ================= Dynamic Graph Retrieval Imports =================
try:
    from GraphMethod.graph_retriever_bm25 import BM25Retriever
    from GraphMethod.graph_retriever_bge import TextWeightedBGE
    from GraphMethod.graph_retriever_bge_ft import TextWeightedBGEFT
    from GraphMethod.graph_retriever_bge_reranker import BGERerankerPlanner
    from GraphMethod.graph_retriever_sgc import HypergraphSGCPlanner
    from GraphMethod.graph_retriever_hgnn_plus import HypergraphHGNNPlusPlanner
    from GraphMethod.graph_retriever_hgnn_plus_proj import HypergraphHGNNPlusProjPlanner
    from GraphMethod.graph_retriever_hgnn_plus_proj_rank import HypergraphHGNNPlusProjRankPlanner
    from GraphMethod.graph_retriever_lightgcn import HypergraphLightGCNPlanner
    from GraphMethod.graph_retriever_lightgcn_proj import HypergraphLightGCNProjPlanner
    from GraphMethod.graph_retriever_lightgcn_proj_rank import HypergraphLightGCNProjRankPlanner
    from GraphMethod.graph_retriever_sgc_proj import HypergraphSGCProjPlanner
    from GraphMethod.graph_retriever_sgc_rank import HypergraphSGCRankPlanner
    from GraphMethod.graph_retriever_sgc_proj_rank import HypergraphSGCProjRankPlanner
except ImportError as e:
    print(f"Error: Failed to import GraphMethod modules. {e}")
    exit(1)

# ================= Registry & Hardware Setup =================
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

def get_device(target_idx):
    if IS_CPU_RUNNING or num_gpus == 0:
        return "cpu"
    return f"cuda:{target_idx % num_gpus}"

GRAPH_DEVICE_DEFAULT = get_device(0)

RETRIEVER_CONFIG = {
    "BM25": {"class": BM25Retriever, "kwargs": {}},
    "BGE": {"class": TextWeightedBGE, "kwargs": {"alpha": 0.6, "device": GRAPH_DEVICE_DEFAULT}},
    "BGE_ft": {"class": TextWeightedBGEFT, "kwargs": {"alpha": 0.6, "device": GRAPH_DEVICE_DEFAULT}},    
    "BGE_Reranker": {"class": BGERerankerPlanner, "kwargs": { "recall_n": 300, "device": get_device(0), "alpha": 0.6}},
    "SGC": {"class": HypergraphSGCPlanner, "kwargs": {"k_hops": 1, "alpha": 2.5, "device": GRAPH_DEVICE_DEFAULT}},
    "HGNN_Plus": {"class": HypergraphHGNNPlusPlanner, "kwargs": {"knn_k": 3, "device": get_device(0)}},
    "HGNN_Plus_Proj": {"class": HypergraphHGNNPlusProjPlanner, "kwargs": {"knn_k": 3, "device": get_device(0)}},
    "HGNN_Plus_Proj_Rank": {"class": HypergraphHGNNPlusProjRankPlanner, "kwargs": {"knn_k": 3, "device": GRAPH_DEVICE_DEFAULT, "recall_n": 300}},
    "LightGCN": {"class": HypergraphLightGCNPlanner, "kwargs": {"k_hops": 3, "device": get_device(0)}},
    "LightGCN_Proj": {"class": HypergraphLightGCNProjPlanner, "kwargs": {"k_hops": 3, "device": get_device(0)}},
    "LightGCN_Proj_Rank": {"class": HypergraphLightGCNProjRankPlanner, "kwargs": {"k_hops": 3, "device": get_device(0), "recall_n": 300}},
    "SGC_Proj": {"class": HypergraphSGCProjPlanner, "kwargs": {"k_hops": 1, "alpha": 2.5, "device": GRAPH_DEVICE_DEFAULT}},
    "SGC_Rank": {"class": HypergraphSGCRankPlanner, "kwargs": {"k_hops": 1, "alpha": 2.5, "device": GRAPH_DEVICE_DEFAULT, "recall_n": 300}, "train_epochs": None},
    "SGC_Proj_Rank": {"class": HypergraphSGCProjRankPlanner, "kwargs": {"k_hops": 1, "alpha": 2.5, "device": GRAPH_DEVICE_DEFAULT, "recall_n": 300}, "train_epochs": None}    
}

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# ================= Argument Parsing =================
def parse_arguments():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation Pipeline for Tool Retrieval & LLM Selection")
    parser.add_argument('--cpu_running', action='store_true', help="Force CPU usage")
    parser.add_argument('--single_test', type=str, required=True, choices=list(RETRIEVER_CONFIG.keys()), help="Specify a single retrieval algorithm to test (Required)")
    parser.add_argument('--test_type', type=str, default='both', choices=['challenge', 'general', 'both'], help="Dataset selection")
    
    # [Fix] 修复了此处 RETRIEVAL_TOP_K_DEFAULT 的拼写错误
    parser.add_argument('--retrieval_top_k', type=int, default=RETRIEVAL_TOP_K_DEFAULT, help="Top-K for graph retrieval")
    
    parser.add_argument('--run_llm', action='store_true', help="Run the LLM selection stage after retrieval")
    parser.add_argument('--llm_model', type=str, default='qwen3-8b', choices=list(LLM_MODELS_MAPPING.keys()), help="LLM model name")
    parser.add_argument('--llm_top_k', type=int, default=10, help="Top-K candidates passed to the LLM selector")
    
    parser.add_argument('--debug', action='store_true', help="Enable debug mode (limits samples)")
    parser.add_argument('--test_num', type=int, default=50, help="Number of samples to process in debug mode")
    
    return parser.parse_args()

# ================= Utils =================
def log_print(msg, color=None, level="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    if level == "info": logging.info(msg)
    elif level == "error": logging.error(msg)
    if color: print(f"{color}{formatted_msg}{Colors.ENDC}")
    else: print(formatted_msg)

def setup_logging():
    log_file = DIR_LLM_RESULTS / "pipeline.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, encoding='utf-8', mode='a')])

def load_json(path):
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def save_json(data, filepath):
    temp_file = filepath.parent / f"{filepath.name}.tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    temp_file.replace(filepath)

def parse_task_data(scenarios_list, intel_dict, desc_text):
    parsed_tasks = []
    for i, scenario in enumerate(tqdm(scenarios_list, desc=desc_text)):
        target_id = scenario.get("target_intel_id") or scenario.get("ground_truth_chain_ids", [None])[0]
        original_query = scenario.get("original_query") or scenario.get("query", scenario.get("golden_sub_tasks", [""])[0])
        test_id = scenario.get("test_id", f"Task_{i}")

        target_desc = scenario.get("ground_truth_description", "")
        if not target_desc and target_id in intel_dict:
            target_desc = intel_dict[target_id].get("description", "")

        task_data = {
            "test_id": test_id, "query": original_query, "target_intel_id": target_id,
            "target_intel_description": target_desc, "methods_result": {}
        }
        parsed_tasks.append(task_data)
    return parsed_tasks

def calculate_metrics(retrieved_items, ground_truth_id, k_list=[5, 10, 20, 50]):
    metrics = {}
    rank = -1
    for idx, item in enumerate(retrieved_items[:max(k_list)]):
        if item.get('id', item.get('chain_id')) == ground_truth_id:
            rank = idx + 1
            break
    for k in k_list:
        hit = 1 if 0 < rank <= k else 0
        mrr = 1.0 / rank if 0 < rank <= k else 0.0
        ndcg = 1.0 / np.log2(rank + 1) if 0 < rank <= k else 0.0
        metrics[f"Recall@{k}"] = hit
        metrics[f"MRR@{k}"] = mrr
        metrics[f"NDCG@{k}"] = ndcg
    return metrics

def print_metrics_table(title, stats_dict, total_samples):
    print(f"\n{Colors.HEADER}=== {title} ==={Colors.ENDC}")
    header = "{:<20} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}"
    print(header.format("Method", "R@50", "R@20", "R@10", "MRR@20", "MRR@10", "MRR@5", "NDCG@20", "NDCG@10", "NDCG@5"))
    print("-" * 130)
    summary_data = {}
    for m, st in stats_dict.items():
        if st["count"] > 0:
            count = st["count"]
            avg_metrics = {k: v / count for k, v in st.items() if k != "count"}
            avg_metrics["Test_Samples"] = count 
            summary_data[m] = avg_metrics
            print(header.format(m, f"{avg_metrics['Recall@50']:.2%}", f"{avg_metrics['Recall@20']:.2%}", f"{avg_metrics['Recall@10']:.2%}", f"{avg_metrics['MRR@20']:.3f}", f"{avg_metrics['MRR@10']:.3f}", f"{avg_metrics['MRR@5']:.3f}", f"{avg_metrics['NDCG@20']:.3f}", f"{avg_metrics['NDCG@10']:.3f}", f"{avg_metrics['NDCG@5']:.3f}"))
    return summary_data

# ================= LLM Functions =================
def extract_json_from_text(text):
    try:
        pattern_block = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern_block, text, re.DOTALL)
        if match: return match.group(1)
        pattern_brace = r"(\{.*\})"
        match = re.search(pattern_brace, text, re.DOTALL)
        if match: return match.group(1)
        return text
    except: return text

def call_llm(user_prompt, model_id):
    messages = [{'role': 'user', 'content': user_prompt}]
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = Generation.call(model=model_id, api_key=dashscope.api_key, messages=messages, result_format='message', temperature=0.1, enable_thinking=False, max_tokens=8000)
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0]['message']['content']
                return {"success": True, "content": extract_json_from_text(content), "raw_content": content, "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
            else:
                log_print(f"API Error [{response.code}]: {response.message}", Colors.WARNING, "warning")
                if "Throttling" in str(response.code): time.sleep(3)
        except Exception as e:
            log_print(f"Request Exception (Attempt {attempt+1}/{LLM_MAX_RETRIES}): {e}", Colors.WARNING, "warning")
            time.sleep(2)
    return {"success": False, "content": "", "input_tokens": 0, "output_tokens": 0}

def process_llm_stage(dataset_name, dataset_path, model_name, model_id, prompt_template, top_k, intel_lookup, retrieval_method, is_debug):
    log_print(f"\n" + "="*50, Colors.HEADER)
    log_print(f">> LLM Selection Stage: Model [{model_name}] | Dataset [{dataset_name}] | Method [{retrieval_method}] | Top-{top_k}", Colors.HEADER)
    
    output_filepath = DIR_LLM_RESULTS / f"{model_name}_{dataset_name}_{retrieval_method}_Top{top_k}.json"
    if not dataset_path.exists():
        log_print(f"Dataset file missing: {dataset_path}", Colors.FAIL)
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f: raw_data = json.load(f)
    test_samples = raw_data.get("details", [])
    
    results_data = {
        "config": {"model_name": model_name, "model_id": model_id, "dataset": dataset_name, "retrieval_method": retrieval_method, "top_k": top_k, "total_samples": len(test_samples)},
        "metrics": {"processed_count": 0, "pipeline_correct_count": 0, "target_in_candidates_count": 0, "selector_correct_count": 0, "total_input_tokens": 0, "total_output_tokens": 0, "pipeline_accuracy": 0.0, "selector_accuracy": 0.0, "avg_input_tokens": 0.0, "avg_output_tokens": 0.0},
        "details": []
    }
    
    processed_ids = set()
    if output_filepath.exists():
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                results_data = saved_data
                processed_ids = {item["test_id"] for item in saved_data.get("details", [])}
            log_print(f"Resuming: Processed {len(processed_ids)} items previously.", Colors.OKGREEN)
        except json.JSONDecodeError: pass

    buffer_count = 0
    for sample in test_samples:
        test_id = sample.get("test_id")
        if test_id in processed_ids: continue
            
        target_intel_id = sample.get("target_intel_id")
        query = sample.get("query")
        
        top_k_candidates = sample.get("methods_result", {}).get(retrieval_method, [])[:top_k]
        target_is_present = any(c.get("id") == target_intel_id for c in top_k_candidates)
        
        if not target_is_present:
            log_print(f"-> [Skip] ID: {test_id} | Target not in Top-{top_k}. Bypassing LLM.", Colors.WARNING)
            llm_response, parsed_json, selected_id, is_correct = {"input_tokens": 0, "output_tokens": 0}, {}, "SKIPPED_NOT_IN_TOPK", False
        else:
            candidate_strs = []
            for item in top_k_candidates:
                c_id = item.get("id", "")
                full_intel_info = intel_lookup.get(c_id, {})
                c_desc = full_intel_info.get("description", item.get("description", ""))
                
                tools_data = full_intel_info.get("tools", [])
                tools_info_list = [f"    - {t.get('tool_name', 'Unknown')}: {t.get('description', '')}" for t in tools_data] if tools_data else [f"    - {t}" for t in item.get("tools", [])]
                formatted_tools = "\n".join(tools_info_list)
                candidate_strs.append(f"ID: {c_id}\nDesc: {c_desc}\nTools:\n{formatted_tools}\n")
                
            candidate_list_text = "\n".join(candidate_strs)
            user_prompt = prompt_template.replace("{query}", query).replace("{candidate_list_text}", candidate_list_text)
            
            log_print(f"Reasoning ID: {test_id} (Target in candidates: {target_is_present}) ...", Colors.CYAN)
            llm_response = call_llm(user_prompt, model_id)
            
            if not llm_response["success"]:
                log_print(f"API Failed or timeout, skipping {test_id}", Colors.FAIL)
                if is_debug: break
                continue
                
            parsed_json, is_correct, selected_id = {}, False, ""
            try:
                parsed_json = json.loads(llm_response["content"])
                selected_id = parsed_json.get("selected_intel_id", "").strip()
                is_correct = (selected_id == target_intel_id)
            except Exception as e:
                log_print(f"JSON Parsing failed: {e}", Colors.FAIL)

        results_data["metrics"]["total_input_tokens"] += llm_response.get("input_tokens", 0)
        results_data["metrics"]["total_output_tokens"] += llm_response.get("output_tokens", 0)
        results_data["metrics"]["processed_count"] += 1
        
        if target_is_present:
            results_data["metrics"]["target_in_candidates_count"] += 1
            if is_correct:
                results_data["metrics"]["selector_correct_count"] += 1
                results_data["metrics"]["pipeline_correct_count"] += 1
                log_print(f"-> [Hit] Correct selection: {selected_id}", Colors.OKGREEN)
            else:
                log_print(f"-> [Miss] Target: {target_intel_id} | Selected: {selected_id}", Colors.FAIL)

        results_data["details"].append({
            "test_id": test_id, "target_is_present_in_topk": target_is_present, "is_correct": is_correct,
            "target_intel_id": target_intel_id, "llm_selected_id": selected_id,
            "tokens": {"input": llm_response.get("input_tokens", 0), "output": llm_response.get("output_tokens", 0)},
            "llm_response_json": parsed_json
        })
        processed_ids.add(test_id)
        buffer_count += 1
        
        m = results_data["metrics"]
        m["pipeline_accuracy"] = m["pipeline_correct_count"] / m["processed_count"] if m["processed_count"] > 0 else 0
        m["selector_accuracy"] = m["selector_correct_count"] / m["target_in_candidates_count"] if m["target_in_candidates_count"] > 0 else 0
        m["avg_input_tokens"] = m["total_input_tokens"] / m["processed_count"] if m["processed_count"] > 0 else 0
        m["avg_output_tokens"] = m["total_output_tokens"] / m["processed_count"] if m["processed_count"] > 0 else 0

        if buffer_count >= LLM_SAVE_BATCH_SIZE:
            save_json(results_data, output_filepath)
            buffer_count = 0
            
        if is_debug:
            log_print(f"Debug mode: Executed single step, exiting LLM loop.", Colors.WARNING)
            break

    save_json(results_data, output_filepath)
    log_print(f"=== LLM Selection Completed: {model_name} on {dataset_name} (Top-{top_k}) ===", Colors.HEADER)
    log_print(f"Pipeline Accuracy: {results_data['metrics']['pipeline_accuracy']:.2%}")
    log_print(f"Selector Accuracy: {results_data['metrics']['selector_accuracy']:.2%}  <-- [Core Paper Metric]")
    print("-" * 50)

# ================= Main Execution Pipeline =================
def run_evaluation_pipeline():
    args = parse_arguments()
    setup_logging()

    debug_suffix = "_debug" if args.debug else ""
    model_prefix = f"_{args.single_test}"
    
    file_challenge = DIR_EVAL_RESULTS / f"final_report_GraphRetrieval_Top{args.retrieval_top_k}{model_prefix}_challenge{debug_suffix}.json"
    file_general = DIR_EVAL_RESULTS / f"final_report_GraphRetrieval_Top{args.retrieval_top_k}{model_prefix}_general{debug_suffix}.json"

    run_challenge = (args.test_type in ['challenge', 'both'])
    run_general = (args.test_type in ['general', 'both'])

    config_to_use = {args.single_test: RETRIEVER_CONFIG[args.single_test]}

    print(f"\n{Colors.OKGREEN}[Mode] Initialization {'Debug' if args.debug else 'Full Zero-Shot Evaluation'}{Colors.ENDC}")

    # ================= Stage 1: Data Preparation =================
    print(f"\n{Colors.HEADER}=== Stage 1: Data Preparation ==={Colors.ENDC}")
    intel_data = load_json(FILE_DATASET)
    intel_dict = {item["id"]: item for item in intel_data}

    challenge_test_tasks = []
    if run_challenge:
        raw_challenge = load_json(FILE_TEST_CHALLENGE)
        challenge_test_tasks = parse_task_data(raw_challenge, intel_dict, "Parsing Challenge Set")
        if args.debug: challenge_test_tasks = challenge_test_tasks[:args.test_num]

    general_test_tasks = []
    if run_general:
        raw_general = load_json(FILE_TEST_GENERAL)
        general_test_tasks = parse_task_data(raw_general, intel_dict, "Parsing General Set")
        if args.debug: general_test_tasks = general_test_tasks[:args.test_num]

    # ================= Stage 2: Graph Retrieval Inference & Proj Training Hook =================
    mode_str = "CPU Mode" if IS_CPU_RUNNING else "GPU Mode"
    print(f"\n{Colors.HEADER}=== Stage 2: Graph Retrieval ({mode_str}) ==={Colors.ENDC}")

    for m_name, config in config_to_use.items():
        if (run_challenge and file_challenge.exists()) and (run_general and file_general.exists()):
            print(f"{Colors.WARNING}[Skip] Evaluation files for retrieval algorithm '{m_name}' already exist.{Colors.ENDC}")
            break
            
        print(f"\n{Colors.OKBLUE}>>> Processing Retrieval Algorithm: {m_name} <<<{Colors.ENDC}")
        kwargs = config["kwargs"].copy()

        # --- Automated Projection Head Pipeline ---
        if "Proj" in m_name:
            proj_weight_path = DIR_CACHE / f"query_projector_{m_name}.pt"
            kwargs["proj_weight_path"] = str(proj_weight_path) 
            
            if not proj_weight_path.exists():
                log_print(f"Projection head weights not found at {proj_weight_path}. Triggering automated training...", Colors.WARNING)
                base_algo_name = m_name.split("_Proj")[0] # e.g. SGC_Proj_Rank -> SGC
                
                if base_algo_name not in RETRIEVER_CONFIG:
                    log_print(f"Base algorithm '{base_algo_name}' not found. Cannot extract graph embeddings.", Colors.FAIL)
                    return
                
                base_config = RETRIEVER_CONFIG[base_algo_name]
                log_print(f"Step 1/2: Initializing Base Model [{base_algo_name}] to extract graph embeddings...", Colors.OKBLUE)
                
                # Init base model to populate graph embeddings
                base_retriever = base_config["class"](intel_data, model_name=EMBEDDING_MODEL, **base_config["kwargs"])
                graph_cache_path = DIR_CACHE / f"{base_algo_name}_graph_embeddings.npy"
                
                # [Fix] 修复此处直接保存 tensor 的隐患，转换为 Numpy 数组
                emb_data = base_retriever.final_set_embeddings
                if torch.is_tensor(emb_data):
                    emb_data = emb_data.detach().cpu().numpy()
                elif isinstance(emb_data, list):
                    emb_data = np.array(emb_data)
                np.save(graph_cache_path, emb_data) 
                
                del base_retriever
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                log_print(f"Step 2/2: Launching Projector Training using extracted features...", Colors.OKGREEN)
                try:
                    from train_projector import run_projector_training
                    run_projector_training(
                        model_type=base_algo_name,
                        graph_cache_path=str(graph_cache_path),
                        proj_save_path=str(proj_weight_path),
                        query_cache_dir=str(DIR_CACHE)
                    )
                except Exception as e:
                    log_print(f"Training failed: {e}", Colors.FAIL)
                    return
            else:
                log_print(f"Found existing Projection Head weights: {proj_weight_path}", Colors.OKGREEN)
        # ------------------------------------------

        retriever = config["class"](intel_data, model_name=EMBEDDING_MODEL, **kwargs)

        gc.collect()
        if not IS_CPU_RUNNING and torch.cuda.is_available(): torch.cuda.empty_cache()

        if run_challenge and not file_challenge.exists():
            for task in tqdm(challenge_test_tasks, desc=f"Challenge Retrieving [{m_name}]"):
                candidates = retriever.retrieve(task["query"], top_k=args.retrieval_top_k)
                task["methods_result"][m_name] = [{"rank": r+1, "id": c.get('id', c.get('chain_id')), "description": c.get('description', c.get('content', '')), "tools": c.get('tools', [])} for r, c in enumerate(candidates)]

        if run_general and not file_general.exists():
            for task in tqdm(general_test_tasks, desc=f"General Retrieving [{m_name}]"):
                candidates = retriever.retrieve(task["query"], top_k=args.retrieval_top_k)
                task["methods_result"][m_name] = [{"rank": r+1, "id": c.get('id', c.get('chain_id')), "description": c.get('description', c.get('content', '')), "tools": c.get('tools', [])} for r, c in enumerate(candidates)]

        del retriever
        gc.collect()
        if not IS_CPU_RUNNING and torch.cuda.is_available(): torch.cuda.empty_cache()

    # Metrics Calc & Report Generation
    all_ks = [5, 10, 20, 50]
    
    if run_challenge and not file_challenge.exists():
        stats_challenge = {m: {"count": 0, **{f"Recall@{k}": 0 for k in all_ks}, **{f"MRR@{k}": 0.0 for k in all_ks}, **{f"NDCG@{k}": 0.0 for k in all_ks}} for m in config_to_use.keys()}
        for task in challenge_test_tasks:
            target_id = task.get("target_intel_id")
            if not target_id: continue
            for m_name in config_to_use.keys():
                if m_name not in task["methods_result"]: continue
                met = calculate_metrics(task["methods_result"][m_name], target_id, k_list=all_ks)
                stats_challenge[m_name]["count"] += 1
                for key, val in met.items(): stats_challenge[m_name][key] += val
        
        summary_challenge = print_metrics_table("Challenge Set Retrieval Eval", stats_challenge, len(challenge_test_tasks))
        final_data_challenge = {"config": {"retrieval_top_k": args.retrieval_top_k, "is_debug_mode": args.debug, "test_set": "Challenge"}, "summary": summary_challenge, "details": challenge_test_tasks}
        with open(file_challenge, 'w', encoding='utf-8') as f: json.dump(final_data_challenge, f, indent=4, ensure_ascii=False)
        print(f"{Colors.OKGREEN}Challenge report saved: {file_challenge}{Colors.ENDC}")

    if run_general and not file_general.exists():
        stats_general = {m: {"count": 0, **{f"Recall@{k}": 0 for k in all_ks}, **{f"MRR@{k}": 0.0 for k in all_ks}, **{f"NDCG@{k}": 0.0 for k in all_ks}} for m in config_to_use.keys()}
        for task in general_test_tasks:
            target_id = task.get("target_intel_id")
            if not target_id: continue
            for m_name in config_to_use.keys():
                if m_name not in task["methods_result"]: continue
                met = calculate_metrics(task["methods_result"][m_name], target_id, k_list=all_ks)
                stats_general[m_name]["count"] += 1
                for key, val in met.items(): stats_general[m_name][key] += val
                
        summary_general = print_metrics_table("General Set Retrieval Eval", stats_general, len(general_test_tasks))
        final_data_general = {"config": {"retrieval_top_k": args.retrieval_top_k, "is_debug_mode": args.debug, "test_set": "General"}, "summary": summary_general, "details": general_test_tasks}
        with open(file_general, 'w', encoding='utf-8') as f: json.dump(final_data_general, f, indent=4, ensure_ascii=False)
        print(f"{Colors.OKGREEN}General report saved: {file_general}{Colors.ENDC}")

    # ================= Stage 3: LLM Selector Inference =================
    if args.run_llm:
        print(f"\n{Colors.HEADER}=== Stage 3: LLM Selector Selection ==={Colors.ENDC}")
        if not PROMPT_FILE.exists():
            log_print(f"Fatal Error: Prompt file missing {PROMPT_FILE}", Colors.FAIL)
            return

        with open(PROMPT_FILE, 'r', encoding='utf-8') as f: prompt_template = f.read()
            
        model_id = LLM_MODELS_MAPPING.get(args.llm_model)
        if not model_id: return

        method_name = args.single_test 
        datasets_to_run = {}
        if run_challenge: datasets_to_run["challenge"] = file_challenge
        if run_general: datasets_to_run["general"] = file_general

        for dataset_name, dataset_path in datasets_to_run.items():
            process_llm_stage(
                dataset_name=dataset_name, dataset_path=dataset_path, model_name=args.llm_model, model_id=model_id,
                prompt_template=prompt_template, top_k=args.llm_top_k, intel_lookup=intel_dict, retrieval_method=method_name, is_debug=args.debug
            )
    else:
        print(f"\n{Colors.OKBLUE}[System] Skipping LLM Selector Stage. Use --run_llm to enable.{Colors.ENDC}")

    print(f"\n{Colors.OKBLUE}Complete Pipeline Execution Finished Successfully.{Colors.ENDC}")

if __name__ == "__main__":
    run_evaluation_pipeline()