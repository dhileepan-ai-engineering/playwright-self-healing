import argparse
import json
import math
import os
import re
import subprocess
import requests
from typing import Any, Dict, List, TypedDict

# Suppress HuggingFace Hub warnings and download progress bars
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

# ==============================================================================
# 0. ENVIRONMENT & VARIABLE INITIALIZATION
# ==============================================================================

# Automatically load environment variables from local .env file
load_dotenv()

DATASET_PATH = os.getenv("HISTORICAL_DATA_PATH", "historical_data.json")
MEMORY_FILE_PATH = os.getenv("BANDIT_MEMORY_PATH", "bandit_memory.json")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# ==============================================================================
# 1. CONFIGURATION & VECTOR DB INITIALIZATION
# ==============================================================================

def load_historical_dataset(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

historical_data = load_historical_dataset(DATASET_PATH)

if os.path.exists(MEMORY_FILE_PATH):
    with open(MEMORY_FILE_PATH, "r", encoding="utf-8") as f:
        bandit_memory = json.load(f)
else:
    bandit_memory = {str(i): [1, 1] for i in range(len(historical_data))}

chroma_client = chromadb.PersistentClient(path="./test_failures_db")
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="playwright_errors", 
    embedding_function=embedding_model
)

"""
IMPORTANT: 

While upserting the collection 'playwright_errors', two forms of the documents are stored in the 
vector database, i.e., chromadb. One form is the vector and the other form is the text itself.
"""
if historical_data:
    documents = [f"Step: {item.get('test_step','')} | Log: {item.get('raw_log','')}" for item in historical_data]
    metadatas = [{"test_step": item.get("test_step",""), "fix": item.get("fix","")} for item in historical_data]
    ids = [str(i) for i in range(len(historical_data))]
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

# Check if GROQ_API_KEY is present (GitHub Actions / Cloud)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    print("Using Cloud Llama 3 via Groq API...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0.0
    )
else:
    print("Using Local Llama 3 via Ollama...")
    llm = ChatOllama(
        model="llama3.1",
        temperature=0.0
    )

# ==============================================================================
# 2. CONTEXTUAL BANDIT RERANKER
# ==============================================================================

total_runs = 0

def contextual_bandit_reranker(results: dict, top_k: int = 2, c: float = 1.5) -> List[Dict[str, Any]]:
    global total_runs
    total_runs += 1

    candidate_ids = results['ids'][0] if results['ids'] else []
    candidate_docs = results['documents'][0] if results['documents'] else []
    candidate_metadatas = results['metadatas'][0] if results['metadatas'] else []
    scored_candidates = []

    for idx, doc_id in enumerate(candidate_ids):
        if doc_id not in bandit_memory:
            bandit_memory[doc_id] = [1, 1]

        successes, total_plays = bandit_memory[doc_id]
        safe_plays = max(1, total_plays)
        safe_runs = max(2, total_runs)

        exploitation = successes / safe_plays
        exploration = c * math.sqrt(math.log(safe_runs) / safe_plays)

        scored_candidates.append({
            "id": doc_id,
            "doc": candidate_docs[idx],
            "metadata": candidate_metadatas[idx],
            "score": exploitation + exploration
        })

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scored_candidates[:top_k]

def update_bandit_memory(matches: List[Dict[str, Any]], success: bool) -> None:
    for match in matches:
        doc_id = match.get("id")
        if doc_id in bandit_memory:
            bandit_memory[doc_id][1] += 1  # Total plays
            if success:
                bandit_memory[doc_id][0] += 1  # Successes
    with open(MEMORY_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(bandit_memory, f, indent=2)

# ==============================================================================
# 3. LANGGRAPH STATE & NODES
# ==============================================================================

class SelfHealingState(TypedDict):
    spec_file_path: str
    raw_error_log: str
    current_code: str
    processed_query: str
    retrieved_matches: List[Dict[str, Any]]
    proposed_code: str
    target_file_path: str
    retry_count: int
    max_retries: int
    status: str
    final_message: str

def preprocess_node(state: SelfHealingState) -> dict:
    """
    IMPORTANT:

    The code from the spec.ts and the cleaned version of the raw error log are held by
    the variables 'current_code' and 'processed_query' respectively in the shared state. 
    """
    file_path = state["spec_file_path"]
    code = ""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

    cleaned_log = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z', '', state["raw_error_log"])
    cleaned_log = re.sub(r'\s+', ' ', cleaned_log).strip()

    return {
        "current_code": code,
        "processed_query": cleaned_log
    }

def retrieval_node(state: SelfHealingState) -> dict:
    """
    IMPORTANT: 

    a) The 'processed_query' is converted to a vector, say query vector, and the vectors closest to the
    query vector in the collection 'playwright_errors' are retrieved along with the metadata and id
    in the form of dictionary. 
    
    The number of vectors retrieved, along with the metadata and id, are subject to the conditions mentioned.

    b) The matches contains a list of dictionaries where each dictionary contains the key, 'id', 'doc',
    'metadata' and 'score'.
    """
    if not historical_data:
        return {"retrieved_matches": []}

    results = collection.query(
        query_texts=[state["processed_query"]],
        n_results=min(3, len(historical_data))
    )
    matches = contextual_bandit_reranker(results, top_k=2)
    return {"retrieved_matches": matches}

def resolve_imported_page_objects(spec_file_path: str) -> dict:
    """
    Parses spec_file_path for relative imports (e.g., import { ... } from './pages/MarketingJournoPage')
    and reads the contents of the imported TypeScript files.
    """
    page_objects = {}
    spec_dir = os.path.dirname(spec_file_path)

    with open(spec_file_path, "r", encoding="utf-8") as f:
        spec_content = f.read()

    # Regex matches relative imports like: from './MarketingJournoPage' or from '../pages/MarketingJournoPage'
    import_matches = re.findall(r'from\s+[\'"](\./[^\'"]+|\.\./[^\'"]+)[\'"]', spec_content)

    for import_path in import_matches:
        # Resolve full path on disk
        resolved_base = os.path.normpath(os.path.join(spec_dir, import_path))
        
        # Append .ts extension if missing
        resolved_path = resolved_base if resolved_base.endswith(".ts") else f"{resolved_base}.ts"

        if os.path.exists(resolved_path):
            with open(resolved_path, "r", encoding="utf-8") as pfile:
                page_objects[resolved_path] = pfile.read()

    return page_objects

def apply_fuzzy_snippet_replace(content: str, original: str, fixed: str) -> str:
    """Finds target line regardless of leading/trailing indentation differences."""
    clean_original = original.strip()
    if not clean_original:
        return content

    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if clean_original in line:
            # Preserve original line's leading whitespace/indentation
            indentation = line[: len(line) - len(line.lstrip())]
            lines[idx] = indentation + fixed.strip()
            return "\n".join(lines)

    print(f"[Warning] Could not match snippet on disk: '{clean_original}'")
    return content

def code_patcher_node(state: SelfHealingState) -> dict:
    llm_context = []
    for match in state["retrieved_matches"]:
        llm_context.append(
            f"Historical Fix Pattern: {match['metadata'].get('fix','')}\n"
            f"Matched Log Structure: {match['doc']}"
        )

    context_str = "\n\n".join(llm_context)
    
    # Dynamically discover and load Page Objects imported by this spec
    imported_page_objects = resolve_imported_page_objects(state["spec_file_path"])
    
    page_objects_context = ""
    for path, code in imported_page_objects.items():
        page_objects_context += f"\n--- FILE: {path} ---\n{code}\n"

    # Fallback message if no imported Page Objects were detected
    if not page_objects_context.strip():
        page_objects_context = "No imported Page Objects found."

    # Force SEARCH / REPLACE diff output format
    system_instruction = (
        "You are an automated code repair assistant.\n"
        "Your job is to spot the exact line causing the test error and provide a JSON edit patch.\n"
        "DO NOT explain anything. Respond ONLY with valid JSON.\n\n"
        "REQUIRED JSON OUTPUT FORMAT:\n"
        "{\n"
        '  "target_file": "<file_path_from_spec_or_imported_files>",\n'
        '  "original_snippet": "<EXACT single line or fragment causing code failure>",\n'
        '  "fixed_snippet": "<corrected single line or fragment>"\n'
        "}"
    )
    
    user_prompt = f"""
        Target Spec File Path: {state['spec_file_path']}

        Spec Source Code:
        ```typescript
        {state['current_code'][:3500]}
        ```
        
        Imported Page Object / Dependent Files:
        {page_objects_context[:2500]}

        Execution Error Log:
        {state['processed_query'][-1500:]}

        Historical Fix Patterns Reference:
        {context_str}

        Provide the JSON patch now. No markdown explanations.
    """

    # Pass System & Human messages distinctly for Llama 3
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    raw_content = str(response.content)

    # Extract JSON block from response
    json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    if not json_match:
        print("[Warning] Llama 3 did not return JSON. Returning unpatched code.")
        return {"proposed_code": state["current_code"], "target_file_path": state["spec_file_path"]}

    try:
        patch_data = json.loads(json_match.group(0))
        target_file_path = patch_data.get("target_file", state["spec_file_path"])
        original_snippet = patch_data.get("original_snippet", "").strip()
        fixed_snippet = patch_data.get("fixed_snippet", "").strip()
    except Exception as e:
        print(f"[Warning] Failed to parse JSON patch: {e}")
        return {"proposed_code": state["current_code"], "target_file_path": state["spec_file_path"]}

    # Read target file content from disk
    if os.path.exists(target_file_path):
        with open(target_file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
    else:
        target_file_path = state["spec_file_path"]
        file_content = state["current_code"]

    # Perform exact or line-indentation preserved replacement
    if original_snippet and original_snippet in file_content:
        patched_code = file_content.replace(original_snippet, fixed_snippet)
    else:
        patched_code = apply_fuzzy_snippet_replace(file_content, original_snippet, fixed_snippet)

    return {
        "proposed_code": patched_code,
        "target_file_path": target_file_path
    }

def validation_node(state: SelfHealingState) -> dict:
    proposed_code = state["proposed_code"]
    
    # Write to target_file_path (Page Object OR Spec File) instead of hardcoded spec_file_path
    write_target = state.get("target_file_path") or state["spec_file_path"]

    # Write proposed patch to disk
    with open(write_target, "w", encoding="utf-8") as f:
        f.write(proposed_code)

    # Always re-run the main spec file test suite
    spec_path = state["spec_file_path"]
    test_cmd = f"npx playwright test {spec_path}"
    result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)

    retry_count = state["retry_count"] + 1

    if result.returncode == 0:
        update_bandit_memory(state["retrieved_matches"], success=True)
        return {
            "retry_count": retry_count,
            "status": "SUCCESS",
            "final_message": f"Successfully healed {write_target} on attempt {retry_count}."
    }
    else:
        update_bandit_memory(state["retrieved_matches"], success=False)
        return {
            "retry_count": retry_count,
            "current_code": proposed_code,
            "raw_error_log": result.stderr or result.stdout,
            "status": "FAILED",
            "final_message": f"Attempt {retry_count} failed to resolve test failure."
    }

def notify_node(state: SelfHealingState) -> dict:
    message = (
        f"🤖 Self-Healing Agent Alert\n"
        f"• File: {state['spec_file_path']}\n"
        f"• Status: {state['status']}\n"
        f"• Attempts: {state['retry_count']}/{state['max_retries']}\n"
        f"• Details: {state['final_message']}"
    )

    if SLACK_WEBHOOK_URL:
        try:
            requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=5)
        except Exception as e:
            print(f"[Warning] Failed to send Slack notification: {e}")

    print(f"\n========================================\n{message}\n========================================\n")
    return {}

# ==============================================================================
# 4. CONDITIONAL ROUTING & GRAPH BUILDING
# ==============================================================================
def should_continue(state: SelfHealingState) -> str:
    if state["status"] == "SUCCESS":
        return "notify"
    elif state["retry_count"] >= state["max_retries"]:
        return "notify"
    else:
        return "retrieval"

workflow = StateGraph(SelfHealingState)

workflow.add_node("preprocess", preprocess_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("code_patcher", code_patcher_node)
workflow.add_node("validation", validation_node)
workflow.add_node("notify", notify_node)

workflow.set_entry_point("preprocess")
workflow.add_edge("preprocess", "retrieval")
workflow.add_edge("retrieval", "code_patcher")
workflow.add_edge("code_patcher", "validation")

workflow.add_conditional_edges(
    "validation",
    should_continue,
    {
        "notify": "notify",
        "retrieval": "retrieval"
    }
)

workflow.add_edge("notify", END)
app = workflow.compile()

# ==============================================================================
# 5. CLI EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini-Powered Playwright Self-Healing Agent")
    parser.add_argument("--spec", required=True, help="Path to the Playwright spec or Page Object file")
    parser.add_argument("--log", required=True, help="Path to raw failure log file or log text string")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum self-healing attempts")

    args = parser.parse_args()

    # Determine if --log is a file path or raw log text
    if os.path.exists(args.log):
        with open(args.log, "r", encoding="utf-8") as f:
            error_log_content = f.read()
    else:
        error_log_content = args.log

    initial_state: SelfHealingState = {
        "spec_file_path": args.spec,
        "raw_error_log": error_log_content,
        "current_code": "",
        "processed_query": "",
        "retrieved_matches": [],
        "proposed_code": "",
        "retry_count": 0,
        "max_retries": args.max_retries,
        "status": "INIT",
        "final_message": ""
    }

    app.invoke(initial_state)