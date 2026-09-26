import argparse
import json
import math
import os
import re
import subprocess
import requests
from typing import Any, Dict, List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.tools import tool

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
    """
    If a document ID is brand new and has never been played, 'total_runs' is '0'. Dividing by zero causes Python to crash with a 
    ZeroDivisionError.
    """
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
# 3. DEfine STRONG OUTPUT SCHEMA (PYDANTIC CONTRACT)
# ==============================================================================

class CodePatch(BaseModel):
    """ 
    Schema representing an exact code patch to be applied to a test file. 
    """
    target_file: str = Field(description="The exact file path containing the broken code. This MUST be either the Target Spec File Path OR one of the file paths listed under 'Imported Page Object / Dependent Files' (e.g., 'pages/MarketingJournoPage.ts').")
    original_snippet: str = Field(description="The exact single line or fragment causing code failure that needs replacing.")
    fixed_snippet: str = Field(description="The correct single line or fragment replacement.")

# ==============================================================================
# 4. Delegates Execution Tooling
# ==============================================================================
@tool
def apply_patch_to_disk(target_file: str, original_snippet: str, fixed_snippet: str, default_content: str) -> str:
    """
    Reads target file from disk, applies exact or fuzzy replacement, and returns modified code.
    """

    # Read target file content from disk if path exists
    if os.path.exists(target_file):
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            print(f"[Warning] Failed to read {target_file}: {e}")

    # Perform exact replacement or fallback to fuzzy snippet replacement
    if original_snippet and original_snippet in file_content:
        return file_content.replace(original_snippet, fixed_snippet)
    
    return apply_fuzzy_snippet_replace(file_content, original_snippet, fixed_snippet)
# ==============================================================================
# 5. LANGGRAPH STATE & NODES
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

    a) The 'processed_query' is converted to a vector, say query vector, and the information of the vectors 
    closest to the query vector in the collection 'playwright_errors' are retrieved such as the 'ids', 'distances',
    'documents', 'metadatas', 'embeddings', 'uris', 'data' and 'included' in the form of dictionary. 
    
    The number of vectors retrieved, along with the documents, metadata and id, are subject to the conditions 
    mentioned.

    The value of the attribute, 'embeddings' will be 'None' simply because that ChromaDB does not include the raw 
    embedding arrays in the payload returned to Python to save memory and network bandwidth.

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
    print(f"[Info] Searching for snippet in {len(lines)} lines of code...")
    for idx, line in enumerate(lines):
        if clean_original in line:
            # Preserve original line's leading whitespace/indentation
            indentation = line[: len(line) - len(line.lstrip())]
            lines[idx] = indentation + fixed.strip()
            return "\n".join(lines)

    print(f"[Warning] Could not match snippet on disk: '{clean_original}'")
    return content

def code_patcher_node(state: SelfHealingState) -> dict:
    # 1. Build Context Strings
    llm_context = [
        f"Historical Fix Pattern: {match['metadata'].get('fix', '')}\nMatched Log Structure: {match['doc']}"
        for match in state.get("retrieved_matches", [])
    ]

    context_str = "\n\n".join(llm_context)
    
    # Load imported page objects
    imported_page_objects = resolve_imported_page_objects(state["spec_file_path"])
    page_objects_context = "\n".join(
        f"\n--- FILE: {path} ---\n{code}\n"
        for path, code in imported_page_objects.items()
    ) or "No imported Page Objects found."

    # 2. System Instruction for LLM
    system_instruction = (
        "You are an automated Playwright repair assistant.\n"
        "Your task is to locate the broken line in either the spec file or imported Page Objects and fix it.\n\n"
        "CRITICAL CONSTRUCTOR RULE:\n"
        "1. If a test fails due to a bad element selector, target the constructor assignment line starting with `this.<propertyName> = page.locator(...)` in the Page Object file.\n"
        "2. NEVER target method execution lines like `await expect(...)` when fixing element locators.\n"
        "3. DO NOT copy comments, header descriptions, or prompt instructions (like `// FIX:`, `// BREAK:`) into `original_snippet` or `fixed_snippet`.\n"
        "4. `original_snippet` MUST be copied EXACTLY character-for-character from the actual source file code."
    )
    
    user_prompt = f"""
        Execution Error Log:
        {state['processed_query'][-1500:]}

        Candidate Code Files to Inspect and Repair:
        --- SPEC FILE: {state['spec_file_path']} ---
        ```typescript
        {state['current_code'][:3500]}
        
        --- IMPORTED PAGE OBJECTS / DEPENDENT FILES ---
        {page_objects_context[:2500]}

        Historical Fix Patterns Reference:
        {context_str}

        Task: Identify which exact candidate file path contains the broken line, set target_file to that file's path, and provide the minimal patch snippet.
        """
    # Pass System & Human messages distinctly for Llama 3
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_prompt)
    ]

    # 3. Native Structured Output Binding (Eliminates regex & json parsing)
    structured_llm = llm.with_structured_output(CodePatch)

    try:
        # LLM output is guaranteed to be a validated CodePatch Pydantic Object
        patch: CodePatch =structured_llm.invoke(messages)
        target_file_path = patch.target_file or state["spec_file_path"]
        original_snippet = patch.original_snippet.strip()
        fixed_snippet = patch.fixed_snippet.strip()
    except Exception as e:
        print(f"[Warning] Structured output extraction failed: {e}. Returning the unpatched code.")
        return {
            "proposed_code": state["current_code"], 
            "target_file_path": state["spec_file_path"]
            }

    # 4. Delegate File Operations to the Isolated tool

    patched_code =apply_patch_to_disk.invoke({
        "target_file": target_file_path,
        "original_snippet": original_snippet,
        "fixed_snippet": fixed_snippet,
        "default_content": state["current_code"]
    })

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
# 6. CONDITIONAL ROUTING & GRAPH BUILDING
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
# 7. CLI EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # Create an instance or object of argparse.ArgumentParser
    parser = argparse.ArgumentParser(description="Playwright Self-Healing Agent")
    # Add arguments using add_argument() method
    parser.add_argument("--spec", required=True, help="Path to the Playwright spec file")
    parser.add_argument("--log", required=True, help="Path to raw failure log file or log text string")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum self-healing attempts")

    args = parser.parse_args()

    # Determine if --log is a file path or raw log text
    if os.path.exists(args.log):
        with open(args.log, "r", encoding="utf-16") as f:
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