"""
rag_pipeline.py (Colab version)

RETRIEVAL is identical to the local version — ChromaDB doesn't care whether
it's running on your Windows machine or a Colab VM.

GENERATION is different: instead of calling Ollama over HTTP, we load the
model directly into GPU memory with `transformers` and generate in-process.
This is the point of moving to Colab — direct GPU access, no Ollama layer
needed since we're not trying to save/quantize the model, just run it.

MODEL LOADING HAPPENS ONCE, at import time, and is cached in a module-level
variable. Loading a model is slow (several seconds to a minute); you do NOT
want to reload it on every single query, which is a mistake beginners often
make when calling generate() functions repeatedly.
"""

import torch
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer

DRIVE_BASE = "/content/drive/MyDrive/PakGuide_RAG"
DB_PATH = f"{DRIVE_BASE}/chroma_db"
COLLECTION_NAME = "pakguide_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# ---- Model config ----
# Defaults to the BASE model (same one you QLoRA fine-tuned from) so this
# runs out of the box. Once you locate your fine-tuned merged weights on
# Drive, point MODEL_PATH at that folder instead — nothing else in this
# file needs to change, because generate() just calls whatever model
# object gets loaded below.
MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_PATH = "/content/drive/MyDrive/PakGuide/finetuned_merged"  # <-- swap to this once found

_tokenizer = None
_model = None


def _load_model():
    """Lazy singleton loader — model loads on first call, then is reused."""
    global _tokenizer, _model
    if _model is None:
        print(f"Loading {MODEL_PATH} onto {'GPU' if torch.cuda.is_available() else 'CPU'}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        print("Model loaded.")
    return _tokenizer, _model


def _get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Same semantic search as the local version — unchanged by the Colab move."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "instruction": results["metadatas"][0][i]["instruction"],
            "response": results["metadatas"][0][i]["response"],
            "topic": results["metadatas"][0][i].get("topic", ""),
            "distance": results["distances"][0][i],
        })
    return matches


def build_prompt(query: str, context: list[dict]) -> str:
    """Same grounding instruction as before — force the model to rely on
    retrieved context instead of guessing."""
    context_block = "\n\n".join(
        f"Q: {c['instruction']}\nA: {c['response']}" for c in context
    )
    return f"""Use ONLY the reference information below to answer the user's question. If the reference information does not contain the answer, say you don't have that information rather than guessing.

Reference information:
{context_block}

User question: {query}"""


def generate(query: str, context: list[dict]) -> str:
    """Generates via a locally-loaded transformers model instead of Ollama.

    Qwen2.5-Instruct expects CHAT-FORMATTED input (system/user turns), not a
    raw string — that's what apply_chat_template() does. Skipping this step
    is a common bug: the model still runs, but produces noticeably worse
    output because it wasn't trained to see plain prompts, only chat turns.
    """
    tokenizer, model = _load_model()
    user_prompt = build_prompt(query, context)

    messages = [
        {"role": "system", "content": "You are PakGuide, an assistant for Pakistani government and educational procedures (NADRA, FBR, HEC, passport, etc.)."},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the input tokens so we only decode the newly generated reply.
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer(query: str) -> dict:
    """Full pipeline: retrieve -> generate. Called by app.py."""
    context = retrieve(query)
    reply = generate(query, context)
    return {"answer": reply, "sources": context}


if __name__ == "__main__":
    test_query = "How do I get NICOP for overseas Pakistanis?"
    result = answer(test_query)
    print("QUERY:", test_query)
    print("\nRETRIEVED SOURCES:")
    for s in result["sources"]:
        print(f"  - ({s['distance']:.3f}) [{s['topic']}] {s['instruction']}")
    print("\nANSWER:", result["answer"])
