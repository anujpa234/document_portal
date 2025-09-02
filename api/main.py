import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime, time

from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparer import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter, read_pdf_via_handler
from utils.token_counter import TokenCounter

# Global token counter instance
token_counter = TokenCounter()

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")  # <--- keep consistent with save_local()

app = FastAPI(title="Document Portal API", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "document-portal"}

# ---------- ANALYZE ----------
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        dh = DocHandler()
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        text = read_pdf_via_handler(dh, saved_path)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# ---------- COMPARE ----------
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    try:
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        _ = ref_path, act_path
        combined_text = dc.combine_documents()
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

# ---------- CHAT: INDEX ----------
@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        wrapped = [FastAPIFileAdapter(f) for f in files]
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        # NOTE: ensure your ChatIngestor saves with index_name="index" or FAISS_INDEX_NAME
        # e.g., if it calls FAISS.save_local(dir, index_name=FAISS_INDEX_NAME)
        ci.built_retriver(  # if your method name is actually build_retriever, fix it there as well
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

# ---------- CHAT: QUERY ----------
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
    use_cache: bool = Form(True), 
    include_cost_info: bool = Form(True),
) -> Any:
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # build retriever + chain
        
        # Track start time for response time analysis
        start_time = time.time()
        
        if use_cache:
            response = rag.invoke_with_memory_and_cache(question)
        else:
            response = rag.invoke_with_memory(question) 
        
        response_time = time.time() - start_time
        
        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG",
            "cached": use_cache,
            "response_time_seconds": round(response_time, 3),
            "model_used": rag.model_name  # Show which model was used
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

# ---------- CHAT: CLEAR HISTORY ----------
@app.post("/chat/clear-history")
async def clear_chat_history(
    session_id: str = Form(...)
) -> Any:
    """Clear conversation history for a session (start fresh)."""
    try:
        # You don't need to load retriever just to clear memory
        rag = ConversationalRAG(session_id=session_id)
        rag.clear_memory()
        
        return {
            "message": "Conversation history cleared successfully",
            "session_id": session_id,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {e}")

# ---------- CHAT: GET CONVERSATION HISTORY ----------
@app.get("/chat/history")
async def get_chat_history(
    session_id: str = Query(...)
) -> Any:
    """Get conversation history for a session."""
    try:
        rag = ConversationalRAG(session_id=session_id)
        history = rag.get_conversation_history()
        
        # Format for frontend display
        formatted_history = []
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                formatted_history.append({
                    "user_message": history[i].content,
                    "ai_response": history[i + 1].content,
                    "timestamp": "recent"  # You could add timestamps to messages
                })
        
        return {
            "history": formatted_history,
            "session_id": session_id,
            "total_exchanges": len(formatted_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {e}")

@app.get("/analytics/model-comparison")
async def get_model_comparison() -> Any:
    """Compare costs and usage between GPT-4o and Gemini-2.0-Flash"""
    try:
        total_costs = token_counter.get_total_costs()
        model_info = token_counter.get_model_info()
        
        # Calculate cost per token for comparison
        comparison = {}
        for model_name, breakdown in total_costs["model_breakdown"].items():
            if breakdown["tokens"] > 0:
                avg_cost_per_token = breakdown["cost"] / breakdown["tokens"]
                avg_cost_per_1k_tokens = avg_cost_per_token * 1000
                
                comparison[model_name] = {
                    "requests": breakdown["requests"],
                    "total_tokens": breakdown["tokens"],
                    "total_cost_usd": round(breakdown["cost"], 6),
                    "avg_cost_per_token": round(avg_cost_per_token, 10),
                    "avg_cost_per_1k_tokens": round(avg_cost_per_1k_tokens, 6),
                    "avg_tokens_per_request": breakdown["tokens"] // breakdown["requests"]
                }
        
        return {
            "model_comparison": comparison,
            "pricing_info": model_info["supported_models"],
            "recommendation": {
                "cheapest_model": min(comparison.keys(), 
                                    key=lambda m: comparison[m]["avg_cost_per_token"]) 
                                if comparison else None,
                "note": "Gemini-2.0-Flash is typically 30x cheaper than GPT-4o"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model comparison: {e}")

@app.get("/analytics/cost-estimate")
async def estimate_costs(
    input_tokens: int = Query(..., description="Estimated input tokens"),
    output_tokens: int = Query(..., description="Estimated output tokens"),
) -> Any:
    """Estimate costs for both models given token counts"""
    try:
        estimates = {}
        
        for model_name in ["gpt-4o", "gemini-2.0-flash"]:
            cost = token_counter.calculate_cost(input_tokens, output_tokens, model_name)
            estimates[model_name] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "estimated_cost_usd": cost,
                "model_description": token_counter.get_model_info()["supported_models"][model_name]["description"]
            }
        
        # Calculate savings
        gpt4o_cost = estimates["gpt-4o"]["estimated_cost_usd"]
        gemini_cost = estimates["gemini-2.0-flash"]["estimated_cost_usd"]
        savings = gpt4o_cost - gemini_cost
        savings_percentage = (savings / gpt4o_cost * 100) if gpt4o_cost > 0 else 0
        
        return {
            "estimates": estimates,
            "comparison": {
                "cost_difference_usd": round(savings, 8),
                "savings_percentage": round(savings_percentage, 2),
                "gemini_vs_gpt4o": f"Gemini is {round(gpt4o_cost / gemini_cost, 1)}x cheaper" if gemini_cost > 0 else "Unable to compare"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to estimate costs: {e}")

# command for executing the fast api
# uvicorn api.main:app --reload    
#uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload