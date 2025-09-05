import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.output_parsers import PydanticOutputParser

from utils.model_loader import ModelLoader
from utils.token_counter import TokenCounter
from utils.rag_evaluator import RAGEvaluator
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType, DocumentAnswer

from typing import List, Optional, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from datetime import datetime, timedelta


from functools import wraps
import hashlib
import json
from typing import Dict, Any
import time

class ResponseCache:
    """Cache for complete RAG responses"""
    
    def __init__(self, ttl_seconds: int = 3600):  # 1 hour default
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
    
    def _generate_key(self, session_id: str, question: str, context_hash: str) -> str:
        """Generate cache key from session + question + recent context"""
        key_data = f"{session_id}:{question}:{context_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_context_hash(self, history: list) -> str:
        """Hash recent conversation context (last 4 messages)"""
        recent_context = history[-4:] if len(history) > 4 else history
        context_str = str([msg.content for msg in recent_context])
        return hashlib.md5(context_str.encode()).hexdigest()[:8]
    
    def get(self, session_id: str, question: str, history: list) -> Any:
        """Get cached response if available and not expired"""
        context_hash = self._get_context_hash(history)
        key = self._generate_key(session_id, question, context_hash)
        
        if key in self.cache:
            cached_item = self.cache[key]
            if time.time() - cached_item['timestamp'] < self.ttl_seconds:
                return cached_item['response']
            else:
                del self.cache[key] # Removing expired key
        return None
    
    def set(self, session_id: str, question: str, history: list, response: Any):
        """Cache the response"""
        context_hash = self._get_context_hash(history)
        key = self._generate_key(session_id, question, context_hash)
        
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }
        
        # Simple cleanup, removing old entries to control the cache in case it becomes too large
        if len(self.cache) > 1000:  # Max 1000 cached responses
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

# Global cache instance
response_cache = ResponseCache(ttl_seconds=1800)  # 30 minutes

class ConversationMemory:
    """
    Manages chat history for conversational RAG sessions.
    """
    
    def __init__(self, max_messages: int = 20, session_timeout_hours: int = 24):
        self.sessions: Dict[str, List[BaseMessage]] = {}
        self.session_timestamps: Dict[str, datetime] = {}
        self.max_messages = max_messages
        self.session_timeout_hours = session_timeout_hours
        self.log = CustomLogger().get_logger(__name__)
    
    def get_history(self, session_id: str) -> List[BaseMessage]:
        """Get chat history for a session."""
        self._cleanup_expired_sessions()
        return self.sessions.get(session_id, [])
    
    def add_exchange(self, session_id: str, user_input: str, ai_response: str):
        """Add a user-AI exchange to session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        # Add user message and AI response
        self.sessions[session_id].extend([
            HumanMessage(content=user_input),
            AIMessage(content=ai_response)
        ])
        
        # Trim to max_messages to manage context window size
        if len(self.sessions[session_id]) > self.max_messages:
            self.sessions[session_id] = self.sessions[session_id][-self.max_messages:]
        
        # Update timestamp
        self.session_timestamps[session_id] = datetime.now()
        
        self.log.info("Added exchange to memory", session_id=session_id, 
                     history_length=len(self.sessions[session_id]))
    
    def clear_session(self, session_id: str):
        """Clear history for a specific session."""
        self.sessions.pop(session_id, None)
        self.session_timestamps.pop(session_id, None)
        self.log.info("Cleared session memory", session_id=session_id)
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        cutoff = datetime.now() - timedelta(hours=self.session_timeout_hours)
        expired_sessions = [
            sid for sid, timestamp in self.session_timestamps.items() 
            if timestamp < cutoff
        ]
        
        for session_id in expired_sessions:
            self.sessions.pop(session_id, None)
            self.session_timestamps.pop(session_id, None)
        
        if expired_sessions:
            self.log.info("Cleaned up expired sessions", count=len(expired_sessions))

# Global memory instance
conversation_memory = ConversationMemory()

# Global token counter instance
token_counter = TokenCounter()

# Global evaluator instance
rag_evaluator = RAGEvaluator()

class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            
            # Memory management
            self.memory = conversation_memory
            
            # Cache management
            self.response_cache = response_cache  
            
            # Evaluation
            self.evaluator = rag_evaluator
            self.enable_evaluation = True

            # Load LLM and prompts once
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Lazy pieces
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            self.log.info("ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            self.log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    def invoke_with_evaluation(self, user_input: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke with memory, cache, token tracking, AND evaluation.
        Returns both answer and evaluation metrics.
        """
        try:
            # Get the normal response
            if hasattr(self, 'invoke_with_memory_and_cache'):
                response = self.invoke_with_memory_and_cache(user_input)
            else:
                response = self.invoke_with_memory(user_input)
            
            # Extract answer and context for evaluation
            answer_text = response.answer if hasattr(response, 'answer') else str(response)
            sources = response.sources if hasattr(response, 'sources') else []
            
            # Get retrieved context (reconstruct from last retrieval)
            contexts = self._get_last_retrieved_contexts(user_input)
            
            # Evaluate if enabled
            evaluation_result = None
            if self.enable_evaluation:
                evaluation_result = self.evaluator.evaluate_response(
                    question=user_input,
                    answer=answer_text,
                    contexts=contexts,
                    session_id=self.session_id,
                    ground_truth=ground_truth
                )
            
            return {
                "answer": response,
                "evaluation": evaluation_result.metrics if evaluation_result else None,
                "evaluation_timestamp": evaluation_result.timestamp.isoformat() if evaluation_result else None
            }
            
        except Exception as e:
            self.log.error("Failed to invoke with evaluation", error=str(e))
            raise DocumentPortalException("Invocation with evaluation error", sys)
    
    def _get_last_retrieved_contexts(self, user_input: str) -> List[str]:
        """Get the contexts that were retrieved for evaluation"""
        try:
            # Simulate retrieval to get contexts (for evaluation)
            if self.retriever:
                # Get rewritten question
                chat_history = self.memory.get_history(self.session_id)
                question_rewriter = (
                    {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                    | self.contextualize_prompt
                    | self.llm
                    | StrOutputParser()
                )
                rewritten_question = question_rewriter.invoke({
                    "input": user_input, 
                    "chat_history": chat_history
                })
                
                # Retrieve documents
                docs = self.retriever.invoke(rewritten_question)
                return [getattr(doc, "page_content", str(doc)) for doc in docs]
            
            return []
            
        except Exception as e:
            self.log.warning("Could not retrieve contexts for evaluation", error=str(e))
            return []

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get evaluation summary for this session"""
        return self.evaluator.get_session_evaluation_summary(self.session_id)
    
    
    def _get_model_name(self) -> str:
        """Extract model name from LLM for cost tracking"""
        try:
            model_name = "unknown"
            
            # based on the attributes it will get model name
            if hasattr(self.llm, 'model_name'):
                model_name = self.llm.model_name
            elif hasattr(self.llm, 'model'):
                model_name = self.llm.model
            elif hasattr(self.llm, '_model_name'):
                model_name = self.llm._model_name
            elif hasattr(self.llm, 'model_id'):
                model_name = self.llm.model_id
            
            # For string representation fallback
            elif hasattr(self.llm, '__class__'):
                class_name = self.llm.__class__.__name__.lower()
                if 'gemini' in class_name:
                    model_name = "gemini-2.0-flash"
                elif 'gpt' in class_name or 'openai' in class_name:
                    model_name = "gpt-4o"
            
            self.log.info(f"Detected model name: {model_name}", session_id=self.session_id)
            return model_name
            
        except Exception as e:
            self.log.warning("Could not extract model name, defaulting to gpt-4o", error=str(e))
            return "gpt-4o"  # Default to GPT-4o
    
    # Method will be invoked for auto memory management
    def invoke_with_memory(self, user_input: str) -> str:
        """
        Automatic memory management 
        """
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            
            # Get existing chat history from memory
            chat_history = self.memory.get_history(self.session_id)
            
            # Invoke the chain with history
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            
            if not answer:
                self.log.warning("No answer generated", user_input=user_input, session_id=self.session_id)
                return "no answer generated."
            
            # Extract answer text (assuming DocumentAnswer model)
            answer_text = answer.answer if hasattr(answer, 'answer') else str(answer)
            
            # Add this exchange to memory
            self.memory.add_exchange(self.session_id, user_input, answer_text)
            
            self.log.info(
                "Chain invoked with memory successfully",
                session_id=self.session_id,
                user_input=user_input,
                history_length=len(chat_history),
                answer_preview=answer_text[:150],
            )
            
            return answer
            
        except Exception as e:
            self.log.error("Failed to invoke ConversationalRAG with memory", error=str(e))
            raise DocumentPortalException("Invocation with memory error", sys)
    
    def invoke_with_memory_and_cache(self, user_input: str) -> str:
        """
        Automatic memory management AND response caching.
        """
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            
            # Get existing chat history from memory
            chat_history = self.memory.get_history(self.session_id)
            
            # Checking the cache first
            cached_response = self.response_cache.get(self.session_id, user_input, chat_history)
            if cached_response is not None:
                self.log.info("Cache hit - returning cached response", 
                            session_id=self.session_id, user_input=user_input[:50])
                
                # Add to memory for conversation continuity
                answer_text = cached_response.answer if hasattr(cached_response, 'answer') else str(cached_response)
                self.memory.add_exchange(self.session_id, user_input, answer_text)
                
                return cached_response
            
            self.log.info("Cache miss - generating new response", 
                        session_id=self.session_id, user_input=user_input[:50])
            
            # Invoke the chain with history
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            
            if not answer:
                self.log.warning("No answer generated", user_input=user_input, session_id=self.session_id)
                return "no answer generated."
            
            # Extract answer text
            answer_text = answer.answer if hasattr(answer, 'answer') else str(answer)
            
            # Cache the response
            self.response_cache.set(self.session_id, user_input, chat_history, answer)
            
            # Add exchange to memory
            self.memory.add_exchange(self.session_id, user_input, answer_text)
            
            self.log.info(
                "Chain invoked with memory and cache successfully",
                session_id=self.session_id,
                user_input=user_input,
                history_length=len(chat_history),
                answer_preview=answer_text[:150],
                cached=False
            )
            
            return answer
            
        except Exception as e:
            self.log.error("Failed to invoke ConversationalRAG with memory and cache", error=str(e))
            raise DocumentPortalException("Invocation with memory and cache error", sys)

    # Clear cache for session, optional if needed
    def clear_cache(self):
        """Clear cached responses for this session (useful when documents are updated)"""
        keys_to_remove = [
            key for key in self.response_cache.cache.keys() 
            if key.startswith(f"{self.session_id}:")
        ]
        for key in keys_to_remove:
            del self.response_cache.cache[key]
        
        self.log.info("Cache cleared for session", session_id=self.session_id)
    
    def clear_memory(self):
        """Clear conversation history for this session."""
        self.memory.clear_session(self.session_id)
        self.log.info("Memory cleared for session", session_id=self.session_id)
        
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get current conversation history."""
        return self.memory.get_history(self.session_id)
    
       
    # ---------- Public API ----------

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load FAISS vectorstore from disk and build retriever + LCEL chain.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,  # ok if you trust the index
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}

            self.retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            self._build_lcel_chain()

            self.log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                k=k,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            self.log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Invoke the LCEL pipeline."""
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                self.log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "no answer generated."
            self.log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )
            return answer
        except Exception as e:
            self.log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    # ---------- Internals ----------

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            self.log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            self.log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            parser = PydanticOutputParser(pydantic_object=DocumentAnswer)
            format_instructions = parser.get_format_instructions()
            
            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                    "format_instructions": lambda _: format_instructions,
                }
                | self.qa_prompt
                | self.llm
                | parser
            )

            self.log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            self.log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)