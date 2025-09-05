from langchain.retrievers import BM25Retriever, EnsembleRetriever
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, vectorstore, documents, k=5):
        # Dense retrieval (existing FAISS)
        self.dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Sparse retrieval (BM25)
        texts = [doc.page_content for doc in documents]
        self.bm25_retriever = BM25Retriever.from_texts(texts)
        
        # Ensemble with weights
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]  # 70% dense, 30% sparse
        )
  
from sentence_transformers import CrossEncoder

class ReRankingRetriever:
    def __init__(self, base_retriever, rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.base_retriever = base_retriever
        self.reranker = CrossEncoder(rerank_model)
    
    def retrieve_and_rerank(self, query: str, top_k: int = 5, rerank_top_k: int = 20):
        # Get more documents initially
        docs = self.base_retriever.get_relevant_documents(query)[:rerank_top_k]
        
        # Re-rank using cross-encoder
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        
        # Sort by scores and return top_k
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_docs[:top_k]]
    
class QueryEnhancer:
    def __init__(self, llm):
        self.llm = llm
    
    def generate_multiple_queries(self, original_query: str, num_queries: int = 3):
        prompt = f"""Generate {num_queries} different versions of this query to improve retrieval:
        
        Original: {original_query}
        
        Return only the alternative queries, one per line."""
        
        response = self.llm.invoke(prompt)
        queries = [original_query] + response.strip().split('\n')
        return queries
    
    def query_expansion(self, query: str):
        prompt = f"""Expand this query with relevant synonyms and related terms:
        
        Query: {query}
        
        Expanded query:"""
        
        return self.llm.invoke(prompt).strip()
    
class QueryRouter:
    def __init__(self, llm):
        self.llm = llm
    
    def classify_intent(self, query: str) -> str:
        prompt = f"""Classify this query intent:
        
        Query: {query}
        
        Categories:
        - factual: asking for specific facts
        - analytical: asking for analysis or reasoning
        - comparison: comparing things
        - summary: requesting summaries
        
        Intent:"""
        
        return self.llm.invoke(prompt).strip().lower()
    
    def route_query(self, query: str, intent: str):
        routing_strategies = {
            "factual": {"k": 3, "search_type": "similarity"},
            "analytical": {"k": 8, "search_type": "mmr"},
            "comparison": {"k": 10, "search_type": "similarity_score_threshold"},
            "summary": {"k": 15, "search_type": "mmr"}
        }
        return routing_strategies.get(intent, {"k": 5, "search_type": "similarity"})

from typing import List, Optional, Dict, Any
from sentence_transformers import util

class ContextCompressor:
    def __init__(self, llm):
        self.llm = llm
    
    def compress_context(self, contexts: List[str], query: str, max_tokens: int = 2000) -> str:
        combined_context = "\n\n".join(contexts)
        
        if len(combined_context.split()) <= max_tokens:
            return combined_context
        
        prompt = f"""Compress this context to the most relevant information for the query.
        Keep only information directly related to: {query}
        
        Context: {combined_context}
        
        Compressed context:"""
        
        return self.llm.invoke(prompt)
    
    def filter_relevant_chunks(self, chunks: List[str], query: str, threshold: float = 0.7):
        # Use embeddings to filter only highly relevant chunks
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        chunk_embeddings = model.encode(chunks)
        
        similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
        relevant_indices = [i for i, sim in enumerate(similarities) if sim > threshold]
        
        return [chunks[i] for i in relevant_indices]

class RecursiveRetriever:
    def __init__(self, retriever, llm, max_depth: int = 3):
        self.retriever = retriever
        self.llm = llm
        self.max_depth = max_depth
    
    def recursive_retrieve(self, query: str, depth: int = 0) -> List[str]:
        if depth >= self.max_depth:
            return []
        
        # Initial retrieval
        docs = self.retriever.get_relevant_documents(query)
        contexts = [doc.page_content for doc in docs]
        
        # Check if we need more information
        need_more_info = self.check_completeness(query, contexts)
        
        if need_more_info:
            # Generate follow-up queries
            follow_up_queries = self.generate_followup_queries(query, contexts)
            
            additional_contexts = []
            for follow_up in follow_up_queries:
                additional_contexts.extend(
                    self.recursive_retrieve(follow_up, depth + 1)
                )
            
            contexts.extend(additional_contexts)
        
        return contexts
    
    def check_completeness(self, query: str, contexts: List[str]) -> bool:
        prompt = f"""Given the query and context, is more information needed?
        
        Query: {query}
        Context: {' '.join(contexts[:500])}...
        
        Answer only: YES or NO"""
        
        response = self.llm.invoke(prompt).strip().upper()
        return response == "YES"
    
class AnswerValidator:
    def __init__(self, llm):
        self.llm = llm
    
    def validate_answer(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        validation_prompt = f"""Validate this answer against the context:
        
        Query: {query}
        Answer: {answer}
        Context: {context}
        
        Check for:
        1. Factual accuracy
        2. Context relevance
        3. Completeness
        4. Potential hallucinations
        
        Return JSON:
        {{
            "is_accurate": true/false,
            "confidence": 0.0-1.0,
            "issues": ["list of issues"],
            "suggestions": "improvement suggestions"
        }}"""
        
        return self.llm.invoke(validation_prompt)
    
    def self_correct(self, query: str, answer: str, context: str, validation_result: Dict):
        if not validation_result["is_accurate"] or validation_result["confidence"] < 0.7:
            correction_prompt = f"""The previous answer has issues. Provide a corrected version:
            
            Query: {query}
            Previous Answer: {answer}
            Context: {context}
            Issues: {validation_result['issues']}
            
            Corrected Answer:"""
            
            return self.llm.invoke(correction_prompt)
        
        return answer

from langchain.agents import Tool, create_react_agent

class RAGAgent:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.tools = self._create_tools()
    
    def _create_tools(self):
        return [
            Tool(
                name="search_documents",
                func=self._search_documents,
                description="Search through uploaded documents"
            ),
            Tool(
                name="web_search", 
                func=self._web_search,
                description="Search the web for current information"
            ),
            Tool(
                name="calculate",
                func=self._calculate,
                description="Perform mathematical calculations"
            )
        ]
    
    def _search_documents(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in docs[:3]])
    
    def plan_and_execute(self, complex_query: str):
        planning_prompt = f"""Break down this complex query into steps:
        
        Query: {complex_query}
        
        Available tools: {[tool.name for tool in self.tools]}
        
        Plan (step by step):"""
        
        plan = self.llm.invoke(planning_prompt)
        # Execute plan using agent framework
        return self._execute_plan(plan)
    
class HierarchicalRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.document_summaries = self._create_summaries()
        self.section_index = self._create_section_index()
    
    def _create_summaries(self):
        # Create document-level summaries for routing
        summaries = {}
        for doc_id, doc in self.documents.items():
            summary = self._summarize_document(doc)
            summaries[doc_id] = summary
        return summaries
    
    def hierarchical_retrieve(self, query: str):
        # First, find relevant documents using summaries
        relevant_docs = self._find_relevant_documents(query)
        
        # Then, search within those documents for specific sections
        relevant_sections = []
        for doc_id in relevant_docs:
            sections = self._search_within_document(query, doc_id)
            relevant_sections.extend(sections)
        
        return relevant_sections

