from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy, 
        context_precision,
        context_recall
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

from logger.custom_logger import CustomLogger

class EvaluationMetric(str, Enum):
    FAITHFULNESS = "faithfulness"
    RELEVANCY = "answer_relevancy"
    PRECISION = "context_precision"
    RECALL = "context_recall"
    CUSTOM_SCORE = "custom_score"

@dataclass
class EvaluationResult:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    metrics: Dict[str, float]
    timestamp: datetime
    session_id: str

class RAGEvaluator:
    """Evaluation system for RAG pipeline"""
    
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        self.evaluation_history: List[EvaluationResult] = []
    
    def evaluate_response(
        self, 
        question: str,
        answer: str, 
        contexts: List[str],
        session_id: str,
        ground_truth: Optional[str] = None,
        use_ragas: bool = True
    ) -> EvaluationResult:
        """Evaluate a single RAG response"""
        
        metrics = {}
        
        if use_ragas and RAGAS_AVAILABLE:
            metrics.update(self._evaluate_with_ragas(question, answer, contexts, ground_truth))
        
        # Custom evaluations
        metrics.update(self._evaluate_custom(question, answer, contexts))
        
        result = EvaluationResult(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            metrics=metrics,
            timestamp=datetime.now(),
            session_id=session_id
        )
        
        self.evaluation_history.append(result)
        
        self.log.info("Response evaluated", 
                     session_id=session_id,
                     metrics=metrics,
                     question_preview=question[:50])
        
        return result
    
    def _evaluate_with_ragas(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str],
        ground_truth: Optional[str]
    ) -> Dict[str, float]:
        """Evaluate using RAGAS framework"""
        try:
            # Prepare data for RAGAS
            data = {
                "question": [question],
                "answer": [answer], 
                "contexts": [contexts],
            }
            
            if ground_truth:
                data["ground_truth"] = [ground_truth]
            
            dataset = Dataset.from_dict(data)
            
            # Select metrics based on available data
            metrics = [faithfulness, answer_relevancy, context_precision]
            if ground_truth:
                metrics.append(context_recall)
            
            # Run evaluation
            result = evaluate(dataset, metrics=metrics)
            
            return {
                "faithfulness": float(result["faithfulness"]),
                "answer_relevancy": float(result["answer_relevancy"]),
                "context_precision": float(result["context_precision"]),
                **({"context_recall": float(result["context_recall"])} if ground_truth else {})
            }
            
        except Exception as e:
            self.log.warning("RAGAS evaluation failed", error=str(e))
            return {}
    
    def _evaluate_custom(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """Custom evaluation metrics"""
        metrics = {}
        
        # Answer length score
        answer_length = len(answer.split())
        metrics["answer_length_score"] = min(1.0, answer_length / 100)  # Optimal ~100 words
        
        # Context utilization score  
        context_words = set()
        for ctx in contexts:
            context_words.update(ctx.lower().split())
        
        answer_words = set(answer.lower().split())
        overlap = len(context_words.intersection(answer_words))
        metrics["context_utilization"] = overlap / len(context_words) if context_words else 0
        
        # Question-answer relevance (simple keyword overlap)
        question_words = set(question.lower().split())
        question_overlap = len(question_words.intersection(answer_words))
        metrics["keyword_relevance"] = question_overlap / len(question_words) if question_words else 0
        
        # Confidence in sources
        source_mentions = answer.lower().count("source") + answer.lower().count("document")
        metrics["source_citation_score"] = min(1.0, source_mentions / 2)  # Up to 2 mentions = 1.0
        
        return metrics
    
    def get_session_evaluation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get evaluation summary for a session"""
        session_evals = [e for e in self.evaluation_history if e.session_id == session_id]
        
        if not session_evals:
            return {"message": "No evaluations found for session", "session_id": session_id}
        
        # Aggregate metrics
        all_metrics = {}
        for eval_result in session_evals:
            for metric, value in eval_result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate averages
        avg_metrics = {
            metric: sum(values) / len(values) 
            for metric, values in all_metrics.items()
        }
        
        return {
            "session_id": session_id,
            "total_evaluations": len(session_evals),
            "average_metrics": avg_metrics,
            "latest_evaluation": session_evals[-1].metrics,
            "evaluation_trend": self._calculate_trend(session_evals)
        }
    
    def _calculate_trend(self, evaluations: List[EvaluationResult]) -> str:
        """Calculate if evaluation metrics are improving or declining"""
        if len(evaluations) < 2:
            return "insufficient_data"
        
        # Compare first half vs second half
        mid = len(evaluations) // 2
        first_half = evaluations[:mid]
        second_half = evaluations[mid:]
        
        first_avg = self._average_score(first_half)
        second_avg = self._average_score(second_half)
        
        if second_avg > first_avg + 0.05:
            return "improving"
        elif second_avg < first_avg - 0.05:
            return "declining" 
        else:
            return "stable"
    
    def _average_score(self, evaluations: List[EvaluationResult]) -> float:
        """Calculate average score across all metrics"""
        all_scores = []
        for eval_result in evaluations:
            all_scores.extend(eval_result.metrics.values())
        return sum(all_scores) / len(all_scores) if all_scores else 0

rag_evaluator = RAGEvaluator()