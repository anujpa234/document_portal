import tiktoken
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

@dataclass
class TokenUsage:
    """Single request token usage class model"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_name: str
    cost_usd: float
    timestamp: datetime
    request_type: str  # "query", "rewrite", etc.

@dataclass
class SessionCosts:
    """Session level cost tracking class model"""
    session_id: str
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    first_request: datetime
    last_request: datetime
    requests: List[TokenUsage]

class TokenCounter:
    """
    Token counting and cost analysis for GPT-4o and Google Gemini-2.0-Flash which I am using for this project
    """
    
    # Model pricing per 1M tokens
    MODEL_PRICING = {
        # OpenAI GPT-4o per 1M tokens
        "gpt-4o": {
            "input": 2.50,    # $2.50 per 1M input tokens
            "output": 10.00   # $10.00 per 1M output tokens
        },
        
        # Google Gemini-2.0-Flash per 1M tokens
        "gemini-2.0-flash": {
            "input": 0.075,   # $0.075 per 1M input tokens  
            "output": 0.30    # $0.30 per 1M output tokens
        }
    }
    
    def __init__(self):
        self.session_costs: Dict[str, SessionCosts] = {}
        self.log = CustomLogger().get_logger(__name__)
        
        # Initializing tokenizer for GPT-4o
        self.gpt4o_tokenizer = None
        try:
            self.gpt4o_tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception as e:
            self.log.warning("Could not load GPT-4o tokenizer", error=str(e))
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens in text for specific model"""
        try:
            model_key = self._normalize_model_name(model_name)
            
            if model_key == "gpt-4o":
                # Using tiktoken for GPT-4o
                if self.gpt4o_tokenizer:
                    return len(self.gpt4o_tokenizer.encode(text))
                else:
                    # Fallback approximation for GPT-4o (3.3 chars ≈ 1 token) as per data taken from website
                    return int(len(text) / 3.3)
            
            elif model_key == "gemini-2.0-flash":
                # Google Gemini approximation (4 chars ≈ 1 token) as per data taken from website
                return int(len(text) / 4)
            
            else:
                # In case of Unknown model fallback
                return int(len(text) / 4)
                
        except Exception as e:
            self.log.warning("Token counting failed, using approximation", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """Calculate cost in USD for token usage"""
        model_key = self._normalize_model_name(model_name)
        
        if model_key not in self.MODEL_PRICING:
            self.log.warning(f"Unknown model for pricing: {model_name}, using GPT-4o pricing")
            model_key = "gpt-4o"
        
        pricing = self.MODEL_PRICING[model_key]
        
        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return round(input_cost + output_cost, 4)  # Round to 4 decimal places for precision
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to match pricing keys"""
        if not model_name:
            return "gpt-4o"  # Default fallback
            
        model_lower = model_name.lower().strip()
        
        # Handling GPT-4o variations
        if any(gpt_variant in model_lower for gpt_variant in [
            "gpt-4o", "gpt4o", "gpt-4-omni", "openai/gpt-4o"
        ]):
            return "gpt-4o"
        
        # Handling Gemini-2.0-Flash variations  
        elif any(gemini_variant in model_lower for gemini_variant in [
            "gemini-2.0-flash", "gemini-2-flash", "gemini2flash", 
            "gemini-flash-2.0", "google/gemini-2.0-flash"
        ]):
            return "gemini-2.0-flash"
        
        else:
            self.log.warning(f"Unknown model name: {model_name}, defaulting to gpt-4o")
            return "gpt-4o"  # Default to GPT-4o if unknown
    
    def track_usage(self, session_id: str, input_text: str, output_text: str, 
                   model_name: str, request_type: str = "query") -> TokenUsage:
        """Track token usage for a request"""
        
        # Count tokens
        input_tokens = self.count_tokens(input_text, model_name)
        output_tokens = self.count_tokens(output_text, model_name)
        total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        cost = self.calculate_cost(input_tokens, output_tokens, model_name)
        
        # Create usage record
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model_name=self._normalize_model_name(model_name),
            cost_usd=cost,
            timestamp=datetime.now(),
            request_type=request_type
        )
        
        # Update session costs
        self._update_session_costs(session_id, usage)
        
        self.log.info("Token usage tracked", 
                     session_id=session_id,
                     input_tokens=input_tokens,
                     output_tokens=output_tokens,
                     cost_usd=cost,
                     model=usage.model_name,
                     request_type=request_type)
        
        return usage
    
    def _update_session_costs(self, session_id: str, usage: TokenUsage):
        """Update session-level cost tracking"""
        if session_id not in self.session_costs:
            self.session_costs[session_id] = SessionCosts(
                session_id=session_id,
                total_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                total_cost_usd=0.0,
                first_request=usage.timestamp,
                last_request=usage.timestamp,
                requests=[]
            )
        
        session = self.session_costs[session_id]
        session.total_requests += 1
        session.total_input_tokens += usage.input_tokens
        session.total_output_tokens += usage.output_tokens
        session.total_tokens += usage.total_tokens
        session.total_cost_usd = round(session.total_cost_usd + usage.cost_usd, 8)
        session.last_request = usage.timestamp
        session.requests.append(usage)
        
        # Keep only last 50 requests per session to prevent memory bloat
        if len(session.requests) > 50:
            session.requests = session.requests[-50:]
    
    def get_session_costs(self, session_id: str) -> Optional[SessionCosts]:
        """Get cost summary for a session"""
        return self.session_costs.get(session_id)
    
    def get_total_costs(self) -> Dict[str, Any]:
        """Get total costs across all sessions"""
        total_sessions = len(self.session_costs)
        total_requests = sum(s.total_requests for s in self.session_costs.values())
        total_tokens = sum(s.total_tokens for s in self.session_costs.values())
        total_cost = sum(s.total_cost_usd for s in self.session_costs.values())
        
        # Model breakdown
        model_breakdown = {}
        for session in self.session_costs.values():
            for request in session.requests:
                model = request.model_name
                if model not in model_breakdown:
                    model_breakdown[model] = {"requests": 0, "tokens": 0, "cost": 0.0}
                model_breakdown[model]["requests"] += 1
                model_breakdown[model]["tokens"] += request.total_tokens
                model_breakdown[model]["cost"] = round(
                    model_breakdown[model]["cost"] + request.cost_usd, 8
                )
        
        return {
            "total_sessions": total_sessions,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "model_breakdown": model_breakdown,
            "supported_models": list(self.MODEL_PRICING.keys())
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about supported models and their pricing"""
        return {
            "supported_models": {
                "gpt-4o": {
                    "description": "OpenAI GPT-4o",
                    "input_cost_per_1M": f"${self.MODEL_PRICING['gpt-4o']['input']}",
                    "output_cost_per_1M": f"${self.MODEL_PRICING['gpt-4o']['output']}",
                    "tokenizer": "tiktoken (accurate)"
                },
                "gemini-2.0-flash": {
                    "description": "Google Gemini-2.0-Flash", 
                    "input_cost_per_1M": f"${self.MODEL_PRICING['gemini-2.0-flash']['input']}",
                    "output_cost_per_1M": f"${self.MODEL_PRICING['gemini-2.0-flash']['output']}",
                    "tokenizer": "approximation (4 chars ≈ 1 token)"
                }
            },
            "note": "Costs are calculated per 1 million tokens"
        }
