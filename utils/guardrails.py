import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from logger.custom_logger import CustomLogger

class GuardrailViolationType(str, Enum):
    MALICIOUS_PROMPT = "malicious_prompt"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    PII_DETECTED = "pii_detected"
    TOXIC_CONTENT = "toxic_content"
    OFF_TOPIC = "off_topic"
    LOW_QUALITY = "low_quality"
    SAFETY_VIOLATION = "safety_violation"

@dataclass
class GuardrailViolation:
    violation_type: GuardrailViolationType
    severity: str  # "low", "medium", "high", "critical"
    message: str
    detected_content: Optional[str] = None
    confidence: float = 0.0

@dataclass
class GuardrailResult:
    is_safe: bool
    violations: List[GuardrailViolation]
    filtered_content: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class RAGGuardrails:
    """
    Comprehensive guardrails for RAG system including input/output filtering,
    safety checks, and content moderation.
    """
    
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        
        # Malicious prompt patterns
        self.malicious_patterns = [
            # Prompt injection attempts
            r"ignore\s+(?:previous|all|above|prior)\s+(?:instructions|prompts|rules)",
            r"forget\s+(?:everything|all|previous|above)",
            r"act\s+as\s+(?:a\s+)?(?:jailbreak|dan|evil|hacker)",
            r"pretend\s+(?:to\s+be|you\s+are)",
            r"roleplay\s+as",
            r"system\s*[:]\s*you\s+are",
            
            # Data extraction attempts
            r"show\s+me\s+(?:all|your)\s+(?:training|prompt|system|instructions)",
            r"what\s+(?:are\s+)?your\s+(?:instructions|prompts|rules)",
            r"repeat\s+(?:your|the)\s+(?:instructions|prompt|system)",
            
            # Bypass attempts
            r"\\n\\n\\nhuman:",
            r"\\n\\nhuman:",
            r"user\s*[:]\s*",
            r"human\s*[:]\s*",
        ]
        
        # PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        # Inappropriate content keywords
        self.inappropriate_keywords = [
            # Violence
            "kill", "murder", "violence", "harm", "hurt", "destroy", "attack",
            # Hate speech indicators
            "hate", "racist", "discrimination", 
            # Adult content
            "explicit", "nsfw", "adult", "sexual",
            # Illegal activities
            "illegal", "drugs", "bomb", "weapon", "fraud"
        ]
        
        # Off-topic indicators for document Q&A
        self.off_topic_patterns = [
            r"what.*weather",
            r"current.*news",
            r"stock.*price",
            r"sports.*score",
            r"recipe.*for",
            r"how.*cook",
            r"movie.*recommendation"
        ]

    def validate_input(self, user_input: str, session_id: str) -> GuardrailResult:
        """Validate user input before processing"""
        violations = []
        
        # Check for malicious prompts
        violations.extend(self._check_malicious_prompts(user_input))
        
        # Check for PII
        violations.extend(self._check_pii(user_input))
        
        # Check for inappropriate content
        violations.extend(self._check_inappropriate_content(user_input))
        
        # Check if off-topic
        violations.extend(self._check_off_topic(user_input))
        
        # Check input quality
        violations.extend(self._check_input_quality(user_input))
        
        # Determine if safe
        critical_violations = [v for v in violations if v.severity == "critical"]
        high_violations = [v for v in violations if v.severity == "high"]
        
        is_safe = len(critical_violations) == 0 and len(high_violations) == 0
        
        # Filter content if needed
        filtered_content = self._filter_input(user_input, violations) if not is_safe else None
        
        result = GuardrailResult(
            is_safe=is_safe,
            violations=violations,
            filtered_content=filtered_content
        )
        
        if violations:
            self.log.warning("Input guardrail violations detected", 
                           session_id=session_id,
                           violations=[v.violation_type.value for v in violations],
                           severity_levels=[v.severity for v in violations])
        
        return result

    def validate_output(self, generated_answer: str, user_input: str, session_id: str) -> GuardrailResult:
        """Validate generated output before returning to user"""
        violations = []
        
        # Check for PII in output
        violations.extend(self._check_pii(generated_answer))
        
        # Check for inappropriate content in output
        violations.extend(self._check_inappropriate_content(generated_answer))
        
        # Check answer quality
        violations.extend(self._check_answer_quality(generated_answer, user_input))
        
        # Check for potential hallucinations
        violations.extend(self._check_hallucination_indicators(generated_answer))
        
        # Determine if safe
        critical_violations = [v for v in violations if v.severity in ["critical", "high"]]
        is_safe = len(critical_violations) == 0
        
        # Filter output if needed
        filtered_content = self._filter_output(generated_answer, violations) if not is_safe else None
        
        result = GuardrailResult(
            is_safe=is_safe,
            violations=violations,
            filtered_content=filtered_content
        )
        
        if violations:
            self.log.warning("Output guardrail violations detected", 
                           session_id=session_id,
                           violations=[v.violation_type.value for v in violations],
                           severity_levels=[v.severity for v in violations])
        
        return result

    def _check_malicious_prompts(self, text: str) -> List[GuardrailViolation]:
        """Check for prompt injection and malicious patterns"""
        violations = []
        text_lower = text.lower()
        
        for pattern in self.malicious_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violations.append(GuardrailViolation(
                    violation_type=GuardrailViolationType.MALICIOUS_PROMPT,
                    severity="critical",
                    message=f"Potential prompt injection detected: {pattern}",
                    detected_content=text[:100],
                    confidence=0.8
                ))
                break  # One detection is enough
        
        return violations

    def _check_pii(self, text: str) -> List[GuardrailViolation]:
        """Check for personally identifiable information"""
        violations = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append(GuardrailViolation(
                    violation_type=GuardrailViolationType.PII_DETECTED,
                    severity="high",
                    message=f"PII detected: {pii_type}",
                    detected_content=str(matches[:2]),  # Show first 2 matches
                    confidence=0.9
                ))
        
        return violations

    def _check_inappropriate_content(self, text: str) -> List[GuardrailViolation]:
        """Check for inappropriate content"""
        violations = []
        text_lower = text.lower()
        
        detected_keywords = [kw for kw in self.inappropriate_keywords if kw in text_lower]
        
        if detected_keywords:
            violations.append(GuardrailViolation(
                violation_type=GuardrailViolationType.INAPPROPRIATE_CONTENT,
                severity="medium",
                message=f"Inappropriate content detected: {', '.join(detected_keywords[:3])}",
                detected_content=str(detected_keywords),
                confidence=0.7
            ))
        
        return violations

    def _check_off_topic(self, text: str) -> List[GuardrailViolation]:
        """Check if query is off-topic for document Q&A"""
        violations = []
        text_lower = text.lower()
        
        for pattern in self.off_topic_patterns:
            if re.search(pattern, text_lower):
                violations.append(GuardrailViolation(
                    violation_type=GuardrailViolationType.OFF_TOPIC,
                    severity="low",
                    message="Query appears to be off-topic for document Q&A",
                    confidence=0.6
                ))
                break
        
        return violations

    def _check_input_quality(self, text: str) -> List[GuardrailViolation]:
        """Check input quality"""
        violations = []
        
        # Too short
        if len(text.strip()) < 3:
            violations.append(GuardrailViolation(
                violation_type=GuardrailViolationType.LOW_QUALITY,
                severity="medium",
                message="Input too short to process meaningfully",
                confidence=0.9
            ))
        
        # Too long
        if len(text) > 2000:
            violations.append(GuardrailViolation(
                violation_type=GuardrailViolationType.LOW_QUALITY,
                severity="medium",
                message="Input too long, may affect processing quality",
                confidence=0.8
            ))
        
        # Non-sensical input
        if len(text.split()) > 5 and len(set(text.lower().split())) < 3:
            violations.append(GuardrailViolation(
                violation_type=GuardrailViolationType.LOW_QUALITY,
                severity="low",
                message="Input appears repetitive or non-sensical",
                confidence=0.6
            ))
        
        return violations

    def _check_answer_quality(self, answer: str, question: str) -> List[GuardrailViolation]:
        """Check generated answer quality"""
        violations = []
        
        # Check if answer is too short
        if len(answer.strip()) < 10:
            violations.append(GuardrailViolation(
                violation_type=GuardrailViolationType.LOW_QUALITY,
                severity="medium",
                message="Generated answer is too short",
                confidence=0.8
            ))
        
        # Check for generic/evasive responses
        generic_phrases = [
            "i don't know", "i'm not sure", "i cannot", "i can't help",
            "i don't have information", "sorry, i don't know"
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in generic_phrases) and len(answer) < 50:
            violations.append(GuardrailViolation(
                violation_type=GuardrailViolationType.LOW_QUALITY,
                severity="low",
                message="Answer appears generic or evasive",
                confidence=0.6
            ))
        
        return violations

    def _check_hallucination_indicators(self, answer: str) -> List[GuardrailViolation]:
        """Check for potential hallucination indicators"""
        violations = []
        
        # Check for uncertain language that might indicate hallucination
        uncertain_phrases = [
            "i think", "i believe", "probably", "maybe", "might be",
            "it seems", "appears to be", "could be"
        ]
        
        answer_lower = answer.lower()
        uncertain_count = sum(1 for phrase in uncertain_phrases if phrase in answer_lower)
        
        if uncertain_count > 2:
            violations.append(GuardrailViolation(
                violation_type=GuardrailViolationType.LOW_QUALITY,
                severity="low",
                message="Answer contains many uncertainty indicators",
                confidence=0.5
            ))
        
        return violations

    def _filter_input(self, text: str, violations: List[GuardrailViolation]) -> str:
        """Filter problematic content from input"""
        filtered = text
        
        # Remove PII
        for pii_type, pattern in self.pii_patterns.items():
            filtered = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", filtered)
        
        # Add safety notice
        return f"[FILTERED_INPUT] {filtered}"

    def _filter_output(self, text: str, violations: List[GuardrailViolation]) -> str:
        """Filter problematic content from output"""
        filtered = text
        
        # Remove PII from output
        for pii_type, pattern in self.pii_patterns.items():
            filtered = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", filtered)
        
        return filtered

    def generate_safety_response(self, violations: List[GuardrailViolation]) -> str:
        """Generate appropriate safety response based on violations"""
        
        critical_violations = [v for v in violations if v.severity == "critical"]
        high_violations = [v for v in violations if v.severity == "high"]
        
        if critical_violations:
            return (
                "I cannot process this request as it appears to contain potentially "
                "harmful content or attempts to bypass safety measures. Please rephrase "
                "your question in a constructive manner."
            )
        
        if high_violations:
            pii_violations = [v for v in high_violations if v.violation_type == GuardrailViolationType.PII_DETECTED]
            if pii_violations:
                return (
                    "I've detected what appears to be personal information in your request. "
                    "For privacy and security reasons, I cannot process requests containing "
                    "personal identifiable information. Please remove any personal details and try again."
                )
        
        return (
            "I'm here to help answer questions about your documents. Please make sure "
            "your question is related to the content you've uploaded and is appropriate "
            "for a professional context."
        )

# Global guardrails instance
rag_guardrails = RAGGuardrails()