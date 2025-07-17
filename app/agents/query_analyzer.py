from typing import Dict, Any, List
from app.models.schemas import QueryType
import re
import logging

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    def __init__(self):
        self.intent_patterns = {
            "question": [r"\?", r"what", r"how", r"why", r"when", r"where", r"who"],
            "request": [r"please", r"can you", r"could you", r"would you"],
            "command": [r"show", r"tell", r"explain", r"describe", r"analyze"],
            "creative": [r"write", r"create", r"generate", r"compose", r"imagine"]
        }
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine intent, complexity, and requirements"""
        analysis = {
            "original_query": query,
            "intent": self.detect_intent(query),
            "complexity": self.assess_complexity(query),
            "entities": self.extract_entities(query),
            "requirements": self.determine_requirements(query),
            "query_type": self.classify_query_type(query)
        }
        
        return analysis
    
    def detect_intent(self, query: str) -> str:
        """Detect the intent of the query"""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return "unknown"
    
    def assess_complexity(self, query: str) -> Dict[str, Any]:
        """Assess query complexity"""
        words = query.split()
        
        complexity_factors = {
            "length": len(words),
            "sentence_count": len(query.split('.')),
            "has_multiple_questions": query.count('?') > 1,
            "has_code": "```" in query or "def " in query,
            "has_math": any(op in query for op in ["∑", "∫", "√", "∞", "π"]),
            "technical_terms": len([w for w in words if len(w) > 8])
        }
        
        # Calculate complexity score
        score = 0
        if complexity_factors["length"] > 20:
            score += 2
        if complexity_factors["sentence_count"] > 2:
            score += 1
        if complexity_factors["has_multiple_questions"]:
            score += 2
        if complexity_factors["has_code"]:
            score += 3
        if complexity_factors["has_math"]:
            score += 2
        if complexity_factors["technical_terms"] > 3:
            score += 1
        
        return {
            "score": score,
            "level": "high" if score >= 5 else "medium" if score >= 3 else "low",
            "factors": complexity_factors
        }
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query (simplified)"""
        # This is a simplified entity extraction
        # In production, use spaCy or similar NLP library
        words = query.split()
        entities = []
        
        for word in words:
            if word.istitle() and len(word) > 2:
                entities.append(word)
        
        return entities
    
    def determine_requirements(self, query: str) -> Dict[str, bool]:
        """Determine what capabilities are required"""
        query_lower = query.lower()
        
        return {
            "needs_context": any(word in query_lower for word in ["context", "previous", "earlier", "before"]),
            "needs_search": any(word in query_lower for word in ["search", "find", "lookup", "what is"]),
            "needs_calculation": any(word in query_lower for word in ["calculate", "compute", "math", "sum"]),
            "needs_reasoning": any(word in query_lower for word in ["why", "reason", "because", "analyze"]),
            "needs_creativity": any(word in query_lower for word in ["creative", "write", "story", "poem"])
        }
    
    def classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["image", "picture", "photo", "visual"]):
            return QueryType.IMAGE
        elif any(word in query_lower for word in ["document", "pdf", "file", "upload"]):
            return QueryType.DOCUMENT
        else:
            return QueryType.TEXT
