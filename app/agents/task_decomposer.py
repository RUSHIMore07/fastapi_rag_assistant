from typing import List, Dict, Any
from app.models.schemas import AgentStep
import logging

logger = logging.getLogger(__name__)

class TaskDecomposer:
    def __init__(self):
        self.task_patterns = {
            "research": ["find", "search", "lookup", "research"],
            "analysis": ["analyze", "compare", "evaluate", "assess"],
            "synthesis": ["summarize", "combine", "integrate", "conclude"],
            "creative": ["write", "create", "generate", "compose"]
        }
    
    async def decompose_task(self, query: str, complexity_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose complex tasks into subtasks"""
        
        if complexity_analysis["level"] == "low":
            return [{"type": "simple_response", "query": query, "priority": 1}]
        
        subtasks = []
        
        # For complex queries, break down into stages
        if complexity_analysis["level"] in ["medium", "high"]:
            # Stage 1: Information gathering
            subtasks.append({
                "type": "information_gathering",
                "query": query,
                "priority": 1,
                "description": "Gather relevant information and context"
            })
            
            # Stage 2: Analysis
            if any(word in query.lower() for word in ["analyze", "compare", "evaluate"]):
                subtasks.append({
                    "type": "analysis",
                    "query": query,
                    "priority": 2,
                    "description": "Analyze gathered information"
                })
            
            # Stage 3: Synthesis
            subtasks.append({
                "type": "synthesis",
                "query": query,
                "priority": 3,
                "description": "Synthesize final response"
            })
        
        return subtasks
    
    def identify_task_type(self, query: str) -> str:
        """Identify the primary task type"""
        query_lower = query.lower()
        
        for task_type, patterns in self.task_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return task_type
        
        return "general"
