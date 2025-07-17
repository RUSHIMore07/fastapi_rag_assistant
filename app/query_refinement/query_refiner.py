import asyncio
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from langchain_openai import ChatOpenAI
from app.config.settings import settings
from app.models.schemas import QueryRefinementRequest, QueryRefinementResult

logger = logging.getLogger(__name__)

class RefinementType(Enum):
    REWRITE = "rewrite"
    DECOMPOSE = "decompose"
    CLARIFY = "clarify"
    EXPAND = "expand"

class QueryRefiner:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model="gpt-4",
            temperature=0.3
        )
        
        self.refinement_prompts = {
            RefinementType.REWRITE: """
            You are an expert at improving search queries. Given a user query, rewrite it to be more specific, clear, and likely to retrieve relevant information.

            Original Query: {query}
            Context: {context}

            Please rewrite this query to be more effective for information retrieval. Make it:
            1. More specific and detailed
            2. Use better keywords
            3. Remove ambiguity
            4. Focus on the core information need

            Return only the improved query, nothing else.
            """,
            
            RefinementType.DECOMPOSE: """
            You are an expert at breaking down complex queries. Given a user query, decompose it into simpler sub-queries that together would answer the original question.

            Original Query: {query}
            Context: {context}

            Break this down into 2-4 simpler, more focused sub-queries. Each sub-query should:
            1. Be independently searchable
            2. Together cover the original question
            3. Be specific and clear

            Return the sub-queries as a numbered list, nothing else.
            """,
            
            RefinementType.CLARIFY: """
            You are an expert at clarifying ambiguous queries. Given a user query, identify ambiguities and provide a clarified version.

            Original Query: {query}
            Context: {context}

            This query might be ambiguous. Please provide a clarified version that:
            1. Removes ambiguity
            2. Specifies the exact information needed
            3. Adds necessary context
            4. Makes assumptions explicit

            Return only the clarified query, nothing else.
            """,
            
            RefinementType.EXPAND: """
            You are an expert at expanding queries with relevant terms. Given a user query, expand it with related keywords and concepts.

            Original Query: {query}
            Context: {context}

            Expand this query by adding:
            1. Related keywords and synonyms
            2. Relevant technical terms
            3. Context-specific terminology
            4. Alternative phrasings

            Return the expanded query, nothing else.
            """
        }
    
    async def refine_query(self, request: QueryRefinementRequest) -> QueryRefinementResult:
        """Refine user query based on specified type"""
        try:
            # Determine refinement type
            refinement_type = self._determine_refinement_type(request.query, request.context)
            
            if request.refinement_type:
                refinement_type = RefinementType(request.refinement_type)
            
            # Get appropriate prompt
            prompt_template = self.refinement_prompts[refinement_type]
            prompt = prompt_template.format(
                query=request.query,
                context=request.context or "No specific context provided"
            )
            
            # Get refinement from LLM
            response = await self.llm.ainvoke(prompt)
            refined_query = response.content.strip()
            
            # Handle decomposition differently
            if refinement_type == RefinementType.DECOMPOSE:
                sub_queries = self._parse_decomposed_queries(refined_query)
                return QueryRefinementResult(
                    original_query=request.query,
                    refined_query=request.query,  # Keep original for decomposition
                    sub_queries=sub_queries,
                    refinement_type=refinement_type.value,
                    confidence_score=0.9,
                    reasoning=f"Decomposed into {len(sub_queries)} sub-queries"
                )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(request.query, refined_query)
            
            return QueryRefinementResult(
                original_query=request.query,
                refined_query=refined_query,
                sub_queries=[],
                refinement_type=refinement_type.value,
                confidence_score=confidence_score,
                reasoning=f"Applied {refinement_type.value} refinement"
            )
            
        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            return QueryRefinementResult(
                original_query=request.query,
                refined_query=request.query,  # Return original on failure
                sub_queries=[],
                refinement_type="none",
                confidence_score=0.0,
                reasoning=f"Refinement failed: {str(e)}"
            )
    
    def _determine_refinement_type(self, query: str, context: Optional[str]) -> RefinementType:
        """Determine the best refinement type for the query"""
        query_lower = query.lower()
        
        # Check for complex queries that need decomposition
        if any(word in query_lower for word in ["and", "also", "both", "compare", "versus"]):
            return RefinementType.DECOMPOSE
        
        # Check for ambiguous queries that need clarification
        if any(word in query_lower for word in ["it", "this", "that", "these", "those"]) and not context:
            return RefinementType.CLARIFY
        
        # Check for short queries that need expansion
        if len(query.split()) < 3:
            return RefinementType.EXPAND
        
        # Default to rewrite for general improvement
        return RefinementType.REWRITE
    
    def _parse_decomposed_queries(self, decomposed_text: str) -> List[str]:
        """Parse decomposed queries from LLM response"""
        lines = decomposed_text.strip().split('\n')
        sub_queries = []
        
        for line in lines:
            line = line.strip()
            # Remove numbering and bullets
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove numbering pattern like "1. " or "- "
                clean_line = line.split('. ', 1)[-1].split('- ', 1)[-1].split('* ', 1)[-1]
                if clean_line:
                    sub_queries.append(clean_line)
        
        return sub_queries
    
    def _calculate_confidence_score(self, original: str, refined: str) -> float:
        """Calculate confidence score for refinement"""
        # Simple heuristic-based confidence scoring
        original_words = set(original.lower().split())
        refined_words = set(refined.lower().split())
        
        # Higher score if refined query has more specific terms
        if len(refined_words) > len(original_words):
            return 0.8
        
        # Lower score if queries are too similar
        overlap = len(original_words.intersection(refined_words))
        if overlap / len(original_words) > 0.8:
            return 0.6
        
        return 0.7
