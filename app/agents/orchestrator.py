from typing import Dict, Any, List
import asyncio
import time
from app.agents.query_analyzer import QueryAnalyzer
from app.agents.task_decomposer import TaskDecomposer
from app.llm_routing.router import LLMRouter
from app.retrieval.vector_store import FAISSVectorStore
from app.multi_modal.handlers import MultiModalHandler
from app.models.schemas import AgentStep, QueryResponse, RetrievalContext
import logging

logger = logging.getLogger(__name__)

class AgenticOrchestrator:
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.task_decomposer = TaskDecomposer()
        self.llm_router = LLMRouter()
        self.vector_store = FAISSVectorStore()
        self.multimodal_handler = MultiModalHandler()
        self.agent_steps = []
    
    async def process_query(self, query_request: Dict[str, Any]) -> QueryResponse:
        """Main orchestration method"""
        start_time = time.time()
        self.agent_steps = []
        
        try:
            # Step 1: Analyze query
            analysis = await self.analyze_query_step(query_request["query"])
            
            # Step 2: Decompose task if complex
            subtasks = await self.decompose_task_step(
                query_request["query"], 
                analysis["complexity"]
            )
            
            # Step 3: Retrieve relevant context
            context = await self.retrieve_context_step(query_request["query"])
            
            # Step 4: Route to appropriate LLM
            selected_provider = self.llm_router.route_query(
                query_request["query"],
                query_request["query_type"],
                query_request.get("preferred_llm")
            )
            
            # Step 5: Generate response
            response = await self.generate_response_step(
                query_request, context, selected_provider
            )
            
            # Step 6: Post-process response
            final_response = await self.post_process_response_step(response, context)
            
            return QueryResponse(
                response=final_response["content"],
                model_used=final_response["model"],
                session_id=query_request["session_id"],
                context_used=context,
                agent_steps=self.agent_steps,
                metadata={
                    "analysis": analysis,
                    "subtasks": subtasks,
                    "processing_time": time.time() - start_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}")
            return QueryResponse(
                response=f"Error processing query: {str(e)}",
                model_used="error",
                session_id=query_request["session_id"],
                agent_steps=self.agent_steps,
                metadata={"error": str(e)}
            )
    
    async def analyze_query_step(self, query: str) -> Dict[str, Any]:
        """Step 1: Analyze the query"""
        start_time = time.time()
        
        try:
            analysis = await self.query_analyzer.analyze_query(query)
            
            self.agent_steps.append(AgentStep(
                agent_name="QueryAnalyzer",
                action="analyze_query",
                input_data={"query": query},
                output_data=analysis,
                execution_time=time.time() - start_time,
                success=True
            ))
            
            return analysis
            
        except Exception as e:
            self.agent_steps.append(AgentStep(
                agent_name="QueryAnalyzer",
                action="analyze_query",
                input_data={"query": query},
                output_data={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            ))
            raise
    
    async def decompose_task_step(self, query: str, complexity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Step 2: Decompose complex tasks"""
        start_time = time.time()
        
        try:
            subtasks = await self.task_decomposer.decompose_task(query, complexity)
            
            self.agent_steps.append(AgentStep(
                agent_name="TaskDecomposer",
                action="decompose_task",
                input_data={"query": query, "complexity": complexity},
                output_data={"subtasks": subtasks},
                execution_time=time.time() - start_time,
                success=True
            ))
            
            return subtasks
            
        except Exception as e:
            self.agent_steps.append(AgentStep(
                agent_name="TaskDecomposer",
                action="decompose_task",
                input_data={"query": query, "complexity": complexity},
                output_data={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            ))
            return []
    
    async def retrieve_context_step(self, query: str) -> RetrievalContext:
        """Step 3: Retrieve relevant context"""
        start_time = time.time()
        
        try:
            search_results = await self.vector_store.similarity_search(query, k=5)
            
            context = RetrievalContext(
                chunks=[result["content"] for result in search_results],
                sources=[result["metadata"].get("source", "unknown") for result in search_results],
                relevance_scores=[result["similarity_score"] for result in search_results],
                metadata={"search_results": search_results}
            )
            
            self.agent_steps.append(AgentStep(
                agent_name="VectorStore",
                action="retrieve_context",
                input_data={"query": query},
                output_data={"context_chunks": len(context.chunks)},
                execution_time=time.time() - start_time,
                success=True
            ))
            
            return context
            
        except Exception as e:
            self.agent_steps.append(AgentStep(
                agent_name="VectorStore",
                action="retrieve_context",
                input_data={"query": query},
                output_data={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            ))
            
            return RetrievalContext(chunks=[], sources=[], relevance_scores=[])
    
    async def generate_response_step(self, query_request: Dict[str, Any], 
                                  context: RetrievalContext, 
                                  provider) -> Dict[str, Any]:
        """Step 4: Generate response using LLM"""
        start_time = time.time()
        
        try:
            # Build prompt with context
            prompt = self.build_prompt_with_context(
                query_request["query"], 
                context
            )
            
            response = await self.llm_router.generate_response(
                prompt,
                provider,
                max_tokens=query_request.get("max_tokens", 1000),
                temperature=query_request.get("temperature", 0.7)
            )
            
            self.agent_steps.append(AgentStep(
                agent_name="LLMRouter",
                action="generate_response",
                input_data={"provider": str(provider), "prompt_length": len(prompt)},
                output_data={"response_length": len(response.get("content", ""))},
                execution_time=time.time() - start_time,
                success=response.get("success", False)
            ))
            
            return response
            
        except Exception as e:
            self.agent_steps.append(AgentStep(
                agent_name="LLMRouter",
                action="generate_response",
                input_data={"provider": str(provider)},
                output_data={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            ))
            raise
    
    async def post_process_response_step(self, response: Dict[str, Any], 
                                       context: RetrievalContext) -> Dict[str, Any]:
        """Step 5: Post-process the response"""
        start_time = time.time()
        
        try:
            # Add citations if context was used
            if context.chunks:
                response["content"] = self.add_citations(response["content"], context)
            
            self.agent_steps.append(AgentStep(
                agent_name="PostProcessor",
                action="post_process_response",
                input_data={"has_context": len(context.chunks) > 0},
                output_data={"final_response_length": len(response["content"])},
                execution_time=time.time() - start_time,
                success=True
            ))
            
            return response
            
        except Exception as e:
            self.agent_steps.append(AgentStep(
                agent_name="PostProcessor",
                action="post_process_response",
                input_data={},
                output_data={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            ))
            return response
    
    def build_prompt_with_context(self, query: str, context: RetrievalContext) -> str:
        """Build prompt with retrieved context"""
        if not context.chunks:
            return query
        
        context_text = "\n".join([
            f"Context {i+1}: {chunk}" 
            for i, chunk in enumerate(context.chunks[:3])
        ])
        
        prompt = f"""Based on the following context, please answer the question:

Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information, please indicate that clearly."""
        
        return prompt
    
    def add_citations(self, response: str, context: RetrievalContext) -> str:
        """Add citations to response"""
        if not context.sources:
            return response
        
        # Simple citation addition
        citations = "\n\nSources:\n"
        for i, source in enumerate(context.sources[:3]):
            citations += f"{i+1}. {source}\n"
        
        return response + citations
