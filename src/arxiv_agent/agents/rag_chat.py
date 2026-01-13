"""RAG Chat agent for interactive paper Q&A."""

from loguru import logger

from arxiv_agent.agents.state import AgentState
from arxiv_agent.config.settings import get_settings
from arxiv_agent.core.llm_service import get_llm_service
from arxiv_agent.core.vector_store import get_vector_store
from arxiv_agent.data.storage import get_db


RAG_SYSTEM_PROMPT = """You are a research assistant helping users understand academic papers.
Use the retrieved context to answer questions accurately.
Always cite the relevant sections when providing information.
If the context doesn't contain enough information, say so clearly.
Be concise but thorough in your explanations."""


class RAGChatAgent:
    """Agent for interactive RAG-based paper Q&A.
    
    Handles:
    - Hybrid retrieval (dense + sparse)
    - Context assembly
    - Chat history management
    - Response generation with citations
    """
    
    def __init__(self):
        """Initialize RAG chat agent."""
        self.vector_store = get_vector_store()
        self.llm = get_llm_service()
        self.settings = get_settings()
        self.db = get_db()
    
    async def run(self, state: AgentState) -> AgentState:
        """Execute RAG chat agent.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        state.current_step = "chatting"
        logger.info(f"RAG Chat agent running for paper: {state.paper_id}")
        
        try:
            await self._process_query(state)
        except Exception as e:
            logger.error(f"RAG chat error: {e}")
            state.add_error(f"Chat failed: {str(e)}")
        
        return state
    
    async def _process_query(self, state: AgentState) -> None:
        """Process a chat query with RAG.
        
        Args:
            state: Current agent state
        """
        if not state.query:
            state.add_error("No query provided")
            return
        
        if not state.paper_id:
            state.add_error("No paper ID provided for chat")
            return
        
        paper_id = state.paper_id
        if not paper_id.startswith("arxiv:"):
            paper_id = f"arxiv:{paper_id}"
        
        # Get or create chat session
        session = self.db.get_or_create_chat_session(paper_id)
        state.chat_session_id = session.id
        
        # Retrieve relevant context using hybrid search
        results = self.vector_store.hybrid_search(
            query=state.query,
            paper_id=paper_id,
            k=self.settings.retrieval.top_k,
        )
        
        if not results:
            logger.warning(f"No chunks found for paper {paper_id}")
            # Fall back to paper metadata
            paper = self.db.get_paper(paper_id)
            if paper:
                context = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
            else:
                state.add_error("No context available for this paper. Try analyzing it first.")
                return
        else:
            # Build context from top chunks
            context_parts = []
            for i, result in enumerate(results[:self.settings.retrieval.rerank_top_k]):
                context_parts.append(
                    f"[Section: {result.chunk.section}]\n{result.chunk.content}"
                )
            context = "\n\n---\n\n".join(context_parts)
        
        # Get chat history
        history = self.db.get_chat_history(session.id, limit=10)
        history_text = ""
        if history:
            history_entries = []
            for msg in history[-6:]:  # Last 3 exchanges
                role_label = "User" if msg.role == "user" else "Assistant"
                history_entries.append(f"{role_label}: {msg.content[:500]}")
            history_text = "\n".join(history_entries)
        
        # Build prompt
        prompt = f"""Paper Context:
{context}

{f"Previous Conversation:{chr(10)}{history_text}" if history_text else ""}

User Question: {state.query}

Provide a helpful, accurate response based on the paper content. 
Cite specific sections when relevant."""
        
        # Generate response
        response = await self.llm.agenerate(
            prompt=prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=2000,
        )
        
        # Prepare retrieved chunks info
        chunk_info = [
            {
                "id": r.chunk.id,
                "section": r.chunk.section,
                "score": round(r.score, 3),
            }
            for r in results[:5]
        ] if results else []
        
        # Save messages to database
        self.db.add_chat_message(session.id, "user", state.query)
        self.db.add_chat_message(
            session.id,
            "assistant",
            response.content,
            retrieved_chunks=chunk_info,
        )
        
        # Update state
        state.chat_history.append({"role": "user", "content": state.query})
        state.chat_history.append({"role": "assistant", "content": response.content})
        state.retrieved_chunks = [
            {"content": r.chunk.content[:200], "section": r.chunk.section}
            for r in results[:5]
        ] if results else []
        
        # Record interaction
        self.db.record_interaction(paper_id, "chatted")
        
        logger.info(f"Generated chat response for {paper_id}")
    
    async def get_follow_up_suggestions(self, state: AgentState) -> list[str]:
        """Generate follow-up question suggestions.
        
        Args:
            state: Current agent state
        
        Returns:
            List of suggested follow-up questions
        """
        if not state.chat_history:
            return [
                "What is the main contribution of this paper?",
                "What methodology does this paper use?",
                "What are the key results?",
                "What are the limitations of this approach?",
            ]
        
        last_response = state.chat_history[-1]["content"] if state.chat_history else ""
        
        prompt = f"""Based on this conversation about a research paper, suggest 3 relevant follow-up questions:

Last response:
{last_response[:500]}

Provide exactly 3 questions, one per line, without numbering."""
        
        response = await self.llm.agenerate(
            prompt=prompt,
            system_prompt="You suggest relevant academic questions.",
            temperature=0.7,
            max_tokens=200,
        )
        
        questions = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        return questions[:3]
