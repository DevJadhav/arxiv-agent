"""Analyzer agent for deep paper analysis."""

from pathlib import Path

from loguru import logger

from arxiv_agent.agents.state import AgentState
from arxiv_agent.config.settings import get_settings
from arxiv_agent.core.api_client import get_api_client
from arxiv_agent.core.llm_service import get_llm_service
from arxiv_agent.core.pdf_processor import get_pdf_processor
from arxiv_agent.core.vector_store import Chunk, get_vector_store
from arxiv_agent.data.models import Analysis
from arxiv_agent.data.storage import get_db


ANALYSIS_SYSTEM_PROMPT = """You are an expert research paper analyst. 
Your task is to provide comprehensive analysis of academic papers.
Be thorough, accurate, and cite specific sections when relevant.
Focus on methodology, key findings, limitations, and practical implications."""

ANALYSIS_PROMPT_TEMPLATE = """Analyze the following research paper:

Title: {title}
Authors: {authors}

Content:
{content}

Provide a comprehensive analysis including:
1. Main contribution and novelty
2. Methodology and approach
3. Key findings and results
4. Limitations and future work
5. Practical applications
6. Related work connections"""


class AnalyzerAgent:
    """Agent for deep paper analysis.
    
    Handles:
    - PDF processing and chunking
    - Vector store indexing
    - Multi-pass LLM analysis
    - Paper-to-code plan generation
    """
    
    def __init__(self):
        """Initialize analyzer agent."""
        self.api_client = get_api_client()
        self.pdf_processor = get_pdf_processor()
        self.vector_store = get_vector_store()
        self.llm = get_llm_service()
        self.settings = get_settings()
        self.db = get_db()
    
    async def run(self, state: AgentState) -> AgentState:
        """Execute analyzer agent.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        state.current_step = "analyzing"
        logger.info(f"Analyzer agent running for task: {state.task_type}")
        
        try:
            if state.task_type == "paper2code":
                await self._generate_code_plan(state)
            else:
                await self._analyze_paper(state)
        except Exception as e:
            logger.error(f"Analyzer error: {e}")
            state.add_error(f"Analysis failed: {str(e)}")
        
        return state
    
    async def _analyze_paper(self, state: AgentState) -> None:
        """Perform comprehensive paper analysis.
        
        Args:
            state: Current agent state
        """
        paper = state.current_paper
        if not paper:
            state.add_error("No paper to analyze")
            return
        
        # Get analysis depth from options
        depth = state.options.get("depth", "standard")
        
        # Check for existing analysis
        existing = self.db.get_latest_analysis(paper.id, depth)
        if existing and not state.options.get("force", False):
            logger.info(f"Using cached analysis for {paper.id}")
            state.analysis = existing.content
            return
        
        # Ensure PDF is available
        if not paper.pdf_path or not Path(paper.pdf_path).exists():
            logger.info(f"Downloading PDF for {paper.id}")
            await self._download_pdf(paper)
        
        # Process PDF if path exists
        processed = None
        if paper.pdf_path and Path(paper.pdf_path).exists():
            processed = self.pdf_processor.process(paper.pdf_path)
            paper.pdf_hash = processed.pdf_hash
            self.db.save_paper(paper)
            
            # Index chunks for RAG
            chunks = self._create_chunks(paper.id, processed)
            if chunks:
                self.vector_store.add_chunks(chunks)
                logger.info(f"Indexed {len(chunks)} chunks for {paper.id}")
        
        # Prepare content for analysis
        if processed:
            content = processed.markdown[:30000]  # Truncate for context window
        else:
            content = paper.abstract
        
        # Generate analysis with LLM
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            title=paper.title,
            authors=", ".join(paper.authors[:5]),
            content=content,
        )
        
        response = await self.llm.agenerate(
            prompt=prompt,
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=4000 if depth == "full" else 2000,
        )
        
        # Build analysis result
        analysis_content = {
            "text": response.content,
            "depth": depth,
            "has_pdf": processed is not None,
        }
        
        if processed:
            analysis_content["sections"] = [
                {"name": s.name, "pages": f"{s.page_start}-{s.page_end}"}
                for s in processed.sections
            ]
            analysis_content["page_count"] = processed.page_count
        
        # Save analysis
        analysis = Analysis(
            paper_id=paper.id,
            analysis_type=depth,
            content=analysis_content,
            model_used=response.model,
            token_count=response.input_tokens + response.output_tokens,
        )
        self.db.save_analysis(analysis)
        
        state.analysis = analysis_content
        state.current_paper = paper
        
        # Record interaction
        self.db.record_interaction(paper.id, "analyzed")
        
        logger.info(f"Completed {depth} analysis for {paper.id}")
    
    async def _generate_code_plan(self, state: AgentState) -> None:
        """Generate paper-to-code implementation plan.
        
        Args:
            state: Current agent state
        """
        paper = state.current_paper
        if not paper:
            state.add_error("No paper for code generation")
            return
        
        # Ensure PDF is available
        if not paper.pdf_path or not Path(paper.pdf_path).exists():
            await self._download_pdf(paper)
        
        # Process PDF
        content = paper.abstract
        if paper.pdf_path and Path(paper.pdf_path).exists():
            processed = self.pdf_processor.process(paper.pdf_path)
            content = processed.markdown[:40000]
        
        # Generate implementation plan using code-optimized model
        code_llm = get_llm_service(agent="code")
        
        plan_schema = {
            "type": "object",
            "properties": {
                "architecture": {
                    "type": "string",
                    "description": "High-level architecture description"
                },
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "file": {"type": "string"},
                            "description": {"type": "string"},
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                        },
                        "required": ["name", "file", "description"]
                    },
                },
                "python_dependencies": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "implementation_order": {
                    "type": "array",
                    "items": {"type": "string"}
                },
            },
            "required": ["architecture", "components", "python_dependencies", "implementation_order"],
        }
        
        plan_prompt = f"""Analyze this research paper and create a detailed implementation plan.

Title: {paper.title}

Paper Content:
{content}

Create a comprehensive implementation plan that would allow a developer to reproduce 
the paper's methodology in Python. Focus on the core algorithm and architecture.
Include specific file names, class/function names, and dependencies."""
        
        try:
            plan = await code_llm.generate_structured(
                prompt=plan_prompt,
                output_schema=plan_schema,
            )
            
            state.code_plan = plan
            
            # Save as analysis
            analysis = Analysis(
                paper_id=paper.id,
                analysis_type="code_plan",
                content=plan,
                model_used=code_llm.model,
            )
            self.db.save_analysis(analysis)
            
            self.db.record_interaction(paper.id, "code_generated")
            
            logger.info(f"Generated code plan for {paper.id}")
            
        except Exception as e:
            logger.error(f"Failed to generate code plan: {e}")
            state.add_error(f"Code plan generation failed: {str(e)}")
    
    async def _download_pdf(self, paper) -> None:
        """Download PDF for a paper.
        
        Args:
            paper: Paper object to download PDF for
        """
        if paper.published_date:
            pdf_dir = self.settings.get_pdf_dir(
                paper.published_date.year,
                paper.published_date.month,
            )
        else:
            pdf_dir = self.settings.data_dir / "pdfs" / "unknown"
        
        try:
            paper.pdf_path = await self.api_client.download_pdf(paper, str(pdf_dir))
            self.db.save_paper(paper)
        except Exception as e:
            logger.warning(f"Failed to download PDF for {paper.id}: {e}")
    
    def _create_chunks(self, paper_id: str, processed) -> list[Chunk]:
        """Create chunks from processed paper for vector indexing.
        
        Args:
            paper_id: Paper ID
            processed: ProcessedPaper object
        
        Returns:
            List of Chunk objects
        """
        # Custom simple text splitter to avoid langchain dependency issues
        class SimpleTextSplitter:
            def __init__(self, chunk_size, chunk_overlap, separators):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators
            
            def split_text(self, text: str) -> list[str]:
                chunks = []
                current_chunk = ""
                
                # Simple splitting by paragraphs then lines
                # This is a basic approximation of recursive splitting
                paragraphs = text.split("\n\n")
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) < self.chunk_size:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + "\n\n"
                        # If paragraph itself is too large, split by chunks
                        if len(current_chunk) > self.chunk_size:
                            raw_chunk = current_chunk
                            current_chunk = ""
                            while raw_chunk:
                                chunks.append(raw_chunk[:self.chunk_size])
                                raw_chunk = raw_chunk[self.chunk_size - self.chunk_overlap:]
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return chunks

        splitter = SimpleTextSplitter(
            chunk_size=self.settings.chunking.chunk_size,
            chunk_overlap=self.settings.chunking.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        )
        
        chunks = []
        chunk_index = 0
        
        for section in processed.sections:
            section_chunks = splitter.split_text(section.content)
            
            for chunk_text in section_chunks:
                if len(chunk_text) < self.settings.chunking.min_chunk_size:
                    continue
                
                chunks.append(Chunk(
                    id=f"{paper_id}_{chunk_index}",
                    content=chunk_text,
                    paper_id=paper_id,
                    section=section.name,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
        
        return chunks
