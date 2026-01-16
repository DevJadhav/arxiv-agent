"""Analyzer agent for deep paper analysis."""

from pathlib import Path

from loguru import logger

from arxiv_agent.agents.state import AgentState
from arxiv_agent.config.settings import get_settings
from arxiv_agent.core.api_client import get_api_client
from arxiv_agent.core.llm_service import get_llm_service, LLMService
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
    
    def compare_papers(
        self, 
        *papers,
        output_format: str = "json"
    ) -> dict | str:
        """Compare multiple papers and identify similarities/differences.
        
        Args:
            *papers: Two or more Paper objects to compare
            output_format: Output format - "json", "markdown", or "html"
            
        Returns:
            Comparison result as dict (json) or formatted string (markdown/html)
            
        Raises:
            ValueError: If fewer than 2 papers provided
        """
        import asyncio
        
        if len(papers) < 2:
            raise ValueError("Comparison requires at least two papers")
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # If we're already in an async context, use create_task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    self._compare_papers_async(papers, output_format)
                ).result()
        else:
            return asyncio.run(
                self._compare_papers_async(papers, output_format)
            )
    
    async def _compare_papers_async(
        self,
        papers: tuple,
        output_format: str
    ) -> dict | str:
        """Async implementation of paper comparison."""
        
        comparison_schema = {
            "type": "object",
            "properties": {
                "similarities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "aspect": {"type": "string"},
                            "description": {"type": "string"},
                            "significance": {"type": "string", "enum": ["high", "medium", "low"]}
                        },
                        "required": ["aspect", "description"]
                    }
                },
                "differences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "aspect": {"type": "string"},
                            "paper1_approach": {"type": "string"},
                            "paper2_approach": {"type": "string"},
                            "significance": {"type": "string", "enum": ["high", "medium", "low"]}
                        },
                        "required": ["aspect", "paper1_approach", "paper2_approach"]
                    }
                },
                "methodology_comparison": {
                    "type": "object",
                    "properties": {
                        "paper1": {"type": "string"},
                        "paper2": {"type": "string"}
                    }
                },
                "contribution_comparison": {
                    "type": "object",
                    "properties": {
                        "paper1": {"type": "string"},
                        "paper2": {"type": "string"}
                    }
                },
                "recommendation": {"type": "string"}
            },
            "required": ["similarities", "differences", "recommendation"]
        }
        
        # Build prompt with paper information
        papers_info = []
        for i, paper in enumerate(papers, 1):
            papers_info.append(f"""
Paper {i}: {paper.title}
ID: {paper.arxiv_id if hasattr(paper, 'arxiv_id') else paper.id}
Authors: {', '.join(paper.authors[:5])}{'...' if len(paper.authors) > 5 else ''}
Categories: {', '.join(paper.categories) if paper.categories else 'N/A'}
Abstract: {paper.abstract}
""")
        
        prompt = f"""Compare the following research papers and analyze their similarities and differences.

{chr(10).join(papers_info)}

Provide a comprehensive comparison including:
1. Key similarities between the papers
2. Major differences in approach, methodology, or focus
3. Methodology comparison for each paper
4. Contribution comparison
5. Recommendation for which paper to read first and why"""

        try:
            result = await self.llm.generate_json(
                prompt=prompt,
                schema=comparison_schema,
                system_prompt="You are an expert research paper analyst specializing in comparing academic papers.",
            )
        except Exception:
            # Fallback to regular generation if structured fails
            response = await self.llm.agenerate(prompt=prompt)
            result = {
                "similarities": [],
                "differences": [],
                "recommendation": response.content,
                "raw_comparison": response.content
            }
        
        # Format output
        if output_format == "json":
            return result
        elif output_format == "markdown":
            return self._format_comparison_markdown(result, papers)
        elif output_format == "html":
            return self._format_comparison_html(result, papers)
        else:
            return result
    
    def _format_comparison_markdown(self, comparison: dict, papers: tuple) -> str:
        """Format comparison result as markdown."""
        lines = [
            "# Paper Comparison",
            "",
            "## Papers Compared",
            "",
        ]
        
        for i, paper in enumerate(papers, 1):
            lines.append(f"{i}. **{paper.title}** ({paper.arxiv_id if hasattr(paper, 'arxiv_id') else paper.id})")
        
        lines.extend(["", "## Similarities", ""])
        for sim in comparison.get("similarities", []):
            sig = f" [{sim.get('significance', 'medium')}]" if 'significance' in sim else ""
            lines.append(f"- **{sim.get('aspect', 'Unknown')}**{sig}: {sim.get('description', '')}")
        
        lines.extend(["", "## Differences", ""])
        for diff in comparison.get("differences", []):
            lines.extend([
                f"### {diff.get('aspect', 'Unknown')}",
                f"- **Paper 1**: {diff.get('paper1_approach', 'N/A')}",
                f"- **Paper 2**: {diff.get('paper2_approach', 'N/A')}",
                ""
            ])
        
        if "methodology_comparison" in comparison:
            lines.extend(["## Methodology Comparison", ""])
            mc = comparison["methodology_comparison"]
            lines.extend([
                f"**Paper 1**: {mc.get('paper1', 'N/A')}",
                "",
                f"**Paper 2**: {mc.get('paper2', 'N/A')}",
                ""
            ])
        
        lines.extend([
            "## Recommendation",
            "",
            comparison.get("recommendation", "No recommendation available."),
        ])
        
        return "\n".join(lines)
    
    def _format_comparison_html(self, comparison: dict, papers: tuple) -> str:
        """Format comparison result as HTML."""
        html = ["<html><body>", "<h1>Paper Comparison</h1>"]
        
        html.append("<h2>Papers Compared</h2><ol>")
        for paper in papers:
            html.append(f"<li><strong>{paper.title}</strong></li>")
        html.append("</ol>")
        
        html.append("<h2>Similarities</h2><ul>")
        for sim in comparison.get("similarities", []):
            html.append(f"<li><strong>{sim.get('aspect', '')}</strong>: {sim.get('description', '')}</li>")
        html.append("</ul>")
        
        html.append("<h2>Differences</h2>")
        for diff in comparison.get("differences", []):
            html.append(f"<h3>{diff.get('aspect', '')}</h3>")
            html.append(f"<p><strong>Paper 1:</strong> {diff.get('paper1_approach', '')}</p>")
            html.append(f"<p><strong>Paper 2:</strong> {diff.get('paper2_approach', '')}</p>")
        
        html.append(f"<h2>Recommendation</h2><p>{comparison.get('recommendation', '')}</p>")
        html.append("</body></html>")
        
        return "\n".join(html)


# Paper-to-code scaffolding functions

def create_project_structure(project_dir: Path, code_plan: dict) -> None:
    """Create project directory structure from code plan.
    
    Args:
        project_dir: Root directory for the project
        code_plan: Dictionary containing:
            - architecture: Description of the architecture
            - components: List of component definitions with file paths
            - python_dependencies: List of required packages
            - implementation_order: Order to implement files
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure from components
    directories = set()
    for component in code_plan.get("components", []):
        file_path = component.get("file", "")
        if "/" in file_path:
            dir_path = project_dir / Path(file_path).parent
            directories.add(dir_path)
    
    # Create all directories
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        # Create __init__.py files
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Auto-generated module."""\n')
    
    # Create src directory if not in components
    src_dir = project_dir / "src"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "__init__.py").write_text('"""Source package."""\n')
    
    # Create tests directory
    tests_dir = project_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "__init__.py").write_text('"""Test package."""\n')
    
    # Create pyproject.toml (write directly without toml library)
    deps = code_plan.get("python_dependencies", [])
    deps_str = ",\n    ".join(f'"{d}"' for d in deps)
    
    pyproject_content = f'''[project]
name = "{project_dir.name}"
version = "0.1.0"
description = "{code_plan.get("architecture", "Generated project")}"
requires-python = ">=3.10"
dependencies = [
    {deps_str}
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
'''
    
    pyproject_path = project_dir / "pyproject.toml"
    pyproject_path.write_text(pyproject_content)
    
    # Create README.md
    readme_content = f"""# {project_dir.name}

{code_plan.get("architecture", "Generated from paper analysis.")}

## Components

"""
    for component in code_plan.get("components", []):
        readme_content += f"- **{component.get('name', 'Unknown')}**: {component.get('description', '')}\n"
    
    readme_content += """
## Installation

```bash
pip install -e .
```

## Usage

See individual component files for usage examples.
"""
    
    readme_path = project_dir / "README.md"
    readme_path.write_text(readme_content)
    
    logger.info(f"Created project structure at {project_dir}")


def validate_code(code: str) -> dict:
    """Validate Python code for syntax and basic correctness.
    
    Args:
        code: Python source code string
        
    Returns:
        Dictionary with validation results:
            - syntax_valid: bool - Whether code parses correctly
            - errors: list - List of error messages if any
            - imports: list - List of detected imports
            - classes: list - List of class names defined
            - functions: list - List of function names defined
    """
    import ast
    
    result = {
        "syntax_valid": False,
        "errors": [],
        "imports": [],
        "classes": [],
        "functions": [],
    }
    
    try:
        tree = ast.parse(code)
        result["syntax_valid"] = True
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    result["imports"].append(f"{module}.{alias.name}")
            elif isinstance(node, ast.ClassDef):
                result["classes"].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                result["functions"].append(node.name)
                
    except SyntaxError as e:
        result["errors"].append(f"Syntax error at line {e.lineno}: {e.msg}")
    except Exception as e:
        result["errors"].append(str(e))
    
    return result
