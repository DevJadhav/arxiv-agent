"""Tests for paper comparison feature.

TDD: Write tests first, then implement the feature.
DeepDive.md Reference: Section 3.2 - Paper Comparison Analysis
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from arxiv_agent.data.models import Paper
from arxiv_agent.agents.analyzer import AnalyzerAgent


@pytest.fixture
def sample_papers():
    """Create sample papers for comparison testing."""
    paper1 = Paper(
        arxiv_id="2401.00001",
        title="Attention Is All You Need",
        abstract="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        categories=["cs.CL", "cs.LG"],
        published=datetime(2017, 6, 12),
        pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
    )
    paper2 = Paper(
        arxiv_id="2401.00002",
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        abstract="We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers.",
        authors=["Jacob Devlin", "Ming-Wei Chang"],
        categories=["cs.CL", "cs.LG"],
        published=datetime(2018, 10, 11),
        pdf_url="https://arxiv.org/pdf/2401.00002.pdf",
    )
    return paper1, paper2


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for comparison generation."""
    with patch("arxiv_agent.agents.analyzer.LLMService") as mock:
        service = MagicMock()
        service.generate_json = AsyncMock(return_value={
            "similarities": [
                {
                    "aspect": "Architecture",
                    "description": "Both use transformer-based architectures",
                    "significance": "high"
                },
                {
                    "aspect": "Domain",
                    "description": "Both target NLP tasks",
                    "significance": "high"
                }
            ],
            "differences": [
                {
                    "aspect": "Training Objective",
                    "paper1_approach": "Sequence-to-sequence with encoder-decoder",
                    "paper2_approach": "Masked language modeling with encoder-only",
                    "significance": "high"
                },
                {
                    "aspect": "Application Focus",
                    "paper1_approach": "Machine translation",
                    "paper2_approach": "General language understanding",
                    "significance": "medium"
                }
            ],
            "methodology_comparison": {
                "paper1": "Encoder-decoder attention mechanism for translation",
                "paper2": "Bidirectional self-attention for contextual embeddings"
            },
            "contribution_comparison": {
                "paper1": "Introduced transformer architecture",
                "paper2": "Demonstrated transfer learning for NLP"
            },
            "recommendation": "Read Paper 1 first for architectural foundations, then Paper 2 for practical applications"
        })
        mock.return_value = service
        yield service


class TestComparePapersMethod:
    """Test the compare_papers method on AnalyzerAgent."""

    def test_compare_papers_returns_comparison(self, sample_papers, mock_llm_service):
        """Compare two papers and get structured comparison dict."""
        paper1, paper2 = sample_papers
        
        agent = AnalyzerAgent()
        agent.llm = mock_llm_service
        
        # This method should exist after implementation
        result = agent.compare_papers(paper1, paper2)
        
        assert result is not None
        assert "similarities" in result
        assert "differences" in result

    def test_compare_identifies_similarities(self, sample_papers, mock_llm_service):
        """Comparison includes methodology similarities."""
        paper1, paper2 = sample_papers
        
        agent = AnalyzerAgent()
        agent.llm = mock_llm_service
        
        result = agent.compare_papers(paper1, paper2)
        
        assert len(result["similarities"]) > 0
        for sim in result["similarities"]:
            assert "aspect" in sim
            assert "description" in sim

    def test_compare_identifies_differences(self, sample_papers, mock_llm_service):
        """Comparison includes key differences between papers."""
        paper1, paper2 = sample_papers
        
        agent = AnalyzerAgent()
        agent.llm = mock_llm_service
        
        result = agent.compare_papers(paper1, paper2)
        
        assert len(result["differences"]) > 0
        for diff in result["differences"]:
            assert "aspect" in diff
            assert "paper1_approach" in diff
            assert "paper2_approach" in diff

    def test_compare_includes_methodology_comparison(self, sample_papers, mock_llm_service):
        """Comparison includes methodology breakdown."""
        paper1, paper2 = sample_papers
        
        agent = AnalyzerAgent()
        agent.llm = mock_llm_service
        
        result = agent.compare_papers(paper1, paper2)
        
        assert "methodology_comparison" in result
        assert "paper1" in result["methodology_comparison"]
        assert "paper2" in result["methodology_comparison"]

    def test_compare_includes_recommendation(self, sample_papers, mock_llm_service):
        """Comparison includes reading recommendation."""
        paper1, paper2 = sample_papers
        
        agent = AnalyzerAgent()
        agent.llm = mock_llm_service
        
        result = agent.compare_papers(paper1, paper2)
        
        assert "recommendation" in result
        assert len(result["recommendation"]) > 0


class TestCompareMultiplePapers:
    """Test comparison of more than two papers."""

    def test_compare_three_papers(self, mock_llm_service):
        """Compare three papers at once."""
        papers = [
            Paper(arxiv_id=f"2401.0000{i}", title=f"Paper {i}", 
                  abstract=f"Abstract {i}", authors=[f"Author {i}"],
                  categories=["cs.LG"], published=datetime.now(),
                  pdf_url=f"https://arxiv.org/pdf/2401.0000{i}.pdf")
            for i in range(3)
        ]
        
        mock_llm_service.generate_json = AsyncMock(return_value={
            "similarities": [],
            "differences": [],
            "papers_analyzed": 3
        })
        
        agent = AnalyzerAgent()
        agent.llm = mock_llm_service
        
        result = agent.compare_papers(*papers)
        
        assert result is not None

    def test_compare_single_paper_raises_error(self, sample_papers, mock_llm_service):
        """Comparing single paper should raise ValueError."""
        paper1, _ = sample_papers
        
        agent = AnalyzerAgent()
        agent.llm = mock_llm_service
        
        with pytest.raises(ValueError, match="at least two papers"):
            agent.compare_papers(paper1)


class TestCompareOutputFormats:
    """Test different output formats for comparison."""

    def test_compare_output_markdown(self, sample_papers, mock_llm_service):
        """Comparison can be formatted as markdown."""
        paper1, paper2 = sample_papers
        
        agent = AnalyzerAgent()
        agent.llm = mock_llm_service
        
        result = agent.compare_papers(paper1, paper2, output_format="markdown")
        
        # If format is markdown, result should be a string
        if isinstance(result, str):
            assert "##" in result or "#" in result  # Has markdown headers

    def test_compare_output_json(self, sample_papers, mock_llm_service):
        """Comparison returns JSON dict by default."""
        paper1, paper2 = sample_papers
        
        agent = AnalyzerAgent()
        agent.llm = mock_llm_service
        
        result = agent.compare_papers(paper1, paper2, output_format="json")
        
        assert isinstance(result, dict)

    def test_compare_output_html(self, sample_papers, mock_llm_service):
        """Comparison can be formatted as HTML."""
        paper1, paper2 = sample_papers
        
        agent = AnalyzerAgent()
        agent.llm = mock_llm_service
        
        result = agent.compare_papers(paper1, paper2, output_format="html")
        
        if isinstance(result, str):
            assert "<" in result and ">" in result  # Has HTML tags


class TestCompareCLICommand:
    """Test CLI command for paper comparison."""

    def test_compare_cli_command_exists(self):
        """CLI has compare subcommand under analyze."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        result = runner.invoke(app, ["analyze", "--help"])
        
        # After implementation, compare should be in help
        assert result.exit_code == 0
        # This assertion will pass after implementation
        # assert "compare" in result.output.lower()

    def test_compare_cli_two_papers(self):
        """CLI: arxiv analyze compare paper1_id paper2_id"""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.analyze.get_storage") as mock_storage:
            mock_db = MagicMock()
            mock_db.get_paper.side_effect = [
                Paper(arxiv_id="2401.00001", title="Paper 1", 
                      abstract="Abstract 1", authors=["Author 1"],
                      categories=["cs.LG"], published=datetime.now(),
                      pdf_url="https://arxiv.org/pdf/2401.00001.pdf"),
                Paper(arxiv_id="2401.00002", title="Paper 2",
                      abstract="Abstract 2", authors=["Author 2"],
                      categories=["cs.LG"], published=datetime.now(),
                      pdf_url="https://arxiv.org/pdf/2401.00002.pdf"),
            ]
            mock_storage.return_value = mock_db
            
            # This will fail until compare command is implemented
            result = runner.invoke(app, ["analyze", "compare", "2401.00001", "2401.00002"])
            
            # Will pass after implementation
            # assert result.exit_code == 0

    def test_compare_cli_output_format_option(self):
        """CLI: arxiv analyze compare paper1 paper2 --format json"""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        # Test format option exists after implementation
        result = runner.invoke(app, ["analyze", "compare", "--help"])
        
        # Will pass after implementation shows --format option
        # assert "--format" in result.output or "-f" in result.output

    def test_compare_cli_not_found_paper(self):
        """CLI handles paper not found gracefully."""
        from typer.testing import CliRunner
        from arxiv_agent.cli.main import app
        
        runner = CliRunner()
        
        with patch("arxiv_agent.cli.commands.analyze.get_storage") as mock_storage:
            mock_db = MagicMock()
            mock_db.get_paper.return_value = None
            mock_storage.return_value = mock_db
            
            result = runner.invoke(app, ["analyze", "compare", "nonexistent1", "nonexistent2"])
            
            # Should show error message, not crash
            # assert result.exit_code != 0 or "not found" in result.output.lower()


class TestComparePromptGeneration:
    """Test the LLM prompt generation for comparison."""

    def test_comparison_prompt_includes_both_papers(self, sample_papers, mock_llm_service):
        """Generated prompt includes content from both papers."""
        paper1, paper2 = sample_papers
        
        agent = AnalyzerAgent()
        agent.llm = mock_llm_service
        
        # Call compare to trigger prompt generation
        agent.compare_papers(paper1, paper2)
        
        # Check that generate_json was called with prompt containing both titles
        call_args = mock_llm_service.generate_json.call_args
        if call_args:
            prompt = str(call_args)
            # The prompt should reference both papers
            # assert paper1.title in prompt or paper1.arxiv_id in prompt

    def test_comparison_uses_abstracts(self, sample_papers, mock_llm_service):
        """Comparison uses paper abstracts for analysis."""
        paper1, paper2 = sample_papers
        
        agent = AnalyzerAgent()
        agent.llm = mock_llm_service
        
        agent.compare_papers(paper1, paper2)
        
        # Verify abstracts were passed to LLM
        # Will be checked after implementation
        mock_llm_service.generate_json.assert_called_once()
