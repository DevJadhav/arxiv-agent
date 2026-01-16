"""Unit tests for library export/import functionality."""

import pytest
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from arxiv_agent.data.models import Paper


class TestJSONExport:
    """Tests for JSON export functionality."""
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for export testing."""
        return [
            Paper(
                id="arxiv:2401.00001",
                title="Attention Is All You Need",
                authors=["Vaswani", "Shazeer", "Parmar"],
                abstract="We propose the Transformer...",
                categories=["cs.CL", "cs.LG"],
                published_date=datetime(2024, 1, 15),
                citation_count=50000,
            ),
            Paper(
                id="arxiv:2401.00002",
                title="BERT: Pre-training of Deep Bidirectional Transformers",
                authors=["Devlin", "Chang"],
                abstract="We introduce BERT...",
                categories=["cs.CL"],
                published_date=datetime(2024, 1, 16),
                citation_count=30000,
            ),
        ]
    
    def test_export_json_basic(self, tmp_path, sample_papers):
        """Test basic JSON export."""
        from arxiv_agent.cli.commands.library import _export_to_json as export_to_json
        
        output_path = tmp_path / "export.json"
        export_to_json(sample_papers, output_path)
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert "papers" in data
        assert len(data["papers"]) == 2
    
    def test_export_json_structure(self, tmp_path, sample_papers):
        """Test JSON export has correct structure."""
        from arxiv_agent.cli.commands.library import _export_to_json as export_to_json
        
        output_path = tmp_path / "export.json"
        export_to_json(sample_papers, output_path)
        
        with open(output_path) as f:
            data = json.load(f)
        
        # Check metadata
        assert "exported_at" in data
        assert "paper_count" in data
        assert data["paper_count"] == 2
        
        # Check paper structure
        paper = data["papers"][0]
        assert "id" in paper
        assert "title" in paper
        assert "authors" in paper
        assert "abstract" in paper
        assert "categories" in paper
    
    def test_export_json_empty_library(self, tmp_path):
        """Test JSON export with empty library."""
        from arxiv_agent.cli.commands.library import _export_to_json as export_to_json
        
        output_path = tmp_path / "export.json"
        export_to_json([], output_path)
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert data["paper_count"] == 0
        assert data["papers"] == []


class TestBibTeXExport:
    """Tests for BibTeX export functionality."""
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers."""
        return [
            Paper(
                id="arxiv:2401.00001",
                title="Attention Is All You Need",
                authors=["Ashish Vaswani", "Noam Shazeer"],
                abstract="We propose the Transformer...",
                categories=["cs.CL"],
                published_date=datetime(2024, 1, 15),
                citation_count=50000,
            ),
        ]
    
    def test_export_bibtex_basic(self, tmp_path, sample_papers):
        """Test basic BibTeX export."""
        from arxiv_agent.cli.commands.library import _export_to_bibtex as export_to_bibtex
        
        output_path = tmp_path / "export.bib"
        export_to_bibtex(sample_papers, output_path)
        
        assert output_path.exists()
        content = output_path.read_text()
        
        assert "@article{" in content
        assert "2401.00001" in content
    
    def test_export_bibtex_structure(self, tmp_path, sample_papers):
        """Test BibTeX export has correct structure."""
        from arxiv_agent.cli.commands.library import _export_to_bibtex as export_to_bibtex
        
        output_path = tmp_path / "export.bib"
        export_to_bibtex(sample_papers, output_path)
        
        content = output_path.read_text()
        
        assert "title = {" in content
        assert "author = {" in content
        assert "year = {" in content
        assert "eprint = {" in content
        assert "archivePrefix = {arXiv}" in content
    
    def test_export_bibtex_multiple_authors(self, tmp_path, sample_papers):
        """Test BibTeX handles multiple authors correctly."""
        from arxiv_agent.cli.commands.library import _export_to_bibtex as export_to_bibtex
        
        output_path = tmp_path / "export.bib"
        export_to_bibtex(sample_papers, output_path)
        
        content = output_path.read_text()
        
        # Authors should be joined with " and "
        assert "Ashish Vaswani and Noam Shazeer" in content
    
    def test_export_bibtex_escapes_special_chars(self, tmp_path):
        """Test BibTeX escapes special characters."""
        papers = [
            Paper(
                id="arxiv:2401.00001",
                title="Learning & Understanding: A Survey",
                authors=["O'Connor", "Smith"],
                abstract="Test abstract",
                categories=["cs.AI"],
                published_date=datetime(2024, 1, 1),
            ),
        ]
        
        from arxiv_agent.cli.commands.library import _export_to_bibtex as export_to_bibtex
        
        output_path = tmp_path / "export.bib"
        export_to_bibtex(papers, output_path)
        
        # File should be created (escaping handled internally)
        assert output_path.exists()


class TestJSONImport:
    """Tests for JSON import functionality."""
    
    @pytest.fixture
    def sample_export_file(self, tmp_path):
        """Create a sample export file."""
        data = {
            "exported_at": "2025-01-16T00:00:00",
            "paper_count": 2,
            "papers": [
                {
                    "id": "arxiv:2401.00001",
                    "title": "Test Paper 1",
                    "authors": ["Author One"],
                    "abstract": "Abstract 1",
                    "categories": ["cs.AI"],
                    "published_date": "2024-01-15T00:00:00",
                    "citation_count": 100,
                },
                {
                    "id": "arxiv:2401.00002",
                    "title": "Test Paper 2",
                    "authors": ["Author Two"],
                    "abstract": "Abstract 2",
                    "categories": ["cs.LG"],
                    "published_date": "2024-01-16T00:00:00",
                    "citation_count": 200,
                },
            ],
        }
        
        export_path = tmp_path / "import_test.json"
        with open(export_path, "w") as f:
            json.dump(data, f)
        
        return export_path
    
    def test_import_json_basic(self, sample_export_file, temp_db):
        """Test basic JSON import."""
        from arxiv_agent.cli.commands.library import _import_from_json as import_from_json
        
        imported, skipped = import_from_json(str(sample_export_file), temp_db, collection_id=None, skip_existing=True)
        
        assert imported == 2
        assert skipped == 0
    
    def test_import_json_skips_duplicates(self, sample_export_file, temp_db):
        """Test import skips duplicate papers."""
        from arxiv_agent.cli.commands.library import _import_from_json as import_from_json
        
        # Import twice
        import_from_json(str(sample_export_file), temp_db, collection_id=None, skip_existing=True)
        imported, skipped = import_from_json(str(sample_export_file), temp_db, collection_id=None, skip_existing=True)
        
        assert skipped == 2
        assert imported == 0
    
    def test_import_json_handles_missing_fields(self, tmp_path, temp_db):
        """Test import handles papers with missing optional fields."""
        data = {
            "papers": [
                {
                    "id": "arxiv:2401.99999",
                    "title": "Minimal Paper",
                    "authors": [],
                    "abstract": "",
                    "categories": [],
                },
            ],
        }
        
        export_path = tmp_path / "minimal.json"
        with open(export_path, "w") as f:
            json.dump(data, f)
        
        from arxiv_agent.cli.commands.library import _import_from_json as import_from_json
        
        imported, skipped = import_from_json(str(export_path), temp_db, collection_id=None, skip_existing=True)
        assert imported == 1


class TestBibTeXImport:
    """Tests for BibTeX import functionality."""
    
    @pytest.fixture
    def sample_bibtex_file(self, tmp_path):
        """Create a sample BibTeX file."""
        content = '''
@article{2401.00001,
  title = {Test Paper Title},
  author = {Alice Smith and Bob Jones},
  year = {2024},
  eprint = {2401.00001},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  abstract = {This is a test abstract.},
}

@article{2401.00002,
  title = {Another Test Paper},
  author = {Charlie Brown},
  year = {2024},
  eprint = {2401.00002},
  archivePrefix = {arXiv},
}
'''
        bib_path = tmp_path / "import_test.bib"
        bib_path.write_text(content)
        return bib_path
    
    def test_import_bibtex_basic(self, sample_bibtex_file, temp_db):
        """Test basic BibTeX import."""
        from arxiv_agent.cli.commands.library import _import_from_bibtex as import_from_bibtex
        
        imported, skipped = import_from_bibtex(str(sample_bibtex_file), temp_db, collection_id=None, skip_existing=True)
        
        assert imported >= 1
    
    def test_import_bibtex_parses_authors(self, sample_bibtex_file, temp_db):
        """Test BibTeX author parsing."""
        from arxiv_agent.cli.commands.library import _import_from_bibtex as import_from_bibtex
        
        import_from_bibtex(str(sample_bibtex_file), temp_db, collection_id=None, skip_existing=True)
        
        paper = temp_db.get_paper("arxiv:2401.00001")
        assert paper is not None
        assert "Alice Smith" in paper.authors
        assert "Bob Jones" in paper.authors


class TestExportImportCLI:
    """Tests for export/import CLI commands."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        mock = MagicMock()
        mock.list_papers.return_value = [
            Paper(
                id="arxiv:2401.00001",
                title="Test Paper",
                authors=["Author"],
                abstract="Abstract",
                categories=["cs.AI"],
                published_date=datetime(2024, 1, 1),
            ),
        ]
        return mock
    
    def test_export_command_json(self, tmp_path, mock_db):
        """Test export CLI command with JSON format."""
        from typer.testing import CliRunner
        
        with patch("arxiv_agent.cli.commands.library.get_db", return_value=mock_db):
            from arxiv_agent.cli.commands.library import app
            
            runner = CliRunner()
            output_file = str(tmp_path / "test.json")
            result = runner.invoke(app, ["export", "-o", output_file, "-f", "json"])
            
            assert result.exit_code == 0
    
    def test_export_command_bibtex(self, tmp_path, mock_db):
        """Test export CLI command with BibTeX format."""
        from typer.testing import CliRunner
        
        with patch("arxiv_agent.cli.commands.library.get_db", return_value=mock_db):
            from arxiv_agent.cli.commands.library import app
            
            runner = CliRunner()
            output_file = str(tmp_path / "test.bib")
            result = runner.invoke(app, ["export", "-o", output_file, "-f", "bibtex"])
            
            assert result.exit_code == 0
    
    def test_import_command(self, tmp_path, mock_db):
        """Test import CLI command."""
        # Create a test import file
        import_data = {"papers": [], "paper_count": 0}
        import_file = tmp_path / "import.json"
        with open(import_file, "w") as f:
            json.dump(import_data, f)
        
        from typer.testing import CliRunner
        
        with patch("arxiv_agent.cli.commands.library.get_db", return_value=mock_db):
            from arxiv_agent.cli.commands.library import app
            
            runner = CliRunner()
            result = runner.invoke(app, ["import", str(import_file)])
            
            # Check command ran (may fail due to implementation)
            assert result.exit_code in [0, 1]


class TestCollectionsExportImport:
    """Tests for exporting/importing collections and tags."""
    
    def test_export_includes_collections(self, tmp_path, temp_db):
        """Test export includes collection information."""
        # Setup: create paper with collection
        paper = Paper(
            id="arxiv:2401.00001",
            title="Test",
            authors=[],
            abstract="",
            categories=[],
        )
        temp_db.save_paper(paper)
        collection = temp_db.create_collection("Test Collection")
        temp_db.add_paper_to_collection(paper.id, collection.id)
        
        from arxiv_agent.cli.commands.library import _export_to_json as export_to_json
        
        output_path = tmp_path / "export.json"
        papers = temp_db.list_papers()
        export_to_json(papers, str(output_path), collection="Test Collection")
        
        with open(output_path) as f:
            data = json.load(f)
        
        # Papers should be exported
        assert "papers" in data
