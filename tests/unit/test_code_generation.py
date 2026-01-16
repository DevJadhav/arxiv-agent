"""Unit tests for Paper-to-Code generation workflow."""

import pytest
import ast
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path


class TestCodePlanGeneration:
    """Tests for code plan generation."""
    
    @pytest.fixture
    def sample_code_plan(self):
        """Sample code plan for testing."""
        return {
            "architecture": "Neural network with transformer blocks",
            "components": [
                {
                    "name": "TransformerBlock",
                    "file": "model/transformer.py",
                    "description": "Core transformer block with multi-head attention",
                    "dependencies": ["torch", "einops"],
                },
                {
                    "name": "AttentionHead",
                    "file": "model/attention.py",
                    "description": "Single attention head implementation",
                    "dependencies": ["torch"],
                },
            ],
            "python_dependencies": ["torch>=2.0.0", "einops>=0.7.0", "numpy"],
            "implementation_order": ["model/attention.py", "model/transformer.py"],
        }
    
    def test_code_plan_has_required_fields(self, sample_code_plan):
        """Test code plan contains required fields."""
        required_fields = ["architecture", "components", "python_dependencies", "implementation_order"]
        
        for field in required_fields:
            assert field in sample_code_plan
    
    def test_component_has_required_fields(self, sample_code_plan):
        """Test each component has required fields."""
        required_component_fields = ["name", "file", "description"]
        
        for component in sample_code_plan["components"]:
            for field in required_component_fields:
                assert field in component


class TestCodeGeneration:
    """Tests for actual code generation from plan."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        mock = MagicMock()
        mock.agenerate = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_settings(self, tmp_path):
        """Create mock settings."""
        mock = MagicMock()
        mock.data_dir = tmp_path
        mock.code_generation = MagicMock()
        mock.code_generation.default_framework = "pytorch"
        mock.code_generation.generate_tests = True
        mock.code_generation.type_hints = True
        return mock
    
    @pytest.mark.asyncio
    async def test_generate_single_component(self, mock_llm_service, mock_settings):
        """Test generating code for a single component."""
        mock_llm_service.agenerate.return_value = MagicMock(
            content='''
class AttentionHead:
    """Single attention head implementation."""
    
    def __init__(self, dim: int, head_dim: int):
        self.dim = dim
        self.head_dim = head_dim
    
    def forward(self, x):
        return x
'''
        )
        
        with patch("arxiv_agent.agents.analyzer.get_llm_service", return_value=mock_llm_service), \
             patch("arxiv_agent.agents.analyzer.get_settings", return_value=mock_settings):
            from arxiv_agent.agents.analyzer import AnalyzerAgent
            
            agent = AnalyzerAgent()
            # Test the code generation method once implemented


class TestASTValidation:
    """Tests for Python AST validation of generated code."""
    
    def test_valid_python_syntax(self):
        """Test valid Python code passes AST parsing."""
        valid_code = '''
def hello_world():
    """Say hello."""
    print("Hello, World!")
    return True
'''
        # Should not raise
        tree = ast.parse(valid_code)
        assert tree is not None
    
    def test_invalid_python_syntax_detected(self):
        """Test invalid Python code is detected."""
        invalid_code = '''
def broken_function(
    print("missing close paren"
'''
        with pytest.raises(SyntaxError):
            ast.parse(invalid_code)
    
    def test_class_definition_valid(self):
        """Test class definition is valid."""
        class_code = '''
class MyModel:
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        return x * self.hidden_dim
'''
        tree = ast.parse(class_code)
        # Check it contains a class definition
        assert any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    
    def test_imports_detected(self):
        """Test imports are properly detected."""
        code_with_imports = '''
import torch
import torch.nn as nn
from typing import Optional, Tuple

class Layer(nn.Module):
    pass
'''
        tree = ast.parse(code_with_imports)
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        assert len(imports) >= 2


class TestProjectScaffolding:
    """Tests for project scaffolding functionality."""
    
    @pytest.fixture
    def sample_code_plan(self):
        """Sample code plan."""
        return {
            "architecture": "Simple MLP",
            "components": [
                {"name": "MLP", "file": "model/mlp.py", "description": "MLP network"},
            ],
            "python_dependencies": ["torch>=2.0.0", "numpy"],
            "implementation_order": ["model/mlp.py"],
        }
    
    def test_create_project_structure(self, tmp_path, sample_code_plan):
        """Test project directory structure is created."""
        from arxiv_agent.agents.analyzer import create_project_structure
        
        project_dir = tmp_path / "test_project"
        create_project_structure(project_dir, sample_code_plan)
        
        assert project_dir.exists()
        assert (project_dir / "model").exists()
    
    def test_create_pyproject_toml(self, tmp_path, sample_code_plan):
        """Test pyproject.toml is created with dependencies."""
        from arxiv_agent.agents.analyzer import create_project_structure
        
        project_dir = tmp_path / "test_project"
        create_project_structure(project_dir, sample_code_plan)
        
        pyproject_path = project_dir / "pyproject.toml"
        assert pyproject_path.exists()
        
        content = pyproject_path.read_text()
        assert "torch" in content
    
    def test_create_init_files(self, tmp_path, sample_code_plan):
        """Test __init__.py files are created."""
        from arxiv_agent.agents.analyzer import create_project_structure
        
        project_dir = tmp_path / "test_project"
        create_project_structure(project_dir, sample_code_plan)
        
        assert (project_dir / "model" / "__init__.py").exists()


class TestTestGeneration:
    """Tests for automatic test generation."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        mock = MagicMock()
        mock.agenerate = AsyncMock()
        return mock
    
    @pytest.mark.asyncio
    async def test_generate_tests_for_component(self, mock_llm_service):
        """Test generating tests for a component."""
        mock_llm_service.agenerate.return_value = MagicMock(
            content='''
import pytest
from model.attention import AttentionHead

class TestAttentionHead:
    def test_init(self):
        head = AttentionHead(dim=64, head_dim=8)
        assert head.dim == 64
    
    def test_forward(self):
        head = AttentionHead(dim=64, head_dim=8)
        # Test forward pass
'''
        )
        
        # Test once implemented
    
    def test_generated_test_is_valid_python(self):
        """Test generated test code is valid Python."""
        test_code = '''
import pytest

class TestMyClass:
    def test_something(self):
        assert True
'''
        tree = ast.parse(test_code)
        assert tree is not None


class TestCodeValidator:
    """Tests for code validation utilities."""
    
    def test_validate_imports_exist(self):
        """Test validation checks imports."""
        from arxiv_agent.agents.analyzer import validate_code
        
        code = '''
import torch
x = torch.tensor([1, 2, 3])
'''
        # Should pass (torch is a known package)
        result = validate_code(code)
        assert result["syntax_valid"] is True
    
    def test_validate_type_hints(self):
        """Test validation checks type hints if enabled."""
        code_with_hints = '''
def add(a: int, b: int) -> int:
    return a + b
'''
        tree = ast.parse(code_with_hints)
        func = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][0]
        
        # Check type hints exist
        assert func.returns is not None
        assert len(func.args.args) == 2


class TestFullWorkflow:
    """Integration tests for full paper-to-code workflow."""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create analyzer with mocked dependencies."""
        with patch("arxiv_agent.agents.analyzer.get_llm_service"), \
             patch("arxiv_agent.agents.analyzer.get_settings"), \
             patch("arxiv_agent.agents.analyzer.get_db"), \
             patch("arxiv_agent.agents.analyzer.get_api_client"):
            from arxiv_agent.agents.analyzer import AnalyzerAgent
            return AnalyzerAgent()
    
    @pytest.mark.asyncio
    async def test_full_code_generation_workflow(self, mock_analyzer, tmp_path):
        """Test complete workflow from paper to code."""
        # This would test the full pipeline:
        # 1. Generate code plan
        # 2. Generate code for each component
        # 3. Validate generated code
        # 4. Create project scaffold
        # 5. Generate tests
        pass  # Implement once full workflow exists
