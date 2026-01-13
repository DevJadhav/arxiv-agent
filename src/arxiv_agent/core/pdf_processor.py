"""PDF processing service with section extraction."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from arxiv_agent.config.settings import get_settings


@dataclass
class Section:
    """Represents a paper section."""
    
    name: str
    content: str
    page_start: int
    page_end: int


@dataclass
class ProcessedPaper:
    """Fully processed paper content."""
    
    full_text: str
    markdown: str
    sections: list[Section]
    page_count: int
    pdf_hash: str


class PDFProcessor:
    """PDF processing with section awareness using PyMuPDF4LLM."""
    
    SECTION_PATTERNS = [
        # Markdown-style headers
        r"^#{1,3}\s*(abstract|introduction|related\s*work|background|"
        r"methodology|methods?|approach|experiments?|results?|"
        r"discussion|conclusion|references?|appendix|acknowledgments?)",
        # Numbered sections
        r"^\d+\.?\s*(abstract|introduction|related\s*work|background|"
        r"methodology|methods?|approach|experiments?|results?|"
        r"discussion|conclusion|references?|appendix|acknowledgments?)",
        # All caps headers
        r"^(ABSTRACT|INTRODUCTION|RELATED\s*WORK|BACKGROUND|"
        r"METHODOLOGY|METHODS?|APPROACH|EXPERIMENTS?|RESULTS?|"
        r"DISCUSSION|CONCLUSION|REFERENCES?|APPENDIX|ACKNOWLEDGMENTS?)\s*$",
    ]
    
    def __init__(self):
        """Initialize PDF processor."""
        self.settings = get_settings()
    
    def compute_hash(self, pdf_path: str) -> str:
        """Compute SHA256 hash of PDF file for deduplication.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            SHA256 hash string
        """
        hasher = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def extract_markdown(self, pdf_path: str) -> str:
        """Extract markdown from PDF using PyMuPDF4LLM.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Markdown text content
        """
        try:
            import pymupdf4llm
            
            md_text = pymupdf4llm.to_markdown(
                pdf_path,
                page_chunks=False,
                write_images=False,
            )
            logger.debug(f"Extracted {len(md_text)} chars from {pdf_path}")
            return md_text
        except Exception as e:
            logger.error(f"Failed to extract markdown from {pdf_path}: {e}")
            raise
    
    def extract_sections(self, markdown: str) -> list[Section]:
        """Extract sections from markdown content.
        
        Args:
            markdown: Markdown text content
        
        Returns:
            List of Section objects
        """
        sections = []
        current_section = "preamble"
        current_content: list[str] = []
        current_page = 1
        section_start_page = 1
        
        for line in markdown.split("\n"):
            # Track page markers (PyMuPDF4LLM uses -----\n for page breaks)
            if line.strip() == "-----":
                current_page += 1
                continue
            
            # Check for section headers
            section_match = None
            line_stripped = line.strip()
            
            for pattern in self.SECTION_PATTERNS:
                if match := re.match(pattern, line_stripped, re.IGNORECASE):
                    section_match = match.group(1).strip().lower()
                    break
            
            if section_match:
                # Save previous section
                if current_content:
                    sections.append(Section(
                        name=current_section,
                        content="\n".join(current_content).strip(),
                        page_start=section_start_page,
                        page_end=current_page,
                    ))
                
                current_section = section_match
                current_content = [line]
                section_start_page = current_page
            else:
                current_content.append(line)
        
        # Save final section
        if current_content:
            sections.append(Section(
                name=current_section,
                content="\n".join(current_content).strip(),
                page_start=section_start_page,
                page_end=current_page,
            ))
        
        logger.debug(f"Extracted {len(sections)} sections")
        return sections
    
    def get_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Number of pages
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logger.warning(f"Failed to get page count: {e}")
            return 0
    
    def extract_plain_text(self, markdown: str) -> str:
        """Convert markdown to plain text.
        
        Args:
            markdown: Markdown text
        
        Returns:
            Plain text without markdown formatting
        """
        # Remove markdown formatting
        text = re.sub(r"#+\s*", "", markdown)  # Headers
        text = re.sub(r"\*+", "", text)  # Bold/italic
        text = re.sub(r"`+", "", text)  # Code
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # Links
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)  # Images
        text = re.sub(r"-----", "", text)  # Page breaks
        text = re.sub(r"\s+", " ", text)  # Collapse whitespace
        return text.strip()
    
    def process(self, pdf_path: str) -> ProcessedPaper:
        """Fully process a PDF file.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            ProcessedPaper with all extracted content
        
        Raises:
            FileNotFoundError: If PDF doesn't exist
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Compute hash for deduplication
        pdf_hash = self.compute_hash(pdf_path)
        
        # Extract content
        markdown = self.extract_markdown(pdf_path)
        
        # Get page count
        page_count = self.get_page_count(pdf_path)
        
        # Extract sections
        sections = self.extract_sections(markdown)
        
        # Generate plain text
        full_text = self.extract_plain_text(markdown)
        
        return ProcessedPaper(
            full_text=full_text,
            markdown=markdown,
            sections=sections,
            page_count=page_count,
            pdf_hash=pdf_hash,
        )
    
    def get_section_by_name(self, processed: ProcessedPaper, section_name: str) -> Section | None:
        """Get a specific section by name.
        
        Args:
            processed: Processed paper
            section_name: Section name to find (case-insensitive)
        
        Returns:
            Section object or None
        """
        section_name_lower = section_name.lower()
        for section in processed.sections:
            if section.name.lower() == section_name_lower:
                return section
        return None


# Global instance
_processor: PDFProcessor | None = None


def get_pdf_processor() -> PDFProcessor:
    """Get PDF processor instance."""
    global _processor
    if _processor is None:
        _processor = PDFProcessor()
    return _processor
