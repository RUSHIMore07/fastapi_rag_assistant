import PyPDF2
import docx
from typing import Dict, Any, List
import logging
import io

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ["pdf", "txt", "docx"]
    
    def process_pdf(self, file_content: bytes) -> str:
        """Process PDF document and return text content"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text  # Return string, not dict
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return ""  # Return empty string on error
    
    def process_docx(self, file_content: bytes) -> str:
        """Process DOCX document and return text content"""
        try:
            doc = docx.Document(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text  # Return string, not dict
            
        except Exception as e:
            logger.error(f"Error processing DOCX: {e}")
            return ""  # Return empty string on error
    
    def process_text(self, file_content: bytes) -> str:
        """Process plain text document and return text content"""
        try:
            text = file_content.decode('utf-8')
            return text  # Return string, not dict
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return ""  # Return empty string on error
