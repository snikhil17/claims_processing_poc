"""
Document text extraction module
"""

import os
from pathlib import Path
from typing import Optional
import logging

# Set availability flags to True initially - we'll check imports at runtime
DOCLING_AVAILABLE = True
PYPDF_AVAILABLE = True  
OCR_AVAILABLE = True

print("ℹ️ Module-level imports skipped - will import at runtime")

logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Extract text from various document formats"""
    
    def __init__(self):
        """Initialize the document extractor"""
        
        # Try to initialize Docling converter at runtime
        self.converter = None
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions  
            from docling.datamodel.base_models import InputFormat
            
            # Configure pipeline for advanced processing
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("DocumentExtractor initialized with Docling")
            
        except ImportError as e:
            logger.warning(f"Docling not available during initialization: {e}")
            self.converter = None
            logger.info("DocumentExtractor initialized with fallback methods")
    
    def extract_text(self, file_path: str) -> Optional[str]:
        """
        Extract text from a document file
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text or None if extraction fails
        """
        
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        file_ext = file_path_obj.suffix.lower()
        
        try:
            if file_ext == '.pdf':
                return self._extract_from_pdf(file_path_obj)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                return self._extract_from_image(file_path_obj)
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return None
    
    def _extract_from_pdf(self, file_path: Path) -> Optional[str]:
        """Extract text from PDF using best available method"""
        
        # Try Docling first (most advanced)
        if self.converter:
            try:
                result = self.converter.convert(str(file_path))
                text = result.document.export_to_markdown()
                if text and len(text.strip()) > 50:
                    logger.info(f"Extracted {len(text)} characters using Docling")
                    return text
            except Exception as e:
                logger.warning(f"Docling extraction failed: {e}")
        
        # Fallback to pypdf
        try:
            import pypdf
            
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\\n"
            
            if text and len(text.strip()) > 50:
                logger.info(f"Extracted {len(text)} characters using pypdf")
                return text
                
        except ImportError as e:
            logger.warning(f"pypdf not available: {e}")
        except Exception as e:
            logger.warning(f"pypdf extraction failed: {e}")
        
        # Last resort: OCR if available
        try:
            from PIL import Image
            import pytesseract
            
            # Convert PDF to images and OCR
            logger.info("Attempting OCR extraction...")
            # This is a basic OCR approach
            # In production, you'd convert PDF pages to images first
            return "OCR extraction not fully implemented in this version"
            
        except ImportError as e:
            logger.warning(f"OCR not available for PDF: {e}")
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
        
        logger.error(f"All extraction methods failed for {file_path}")
        return None
    
    def _extract_from_image(self, file_path: Path) -> Optional[str]:
        """Extract text from image using OCR with improved accuracy"""
        
        logger.info(f"Image extraction - converter available: {self.converter is not None}")
        
        # Try Docling first for images
        if self.converter:
            try:
                logger.info("Attempting Docling extraction...")
                result = self.converter.convert(str(file_path))
                text = result.document.export_to_markdown()
                if text and len(text.strip()) > 10:
                    logger.info(f"Extracted {len(text)} characters from image using Docling")
                    # Save OCR output to file
                    self._save_ocr_output(file_path, text)
                    return text
                else:
                    logger.warning(f"Docling returned insufficient text: {len(text) if text else 0} chars")
            except Exception as e:
                logger.warning(f"Docling image extraction failed: {e}")
        else:
            logger.warning("Docling converter not available")
        
        # Fallback to pytesseract with improved configuration
        try:
            from PIL import Image
            import pytesseract
            
            logger.info("Attempting pytesseract extraction with enhanced config...")
            
            # Open and preprocess image
            image = Image.open(file_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image for better OCR
            # Increase contrast and sharpness
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)  # Increase contrast
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)  # Increase sharpness
            
            # Configure pytesseract for better accuracy
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$.,/-+%() '
            
            text = pytesseract.image_to_string(image, config=custom_config)
            
            if text and len(text.strip()) > 10:
                logger.info(f"Extracted {len(text)} characters using enhanced OCR")
                # Save OCR output to file
                self._save_ocr_output(file_path, text)
                return text
            else:
                logger.warning(f"Pytesseract returned insufficient text: {len(text) if text else 0} chars")
                
        except ImportError as e:
            logger.warning(f"Pytesseract not available: {e}")
        except Exception as e:
            logger.warning(f"Enhanced OCR extraction failed: {e}")
        
        logger.error(f"No image extraction method available for {file_path}")
        return None
    
    def _save_ocr_output(self, file_path: Path, text: str):
        """Save OCR output to a file in the ocr_output directory"""
        try:
            # Create ocr_output directory if it doesn't exist
            ocr_dir = Path(__file__).parent.parent / "data" / "ocr_output"
            ocr_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename based on input file
            base_name = file_path.stem
            output_file = ocr_dir / f"{base_name}_ocr.txt"
            
            # Save the text
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"OCR output saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save OCR output for {file_path}: {e}")
