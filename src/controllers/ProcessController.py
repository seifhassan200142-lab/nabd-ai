import os
from config import Config
from .BaseController import BaseController
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class ProcessController(BaseController):
    
    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        self.project_path = os.path.join(self.files_dir, project_id)
        
    def process_medical_document(self, file_name: str):
        file_path = os.path.join(self.project_path, file_name)
        
        if not os.path.exists(file_path):
            return None
            
        print(f"[*] Starting Multimodal Parsing for: {file_name}")
            
        # Initialize processing blueprint parameters for table and layout isolation
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        result = converter.convert(file_path)
        
        # Save structural bounding boxes and images within the isolated patient repository
        image_counter = 1
        for element, _ in result.document.iterate_items():
            if hasattr(element, "image") and element.image is not None:
                if hasattr(element.image, "pil_image") and element.image.pil_image is not None:
                    image_filename = f"image_{image_counter}.png"
                    image_path = os.path.join(self.project_path, image_filename)
                    
                    element.image.pil_image.save(image_path)
                    image_counter += 1
                    
        # Map generated images securely into the structural markdown text stream
        placeholder = "[IMAGE_PLACEHOLDER_TOKEN]"
        markdown_text = result.document.export_to_markdown(image_placeholder=placeholder)
        
        for i in range(1, image_counter):
            image_filename = f"image_{i}.png"
            markdown_text = markdown_text.replace(placeholder, f"![Image]({image_filename})", 1)
            
        return markdown_text
    
    def chunk_medical_document(self, markdown_text: str, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        print("\n[*] Starting intelligent Medical Chunking...")
        
        # Step 1: Hierarchical Markdown splitting to prevent context fragmentation
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_text)
        
        # Step 2: Recursive processing to downscale oversized sub-blocks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        final_chunks = text_splitter.split_documents(md_header_splits)
        print(f"[+] Chunking complete. Created {len(final_chunks)} medical chunks.")
        
        return final_chunks
    
    