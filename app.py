import streamlit as st
import time
import fitz
from pathlib import Path
import tempfile
import os
import uuid
import shutil
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline
import PyPDF2
from typing import List
import atexit


# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    layout="wide"
)

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    def extract_text(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1  # +1 for space
            if current_size > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

class EmbeddingsHandler:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.embeddings = []
        self.ids = []
        self.chunks = []
    
    def create_embeddings(self, chunks: list, pdf_name: str):
        """Create embeddings for text chunks and store them in Faiss index"""
        chunk_embeddings = self.model.encode(chunks)
        self.index.add(np.array(chunk_embeddings).astype(np.float32))
        self.embeddings.extend(chunk_embeddings)
        self.ids.extend([f"chunk_{i}_{pdf_name}" for i in range(len(chunks))])
        self.chunks.extend(chunks)
    
    def search_similar_chunks(self, query: str, n_results: int = 3) -> list:
        """Search for chunks similar to the query using Faiss"""
        query_embedding = self.model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_embedding, n_results)
        results = [self.chunks[idx] for idx in indices[0]]
        return results

class LLMHandler:
    def __init__(self):
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )
    
    def get_answer(self, question: str, context: str) -> str:
        """Get answer from the LLM"""
        result = self.qa_pipeline(
            question=question,
            context=context,
        )
        return result['answer']

def clean_text(text):
    """Clean text for better matching"""
    text = ' '.join(text.split())
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

def find_text_segments(full_text, search_text, max_segment_length=100):
    """Find smaller segments of text that are part of the larger chunk"""
    clean_search = clean_text(search_text)
    words = clean_search.split()
    segments = []
    
    for i in range(len(words)):
        for j in range(3, 6):
            if i + j <= len(words):
                segment = ' '.join(words[i:i+j])
                if len(segment) <= max_segment_length:
                    segments.append(segment)
    
    return segments

def highlight_pdf(pdf_path, text_to_highlight, source_index):
    """Highlight text in the original PDF with incremental save"""
    try:
        doc = fitz.open(pdf_path)
        
        colors = [(1, 0.85, 0), (0.5, 1, 0.5), (1, 0.7, 0.7), (0.7, 0.7, 1)]
        color = colors[source_index % len(colors)]
        
        clean_chunk = clean_text(text_to_highlight)
        segments = find_text_segments(clean_chunk, text_to_highlight)
        found_match = False
        
        for page in doc:
            # Try exact matching first
            text_instances = page.search_for(text_to_highlight)
            
            # If no exact matches, try with cleaned text
            if not text_instances:
                text_instances = page.search_for(clean_chunk)
            
            # If still no matches, try with smaller segments
            if not text_instances:
                for segment in segments:
                    segment_instances = page.search_for(segment)
                    for inst in segment_instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(stroke=color)
                        highlight.update()
                        found_match = True
            else:
                # Highlight exact or cleaned text matches
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=color)
                    highlight.update()
                    found_match = True
        
        if found_match:
            doc.saveIncr()
        doc.close()
        return found_match
            
    except Exception as e:
        st.error(f"Error highlighting PDF: {str(e)}")
        return False

if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = Path(tempfile.gettempdir()) / f'pdf_chat_{uuid.uuid4()}'
    st.session_state.temp_dir.mkdir(exist_ok=True)

if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = {}
    st.session_state.processor = PDFProcessor()
    st.session_state.embeddings = EmbeddingsHandler()
    st.session_state.llm = LLMHandler()

st.markdown("""
    <style>
    .css-1p05t8e {border-radius: 10px}
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
        color: #000000;
        caret-color: #000000;
    }
    .stTextInput>div>div>input::placeholder {
        color: #888888;
    }
    .uploadedFile {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .pdf-preview {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload your PDFs", type=['pdf'], accept_multiple_files=True)
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_pdfs:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                pdf_id = uuid.uuid4()
                pdf_path = st.session_state.temp_dir / f"original_{pdf_id}_{uploaded_file.name}"
                
                with open(pdf_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                text = st.session_state.processor.extract_text(uploaded_file)
                chunks = st.session_state.processor.split_text(text)
                st.session_state.embeddings.create_embeddings(chunks, uploaded_file.name)
                
                st.session_state.processed_pdfs[uploaded_file.name] = {
                    'path': str(pdf_path),
                    'chunks': chunks,
                    'id': str(pdf_id)
                }
                
                st.success(f"{uploaded_file.name} processed successfully!")
    
    if st.session_state.processed_pdfs:
        st.header("Processed Documents")
        for pdf_name, pdf_info in st.session_state.processed_pdfs.items():
            st.write(f"✓ {pdf_name}")
            with open(pdf_info['path'], "rb") as f:
                st.download_button(
                    label=f"Download {pdf_name}",
                    data=f.read(),
                    file_name=pdf_name,
                    mime="application/pdf",
                    key=f"original_{pdf_info['id']}"
                )

if st.session_state.processed_pdfs:
    st.header("Ask questions about your PDFs")
    question = st.text_input("Enter your question:", placeholder="What would you like to know about the documents?")
    
    if question:
        with st.spinner("Thinking..."):
            similar_chunks = st.session_state.embeddings.search_similar_chunks(question)
            answer = st.session_state.llm.get_answer(question, " ".join(similar_chunks))
            
            st.markdown("### Answer")
            st.write(answer)
            
            with st.expander("View source context"):
                for i, chunk in enumerate(similar_chunks, 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(f"*{chunk}*")
                    
                    for pdf_name, pdf_info in st.session_state.processed_pdfs.items():
                        if chunk in pdf_info['chunks']:
                            button_key = f"download_{pdf_name}_{i}"
                            
                            found_highlight = highlight_pdf(pdf_info['path'], chunk, i-1)
                            
                            st.markdown(f"**From document: {pdf_name}**")
                            
                            with open(pdf_info['path'], "rb") as f:
                                button_label = "Download PDF with highlights" if found_highlight else "Download PDF"
                                st.download_button(
                                    label=button_label,
                                    data=f.read(),
                                    file_name=pdf_name,
                                    mime="application/pdf",
                                    key=button_key
                                )
                    st.markdown("---")
else:
    st.markdown("""
        Welcome to PDF Chat Assistant!
        
        To get started:
        1. Upload one or more PDF documents using the sidebar
        2. Wait for the processing to complete
        3. Ask questions about your documents
        
        The assistant will provide answers based on the content of all uploaded PDFs and highlight relevant sections.
    """)

st.markdown("---")
st.markdown("Built with Streamlit and Hugging Face")

def cleanup_temp_dir():
    try:
        if st.session_state.temp_dir.exists():
            shutil.rmtree(st.session_state.temp_dir)
    except Exception:
        pass

atexit.register(cleanup_temp_dir)



























# import streamlit as st
# import time
# import PyPDF2
# from pathlib import Path
# import tempfile
# import os
# import uuid
# import shutil
# import re
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# from transformers import pipeline
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import letter
# from io import BytesIO
# from typing import List
# import atexit

# # Page configuration
# st.set_page_config(
#     page_title="PDF Chat Assistant",
#     layout="wide"
# )

# class PDFProcessor:
#     def __init__(self, chunk_size: int = 1000):
#         self.chunk_size = chunk_size
    
#     def extract_text(self, pdf_file) -> str:
#         """Extract text from uploaded PDF file"""
#         pdf_reader = PyPDF2.PdfReader(pdf_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#         return text
    
#     def split_text(self, text: str) -> List[str]:
#         """Split text into chunks"""
#         words = text.split()
#         chunks = []
#         current_chunk = []
#         current_size = 0
        
#         for word in words:
#             current_size += len(word) + 1  # +1 for space
#             if current_size > self.chunk_size:
#                 chunks.append(" ".join(current_chunk))
#                 current_chunk = [word]
#                 current_size = len(word)
#             else:
#                 current_chunk.append(word)
        
#         if current_chunk:
#             chunks.append(" ".join(current_chunk))
        
#         return chunks

# class EmbeddingsHandler:
#     def __init__(self):
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.dimension = 384
#         self.index = faiss.IndexFlatL2(self.dimension)
#         self.embeddings = []
#         self.ids = []
#         self.chunks = []
    
#     def create_embeddings(self, chunks: list, pdf_name: str):
#         """Create embeddings for text chunks and store them in Faiss index"""
#         chunk_embeddings = self.model.encode(chunks)
#         self.index.add(np.array(chunk_embeddings).astype(np.float32))
#         self.embeddings.extend(chunk_embeddings)
#         self.ids.extend([f"chunk_{i}_{pdf_name}" for i in range(len(chunks))])
#         self.chunks.extend(chunks)
    
#     def search_similar_chunks(self, query: str, n_results: int = 3) -> list:
#         """Search for chunks similar to the query using Faiss"""
#         query_embedding = self.model.encode([query]).astype(np.float32)
#         distances, indices = self.index.search(query_embedding, n_results)
#         results = [self.chunks[idx] for idx in indices[0]]
#         return results

# class LLMHandler:
#     def __init__(self):
#         self.qa_pipeline = pipeline(
#             "question-answering",
#             model="deepset/roberta-base-squad2"
#         )
    
#     def get_answer(self, question: str, context: str) -> str:
#         """Get answer from the LLM"""
#         result = self.qa_pipeline(
#             question=question,
#             context=context,
#         )
#         return result['answer']

# def clean_text(text):
#     """Clean text for better matching"""
#     text = ' '.join(text.split())
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     return text.lower()

# def find_text_segments(full_text, search_text, max_segment_length=100):
#     """Find smaller segments of text that are part of the larger chunk"""
#     clean_search = clean_text(search_text)
#     words = clean_search.split()
#     segments = []
    
#     for i in range(len(words)):
#         for j in range(3, 6):
#             if i + j <= len(words):
#                 segment = ' '.join(words[i:i+j])
#                 if len(segment) <= max_segment_length:
#                     segments.append(segment)
    
#     return segments

# def highlight_pdf(pdf_path, text_to_highlight, source_index):
#     """Highlight text in the original PDF with incremental save using PyPDF2 and reportlab"""
#     try:
#         # Read the original PDF
#         with open(pdf_path, "rb") as f:
#             reader = PyPDF2.PdfReader(f)
#             writer = PyPDF2.PdfWriter()

#             # Create a temporary file to save the highlighted PDF
#             packet = BytesIO()
#             can = canvas.Canvas(packet, pagesize=letter)

#             # Iterate over each page and search for the text
#             for page_num, page in enumerate(reader.pages):
#                 text = page.extract_text()
#                 text_instances = find_text_instances(text, text_to_highlight)

#                 for inst in text_instances:
#                     # Simulate highlighting by adding a rectangle overlay
#                     x0, y0, x1, y1 = inst
#                     can.setFillColorRGB(1, 0.85, 0)  # yellow color
#                     can.rect(x0, y0, x1 - x0, y1 - y0, fill=1)

#                 # Draw the original page and overlay highlights
#                 packet.seek(0)
#                 writer.add_page(page)

#             # Save the final PDF with highlights
#             with open(pdf_path, "wb") as f:
#                 writer.write(f)

#         return True
#     except Exception as e:
#         st.error(f"Error highlighting PDF: {str(e)}")
#         return False

# def find_text_instances(text, search_text):
#     """Find positions of the search text in the extracted PDF text"""
#     instances = []
#     start = 0
#     while True:
#         start = text.find(search_text, start)
#         if start == -1:
#             break
#         end = start + len(search_text)
#         # Here, we would need to calculate the position of the text within the page
#         instances.append((start, end))  # Simplified for illustration; in reality, this requires more sophisticated parsing
#         start = end
#     return instances

# if 'temp_dir' not in st.session_state:
#     st.session_state.temp_dir = Path(tempfile.gettempdir()) / f'pdf_chat_{uuid.uuid4()}'
#     st.session_state.temp_dir.mkdir(exist_ok=True)

# if 'processed_pdfs' not in st.session_state:
#     st.session_state.processed_pdfs = {}
#     st.session_state.processor = PDFProcessor()
#     st.session_state.embeddings = EmbeddingsHandler()
#     st.session_state.llm = LLMHandler()

# st.markdown(""" 
#     <style>
#     .css-1p05t8e {border-radius: 10px}
#     .stTextInput>div>div>input {
#         background-color: #f0f2f6;
#         color: #000000;
#         caret-color: #000000;
#     }
#     .stTextInput>div>div>input::placeholder {
#         color: #888888;
#     }
#     .uploadedFile {
#         background-color: #f0f2f6;
#         padding: 20px;
#         border-radius: 10px;
#     }
#     .pdf-preview {
#         border: 1px solid #ddd;
#         border-radius: 5px;
#         padding: 10px;
#         margin-top: 10px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# with st.sidebar:
#     st.header("Document Upload")
#     uploaded_files = st.file_uploader("Upload your PDFs", type=['pdf'], accept_multiple_files=True)
    
#     for uploaded_file in uploaded_files:
#         if uploaded_file.name not in st.session_state.processed_pdfs:
#             with st.spinner(f"Processing {uploaded_file.name}..."):
#                 pdf_id = uuid.uuid4()
#                 pdf_path = st.session_state.temp_dir / f"original_{pdf_id}_{uploaded_file.name}"
                
#                 with open(pdf_path, 'wb') as f:
#                     f.write(uploaded_file.getvalue())
                
#                 text = st.session_state.processor.extract_text(uploaded_file)
#                 chunks = st.session_state.processor.split_text(text)
#                 st.session_state.embeddings.create_embeddings(chunks, uploaded_file.name)
                
#                 st.session_state.processed_pdfs[uploaded_file.name] = {
#                     'path': str(pdf_path),
#                     'chunks': chunks,
#                     'id': str(pdf_id)
#                 }
                
#                 st.success(f"{uploaded_file.name} processed successfully!")
    
#     if st.session_state.processed_pdfs:
#         st.header("Processed Documents")
#         for pdf_name, pdf_info in st.session_state.processed_pdfs.items():
#             st.write(f"✓ {pdf_name}")
#             with open(pdf_info['path'], "rb") as f:
#                 st.download_button(
#                     label=f"Download {pdf_name}",
#                     data=f.read(),
#                     file_name=pdf_name,
#                     mime="application/pdf",
#                     key=f"original_{pdf_info['id']}"
#                 )

# if st.session_state.processed_pdfs:
#     st.header("Ask questions about your PDFs")
#     question = st.text_input("Enter your question:", placeholder="What would you like to know about the documents?")
    
#     if question:
#         with st.spinner("Thinking..."):
#             similar_chunks = st.session_state.embeddings.search_similar_chunks(question)
#             answer = st.session_state.llm.get_answer(question, " ".join(similar_chunks))
            
#             st.markdown("### Answer")
#             st.write(answer)
            
#             with st.expander("View source context"):
#                 for i, chunk in enumerate(similar_chunks, 1):
#                     st.markdown(f"**Source {i}:**")
#                     st.markdown(f"*{chunk}*")
                    
#                     for pdf_name, pdf_info in st.session_state.processed_pdfs.items():
#                         if chunk in pdf_info['chunks']:
#                             button_key = f"download_{pdf_name}_{i}"
                            

#                             found_highlight = highlight_pdf(pdf_info['path'], chunk, i-1)
                            
#                             st.markdown(f"**From document: {pdf_name}**")
                            
#                             with open(pdf_info['path'], "rb") as f:
#                                 button_label = "Download PDF with highlights" if found_highlight else "Download PDF"
#                                 st.download_button(
#                                     label=button_label,
#                                     data=f.read(),
#                                     file_name=pdf_name,
#                                     mime="application/pdf",
#                                     key=button_key
#                                 )
#                     st.markdown("---")
# else:
#     st.markdown(""" 
#         Welcome to PDF Chat Assistant!
        
#         To get started:
#         1. Upload one or more PDF documents using the sidebar
#         2. Wait for the processing to complete
#         3. Ask questions about your documents
        
#         The assistant will provide answers based on the content of all uploaded PDFs and highlight relevant sections.
#     """)

# st.markdown("---")
# st.markdown("Built with Streamlit and Hugging Face")

# def cleanup_temp_dir():
#     try:
#         if st.session_state.temp_dir.exists():
#             shutil.rmtree(st.session_state.temp_dir)
#     except Exception:
#         pass

# atexit.register(cleanup_temp_dir)








# import streamlit as st
# import time
# import PyPDF2
# from pathlib import Path
# import tempfile
# import os
# import uuid
# import shutil
# import re
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# from transformers import pipeline
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import letter
# from io import BytesIO
# from typing import List
# import atexit
# import fitz  # PyMuPDF for better text extraction and highlighting

# # Page configuration
# st.set_page_config(
#     page_title="PDF Chat Assistant",
#     layout="wide"
# )

# class PDFProcessor:
#     def __init__(self, chunk_size: int = 1000):
#         self.chunk_size = chunk_size
    
#     def extract_text(self, pdf_file) -> str:
#         """Extract text from uploaded PDF file using PyMuPDF"""
#         pdf_bytes = pdf_file.read()
#         pdf_file.seek(0)  # Reset file pointer
        
#         doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#         text = ""
#         for page in doc:
#             text += page.get_text()
#         doc.close()
#         return text
    
#     def split_text(self, text: str) -> List[str]:
#         """Split text into chunks"""
#         words = text.split()
#         chunks = []
#         current_chunk = []
#         current_size = 0
        
#         for word in words:
#             current_size += len(word) + 1  # +1 for space
#             if current_size > self.chunk_size:
#                 chunks.append(" ".join(current_chunk))
#                 current_chunk = [word]
#                 current_size = len(word)
#             else:
#                 current_chunk.append(word)
        
#         if current_chunk:
#             chunks.append(" ".join(current_chunk))
        
#         return chunks

# class EmbeddingsHandler:
#     def __init__(self):
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.dimension = 384
#         self.index = faiss.IndexFlatL2(self.dimension)
#         self.embeddings = []
#         self.ids = []
#         self.chunks = []
    
#     def create_embeddings(self, chunks: list, pdf_name: str):
#         """Create embeddings for text chunks and store them in Faiss index"""
#         chunk_embeddings = self.model.encode(chunks)
#         self.index.add(np.array(chunk_embeddings).astype(np.float32))
#         self.embeddings.extend(chunk_embeddings)
#         self.ids.extend([f"chunk_{i}_{pdf_name}" for i in range(len(chunks))])
#         self.chunks.extend(chunks)
    
#     def search_similar_chunks(self, query: str, n_results: int = 3) -> list:
#         """Search for chunks similar to the query using Faiss"""
#         query_embedding = self.model.encode([query]).astype(np.float32)
#         distances, indices = self.index.search(query_embedding, n_results)
#         results = [self.chunks[idx] for idx in indices[0]]
#         return results

# class LLMHandler:
#     def __init__(self):
#         self.qa_pipeline = pipeline(
#             "question-answering",
#             model="deepset/roberta-base-squad2"
#         )
    
#     def get_answer(self, question: str, context: str) -> str:
#         """Get answer from the LLM"""
#         result = self.qa_pipeline(
#             question=question,
#             context=context,
#         )
#         return result['answer']

# def clean_text(text):
#     """Clean text for better matching"""
#     text = ' '.join(text.split())
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     return text.lower()

# def highlight_pdf(pdf_path: str, text_to_highlight: str) -> str:
#     """Highlight text in PDF and return path to highlighted version"""
#     try:
#         # Create output path for highlighted PDF
#         output_path = str(pdf_path).replace('original_', 'highlighted_')
        
#         # Open the PDF with PyMuPDF
#         doc = fitz.open(pdf_path)
        
#         # Prepare text for searching (remove extra whitespace)
#         search_text = ' '.join(text_to_highlight.split())
        
#         # Search and highlight on each page
#         for page in doc:
#             # Search for text instances
#             instances = page.search_for(search_text)
            
#             # Add highlight for each instance
#             for inst in instances:
#                 highlight = page.add_highlight_annot(inst)
#                 highlight.set_colors(stroke=(1, 1, 0))  # Yellow highlight
                
#         # Save the highlighted PDF
#         doc.save(output_path)
#         doc.close()
        
#         return output_path
#     except Exception as e:
#         st.error(f"Error highlighting PDF: {str(e)}")
#         return pdf_path  # Return original path if highlighting fails

# # Initialize session state
# if 'temp_dir' not in st.session_state:
#     st.session_state.temp_dir = Path(tempfile.gettempdir()) / f'pdf_chat_{uuid.uuid4()}'
#     st.session_state.temp_dir.mkdir(exist_ok=True)

# if 'processed_pdfs' not in st.session_state:
#     st.session_state.processed_pdfs = {}
#     st.session_state.processor = PDFProcessor()
#     st.session_state.embeddings = EmbeddingsHandler()
#     st.session_state.llm = LLMHandler()

# # Custom CSS
# st.markdown(""" 
#     <style>
#     .css-1p05t8e {border-radius: 10px}
#     .stTextInput>div>div>input {
#         background-color: #f0f2f6;
#         color: #000000;
#         caret-color: #000000;
#     }
#     .stTextInput>div>div>input::placeholder {
#         color: #888888;
#     }
#     .uploadedFile {
#         background-color: #f0f2f6;
#         padding: 20px;
#         border-radius: 10px;
#     }
#     .pdf-preview {
#         border: 1px solid #ddd;
#         border-radius: 5px;
#         padding: 10px;
#         margin-top: 10px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Sidebar for document upload
# with st.sidebar:
#     st.header("Document Upload")
#     uploaded_files = st.file_uploader("Upload your PDFs", type=['pdf'], accept_multiple_files=True)
    
#     for uploaded_file in uploaded_files:
#         if uploaded_file.name not in st.session_state.processed_pdfs:
#             with st.spinner(f"Processing {uploaded_file.name}..."):
#                 pdf_id = uuid.uuid4()
#                 pdf_path = st.session_state.temp_dir / f"original_{pdf_id}_{uploaded_file.name}"
                
#                 # Save uploaded file
#                 with open(pdf_path, 'wb') as f:
#                     f.write(uploaded_file.getvalue())
                
#                 # Process the PDF
#                 text = st.session_state.processor.extract_text(uploaded_file)
#                 chunks = st.session_state.processor.split_text(text)
#                 st.session_state.embeddings.create_embeddings(chunks, uploaded_file.name)
                
#                 st.session_state.processed_pdfs[uploaded_file.name] = {
#                     'path': str(pdf_path),
#                     'chunks': chunks,
#                     'id': str(pdf_id)
#                 }
                
#                 st.success(f"{uploaded_file.name} processed successfully!")
    
#     if st.session_state.processed_pdfs:
#         st.header("Processed Documents")
#         for pdf_name, pdf_info in st.session_state.processed_pdfs.items():
#             st.write(f"✓ {pdf_name}")
#             with open(pdf_info['path'], "rb") as f:
#                 st.download_button(
#                     label=f"Download {pdf_name}",
#                     data=f.read(),
#                     file_name=pdf_name,
#                     mime="application/pdf",
#                     key=f"original_{pdf_info['id']}"
#                 )

# # Main content area
# if st.session_state.processed_pdfs:
#     st.header("Ask questions about your PDFs")
#     question = st.text_input("Enter your question:", placeholder="What would you like to know about the documents?")
    
#     if question:
#         with st.spinner("Thinking..."):
#             similar_chunks = st.session_state.embeddings.search_similar_chunks(question)
#             answer = st.session_state.llm.get_answer(question, " ".join(similar_chunks))
            
#             st.markdown("### Answer")
#             st.write(answer)
            
#             with st.expander("View source context"):
#                 for i, chunk in enumerate(similar_chunks, 1):
#                     st.markdown(f"**Source {i}:**")
#                     st.markdown(f"*{chunk}*")
                    
#                     for pdf_name, pdf_info in st.session_state.processed_pdfs.items():
#                         if chunk in pdf_info['chunks']:
#                             # Create highlighted version of the PDF
#                             highlighted_pdf_path = highlight_pdf(pdf_info['path'], chunk)
                            
#                             st.markdown(f"**From document: {pdf_name}**")
                            
#                             with open(highlighted_pdf_path, "rb") as f:
#                                 st.download_button(
#                                     label="Download PDF with highlights",
#                                     data=f.read(),
#                                     file_name=f"highlighted_{pdf_name}",
#                                     mime="application/pdf",
#                                     key=f"highlighted_{pdf_info['id']}_{i}"
#                                 )
#                     st.markdown("---")
# else:
#     st.markdown(""" 
#         Welcome to PDF Chat Assistant!
        
#         To get started:
#         1. Upload one or more PDF documents using the sidebar
#         2. Wait for the processing to complete
#         3. Ask questions about your documents
        
#         The assistant will provide answers based on the content of all uploaded PDFs and highlight relevant sections.
#     """)

# st.markdown("---")
# st.markdown("Built with Streamlit and Hugging Face")

# # Cleanup function
# def cleanup_temp_dir():
#     try:
#         if st.session_state.temp_dir.exists():
#             shutil.rmtree(st.session_state.temp_dir)
#     except Exception:
#         pass

# atexit.register(cleanup_temp_dir)