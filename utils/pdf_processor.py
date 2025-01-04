import PyPDF2
from typing import List

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
