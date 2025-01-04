from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingsHandler:
    def __init__(self):
        # Load the SentenceTransformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Faiss index for storing embeddings
        self.dimension = 384  # Dimension of the embeddings (based on the model used)
        self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance (Euclidean distance)
        self.embeddings = []  # To store the embeddings
        self.ids = []  # To store the corresponding chunk IDs
        self.chunks = []  # To store the actual chunk text
    
    def create_embeddings(self, chunks: list, pdf_name: str):
        """Create embeddings for text chunks and store them in Faiss index"""
        # Compute embeddings for the chunks
        chunk_embeddings = self.model.encode(chunks)
        
        # Add the embeddings to the Faiss index
        self.index.add(np.array(chunk_embeddings).astype(np.float32))
        
        # Store the embeddings, chunk IDs, and chunk text
        self.embeddings.extend(chunk_embeddings)
        self.ids.extend([f"chunk_{i}_{pdf_name}" for i in range(len(chunks))])
        self.chunks.extend(chunks)  # Storing the actual chunk text
    
    def search_similar_chunks(self, query: str, n_results: int = 3) -> list:
        """Search for chunks similar to the query using Faiss"""
        # Get the embedding for the query
        query_embedding = self.model.encode([query]).astype(np.float32)
        
        # Perform the search in Faiss
        distances, indices = self.index.search(query_embedding, n_results)
        
        # Retrieve the actual text of the most similar chunks
        results = [self.chunks[idx] for idx in indices[0]]  # Use chunk text, not IDs
        return results
