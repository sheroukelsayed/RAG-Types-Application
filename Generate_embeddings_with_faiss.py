import faiss
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import os

# Initialize the Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDFs using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Function to create embeddings for a document
def create_embeddings(texts):
    embeddings = embedding_model.encode(texts)
    return embeddings

# Initialize FAISS index and metadata storage
index = None
metadata = []

# Function to add embeddings to FAISS index
def add_embeddings_to_faiss(embeddings, metadata):
    global index
    # Check if FAISS index exists, if not, initialize it
    if index is None:
        dimension = len(embeddings[0])  # Dimensionality of the embedding vector
        index = faiss.IndexFlatL2(dimension)  # L2 distance-based index (you can change this to other types)
    
    # Convert embeddings to numpy array
    embedding_array = np.array(embeddings).astype('float32')
    
    # Add embeddings to FAISS index
    index.add(embedding_array)
    
    # Add metadata
    metadata.extend(metadata)

# Function to save the FAISS index and metadata to disk
def save_faiss_index_and_metadata(index, metadata, index_path, metadata_path):
    faiss.write_index(index, index_path)
    np.save(metadata_path, np.array(metadata))

# Function to load the FAISS index and metadata from disk
def load_faiss_index_and_metadata(index_path, metadata_path):
    global index, metadata
    index = faiss.read_index(index_path)
    metadata = np.load(metadata_path, allow_pickle=True).tolist()

# Absolute paths to the PDFs
base_dir = ".\Medical Resourcers"
pdf_paths = [
    os.path.join(base_dir, "Cognitive-Psychology-Sternberge.pdf"),
    os.path.join(base_dir, "Diagnostic_and_statistical_manual_of_mental_disorders _ DSM-5.pdf")
]
documents = []
for pdf_path in pdf_paths:
    document_text = extract_text_from_pdf(pdf_path)
    print(document_text)
    documents.append(document_text)

# Create embeddings for the extracted text
embeddings = create_embeddings(documents)

# Add documents and embeddings to FAISS
metadata = [{"source": f"document_{i}"} for i in range(len(documents))]
add_embeddings_to_faiss(embeddings, metadata)

# Save the FAISS index and metadata to disk
save_faiss_index_and_metadata(index, metadata, "faiss_index.index", "faiss_metadata.npy")

print("Documents and embeddings have been saved to FAISS vector database.")
