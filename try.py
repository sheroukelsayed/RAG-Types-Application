import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle
import fitz  # PyMuPDF

# Initialize the Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDFs using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Function to split document text into smaller parts
def split_document(text, chunk_size=7500):
    # Split document text into chunks of size chunk_size (or smaller)
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Function to create embeddings for a document part
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
document_parts = []
for pdf_path in pdf_paths:
    document_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted text from {pdf_path}")
    
    # Split the document into smaller parts
    parts = split_document(document_text)
    document_parts.extend(parts)  # Add the parts to the list of all document parts
    documents.append(document_text)  # Add full document for reference

# Create embeddings for the extracted document parts
embeddings = create_embeddings(document_parts)

# Add document parts and embeddings to FAISS
metadata = [{"source": f"document_{i}", "part_id": i} for i in range(len(document_parts))]
add_embeddings_to_faiss(embeddings, metadata)

# ------------------------------
# Interaction with the chatbot
# ------------------------------
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to retrieve most similar document part based on the user query
def get_similar_document_part(user_query):
    # Encode the user's question into an embedding
    user_query_embedding = embedding_model.encode(user_query)

    # Perform a search in the FAISS index for the most relevant document part
    D, I = index.search(np.array([user_query_embedding]), k=1)  # Search for the most similar part
    #print("D",D)
    #print("I",I)
    # Get the most similar document part's metadata, which includes the full text of the part
    most_similar_part = metadata[I[0][0]]
    #print("most_similar_part", most_similar_part)
    document_id = most_similar_part['source']
    #print(" document_id",  document_id)
    part_id = most_similar_part['part_id']
    #print("part_id", part_id)
    
    # Retrieve the corresponding document part based on part_id
    document_text = document_parts[part_id]
    #print("document_text", document_text)
    
    # Return the document part's text
    return document_text, document_id, part_id

# Function to generate OpenAI response based on the query and retrieved document part
def generate_openai_response(user_query):
    # Get the most similar document part based on the user's query
    document_text, document_id, part_id = get_similar_document_part(user_query)
    
    # Format the prompt with conversation context and the retrieved document
    conversation_context = "\n".join(conversation_history)
    prompt = f"""
    Please note that this app is  for  medical advice. The chatbot provides information based on provied documents,
    along with  advice and exercises. If unsure, it may ask you for more details. The responses are intended to be compassionate,
    informative, and empathetic.

    The following is a conversation with a mental health chatbot. The chatbot provides information and guidance with the help of  relevant books with help of  given medical document:
    
    Relevant Documents:
    "{document_text}"
    

    {conversation_context}
    
    User: {user_question}
    Chatbot:
    """
    
    # Query OpenAI with the augmented prompt
    response = client.completions.create(
        model = "gpt-3.5-turbo-instruct",  # Choose your OpenAI engine
        prompt=prompt,
        max_tokens=250,
        temperature=0.7
    )

    return response.choices[0].text.strip(), document_text

# Terminal-based conversation loop
print("Welcome to the Mental Health Support Chatbot!")
print("**Disclaimer:** This app is not a substitute for professional medical advice.")
print("Type 'exit' to end the chat.")

conversation_history = []

while True:
    # Get user input
    user_question = input("You: ")
    if user_question.lower() == 'exit':
        print("Chatbot: Goodbye! Stay safe and take care.")
        break
    
    # Add user question to conversation history
    conversation_history.append(f"User: {user_question}")

    # Generate chatbot response
    chatbot_response, document_text = generate_openai_response(user_question)
    
    # Add chatbot response to conversation history
    conversation_history.append(f"Chatbot: {chatbot_response}")
    
    # Display chatbot response with the retrieved relevant document part
  
    print(f"Relevant resources: {document_text}")
    print(f"Chatbot: {chatbot_response}")