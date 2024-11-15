import openai
import streamlit as st
from sentence_transformers import SentenceTransformer
from collections import deque
import numpy as np
import faiss

# Initialize OpenAI API key
openai.api_key = "your-openai-api-key"  # Replace with your OpenAI API key
# Initialize the SentenceTransformer for encoding questions
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and metadata
index = faiss.read_index("faiss_index.index")  # Path to your FAISS index
faiss_metadata = np.load("faiss_metadata.npy", allow_pickle=True)  # Path to your FAISS metadata

# Streamlit UI Setup
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬", layout="centered")

# Display title, image, and contact info
st.title("Mental Health Support Chatbot")
st.image("https://via.placeholder.com/150", width=150)  # Replace with your logo image URL
st.write("Contact: support@mentalhealth.com")
st.markdown("""
    **Please note that this app is not a substitute for professional medical advice,**
    and it has not been reviewed by doctors. You should not rely on it for medical treatments,
    prescriptions, or medications. The chatbot provides information based on documents, along
    with general advice and exercises. If unsure, it may ask you for more details.
""")

# Memory to store conversation (using buffer memory approach with deque)
if "conversation" not in st.session_state:
    st.session_state.conversation = deque(maxlen=15)  # Keep a memory of the last 10 messages

# Create a function to generate OpenAI response with prompt engineering and FAISS
def generate_openai_response(user_question, conversation_history):
    # Encode the user's question into an embedding
    user_question_embedding = embedding_model.encode(user_question, convert_to_tensor=True)
    
    # Perform a search in the FAISS index for the most relevant document
    D, I = index.search(np.array([user_question_embedding]), k=2)  # Search for the most similar document
    
    # Get the most similar document's metadata (assuming metadata contains the document text)
    # Get the most similar documents' metadata
    most_similar_documents = [faiss_metadata[i] for i in I[0]]
    
    # Format the prompt with conversation context and the retrieved document
    conversation_context = "\n".join(conversation_history)
    prompt = f"""
    Please note that this app is not a substitute for professional medical advice, and it has not been reviewed by doctors.
    You should not rely on it for medical treatments, prescriptions, or medications. The chatbot provides information based on documents,
    along with general advice and exercises. If unsure, it may ask you for more details. The responses are intended to be compassionate,
    informative, and empathetic.

    The following is a conversation with a mental health chatbot. The chatbot provides information and guidance based on relevant documents:
    
    Relevant Documents:
    1. "{most_similar_documents[0]}"
    2. "{most_similar_documents[1]}"

    {conversation_context}
    
    User: {user_question}
    Chatbot:
    """
    
    # Query OpenAI with the augmented prompt
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose your OpenAI engine
        prompt=prompt,
        max_tokens=150,
        temperature=0.8
    )
    
    return response.choices[0].text.strip()

# Create a container for the conversation
conversation_container = st.container()

# Display the conversation in the chat container
with conversation_container:
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**Chatbot:** {msg['text']}")

# Input fields for user question
user_question = st.text_input("Your Question", key="question_input")
submit_button = st.button("Submit")

# When the user submits the question
if submit_button and user_question:
    # Add user message to conversation history
    st.session_state.conversation.append({"role": "user", "text": user_question})

    # Get the response from OpenAI using prompt engineering and FAISS
    chatbot_response = generate_openai_response(user_question, [msg["text"] for msg in st.session_state.conversation])

    # Add chatbot response to the conversation history
    st.session_state.conversation.append({"role": "chatbot", "text": chatbot_response})

    # Clear the input box
    st.session_state.question_input = ""

# Finish button to clear the memory
if st.button("Finish"):
    st.session_state.conversation.clear()
    st.write("Conversation memory cleared.")