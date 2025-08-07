import streamlit as st
import os
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests

# Load environment variables from the .env file
load_dotenv()

# --- Functions from your existing main.py file ---

# A function to load and merge data (not needed for the deployed app, but good to have)
@st.cache_data
def load_data(symptoms_file, precautions_file):
    """
    Loads data from two CSV files, merges them, and structures it
    into a list of dictionaries for easier processing.
    """
    symptoms_df = pd.read_csv(symptoms_file)
    symptoms_df.fillna("none", inplace=True)
    precautions_df = pd.read_csv(precautions_file)
    precautions_df.fillna("none", inplace=True)
    merged_df = pd.merge(symptoms_df, precautions_df, on='Disease', how='inner')
    
    documents = []
    for index, row in merged_df.iterrows():
        symptom_cols = [col for col in merged_df.columns if 'Symptom' in str(col)]
        symptoms = [str(row[s]) for s in symptom_cols if str(row[s]) != "none"]
        
        precaution_cols = [col for col in merged_df.columns if 'Precaution' in str(col)]
        precautions = [str(row[p]) for p in precaution_cols if str(row[p]) != "none"]
        
        disease_info = {
            "disease": row['Disease'],
            "symptoms": ", ".join(symptoms),
            "precautions": ", ".join(precautions)
        }
        documents.append(disease_info)
    
    return documents

# A function to initialize Pinecone and the embedding model
@st.cache_resource
def setup_pinecone_and_model():
    """
    Initializes the Pinecone client and a Sentence Transformer model.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index_name = "medical-rag"
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    return pc, index_name, model

# A function to perform the RAG query with Ollama
def rag_query_with_ollama(pc, index_name, model, query_text):
    """
    Performs a vector search on Pinecone and uses the retrieved context
    to generate an anwser with a local Ollama model (TinyLlama).
    """
    index = pc.Index(index_name)
    query_embedding = model.encode(query_text).tolist()
    
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    context = ""
    for match in results['matches']:
        metadata = match['metadata']
        context += f"Disease: {metadata['disease']}. Symptoms: {metadata['symptoms']}. Precautions: {metadata['precautions']}\n"
    
    prompt = f"Using the following medical context, answer the user's question. Specifically, provide the name of the possible disease and a medication or a precaution for it. If the context doesn't contain a direct answer, state that you cannot provide medical advice and suggest consulting a doctor.\n\nContext: {context}\n\nQuestion: {query_text}\n\nAnswer:"
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        return response.json()['response']
        
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama: {e}. Please ensure Ollama is running and you have pulled the 'tinyllama' model."

# --- Streamlit UI Code ---
def main():
    st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🩺")
    st.title("Medical RAG Chatbot 🩺")
    st.markdown("Ask me about diseases, symptoms, or precautions, and I will use a Retrieval-Augmented Generation (RAG) model to answer.")

    # Initialize Pinecone and model
    pc, index_name, model = setup_pinecone_and_model()

    # User input
    user_query = st.text_input("Enter your symptoms or a medical question:", key="query_input")
    
    # Process query on button click
    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Searching for a diagnosis..."):
                response = rag_query_with_ollama(pc, index_name, model, user_query)
                st.success("Answer generated!")
                st.write(response)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()

