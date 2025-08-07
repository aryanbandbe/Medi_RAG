import streamlit as st
import os
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
import json
import time

# Load environment variables from the .env file
load_dotenv()

# --- Functions from your existing main.py file ---

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

@st.cache_resource
def setup_pinecone_and_model():
    """
    Initializes the Pinecone client and a Sentence Transformer model.
    """
    # API key is now retrieved from Streamlit secrets
    api_key = st.secrets.get("PINECONE_API_KEY")
    if not api_key:
        st.error("Pinecone API Key not found in Streamlit secrets. Please add it.")
        return None, None, None

    pc = Pinecone(api_key=api_key)
    index_name = "medical-rag"
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    return pc, index_name, model

# Step 4: Perform RAG Query using the Gemini API
def rag_query_with_gemini(pc, index_name, model, query_text):
    """
    Performs a vector search on Pinecone and uses the retrieved context
    to generate an answer with the Gemini API.
    """
    # Gemini API key is also retrieved from Streamlit secrets
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    if not gemini_api_key:
        return "Gemini API Key not found in Streamlit secrets. Please add it."

    index = pc.Index(index_name)
    query_embedding = model.encode(query_text).tolist()
    
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    context = ""
    for match in results['matches']:
        metadata = match['metadata']
        context += f"Disease: {metadata['disease']}. Symptoms: {metadata['symptoms']}. Precautions: {metadata['precautions']}\n"
    
    # Construct a detailed prompt with the retrieved context
    prompt = f"Using the following medical context, answer the user's question. Specifically, provide the name of the possible disease and a medication or a precaution for it. If the context doesn't contain a direct answer, state that you cannot provide medical advice and suggest consulting a doctor.\n\nContext: {context}\n\nQuestion: {query_text}\n\nAnswer:"
    
    # API call to Gemini
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={gemini_api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    
    response = None
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            if result.get("candidates"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "Could not generate a response. The API might have returned an empty result."
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with Gemini API: {e}")
            retries += 1
            if retries < max_retries:
                time.sleep(2 ** retries)  # Exponential backoff
        except (KeyError, IndexError) as e:
            st.error(f"Unexpected API response format: {e}")
            return "Unexpected API response format."
    return "Failed to get a response from the Gemini API after multiple retries."


# --- Streamlit UI Code ---
def main():
    st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🩺")
    st.title("Medical RAG Chatbot 🩺")
    st.markdown("Ask me about diseases, symptoms, or precautions, and I will use a Retrieval-Augmented Generation (RAG) model to answer.")

    # Initialize Pinecone and model
    pc, index_name, model = setup_pinecone_and_model()

    if pc and index_name and model:
        # User input
        user_query = st.text_input("Enter your symptoms or a medical question:", key="query_input")
        
        # Process query on button click
        if st.button("Get Answer"):
            if user_query:
                with st.spinner("Searching for a diagnosis..."):
                    response = rag_query_with_gemini(pc, index_name, model, user_query)
                    st.success("Answer generated!")
                    st.write(response)
            else:
                st.warning("Please enter a query.")

if __name__ == "__main__":
    main()

