import os
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests

# Load environment variables from the .env file
load_dotenv()

# Step 1: Load and Preprocess Data
def load_data(symptoms_file, precautions_file):
    """
    Loads data from two CSV files, merges them, and structures it
    into a list of dictionaries for easier processing.
    """
    print("Loading and merging datasets...")
    
    # Load the disease and symptoms data
    symptoms_df = pd.read_csv(symptoms_file)
    symptoms_df.fillna("none", inplace=True)
    
    # Load the disease and precautions data
    precautions_df = pd.read_csv(precautions_file)
    precautions_df.fillna("none", inplace=True)

    # Merge the two dataframes on the 'Disease' column
    # We use an 'inner' merge to ensure we only get diseases present in both files.
    merged_df = pd.merge(symptoms_df, precautions_df, on='Disease', how='inner')
    
    documents = []
    for index, row in merged_df.iterrows():
        # Get all symptom columns and filter out 'none' values
        symptom_cols = [col for col in merged_df.columns if 'Symptom' in str(col)]
        symptoms = [str(row[s]) for s in symptom_cols if str(row[s]) != "none"]
        
        # Get all precaution columns and filter out 'none' values
        precaution_cols = [col for col in merged_df.columns if 'Precaution' in str(col)]
        precautions = [str(row[p]) for p in precaution_cols if str(row[p]) != "none"]
        
        disease_info = {
            "disease": row['Disease'],
            "symptoms": ", ".join(symptoms),
            "precautions": ", ".join(precautions)
        }
        documents.append(disease_info)
    
    print(f"Loaded and merged {len(documents)} medical records.")
    return documents

# Step 2: Initialize Pinecone and Embedding Model
def setup_pinecone_and_model():
    """
    Initializes the Pinecone client and a Sentence Transformer model.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    index_name = "medical-rag"
    
    # Use a Sentence-Transformer model for creating 384-dimensional embeddings.
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    
    print(f"Pinecone client initialized with index '{index_name}'.")
    print(f"Embedding model '{model_name}' loaded.")
    
    return pc, index_name, model

# Step 3: Embed Data and Upsert to Pinecone
def upsert_data_to_pinecone(pc, index_name, model, documents):
    """
    Embeds the preprocessed data and stores it in the Pinecone index.
    This function should only be run once to populate the database.
    """
    index = pc.Index(index_name)
    
    print("Starting data upsert to Pinecone...")
    for i, doc in enumerate(documents):
        # Create a single text string for embedding that combines all relevant info
        text_to_embed = f"Disease: {doc['disease']}. Symptoms: {doc['symptoms']}. Precautions: {doc['precautions']}"
        embedding = model.encode(text_to_embed).tolist()
        
        # Prepare the vector for upsert with a unique ID and metadata
        vector = {
            'id': str(i),
            'values': embedding,
            'metadata': doc
        }
        
        index.upsert(vectors=[vector])
        if (i + 1) % 50 == 0:
            print(f"Upserted {i + 1} documents.")
    
    print("All data successfully upserted to Pinecone.")

# Step 4: Perform RAG Query using a local Ollama model
def rag_query_with_ollama(pc, index_name, model, query_text):
    """
    Performs a vector search on Pinecone and uses the retrieved context
    to generate an anwser with a local Ollama model (TinyLlama).
    """
    index = pc.Index(index_name)
    query_embedding = model.encode(query_text).tolist()
    
    # Search Pinecone for the most relevant context chunks
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    # Extract the context from the search results
    context = ""
    for match in results['matches']:
        metadata = match['metadata']
        context += f"Disease: {metadata['disease']}. Symptoms: {metadata['symptoms']}. Precautions: {metadata['precautions']}\n"
    
    # Construct a detailed prompt with the retrieved context
    prompt = f"Using the following medical context, answer the user's question. Specifically, provide the name of the possible disease and a medication or a precaution for it. If the context doesn't contain a direct answer, state that you cannot provide medical advice and suggest consulting a doctor.\n\nContext: {context}\n\nQuestion: {query_text}\n\nAnswer:"
    
    # Send the prompt to the local Ollama server running TinyLlama
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False # Get the full response in one go
        })
        response.raise_for_status() # Check for HTTP errors
        
        generated_answer = response.json()['response']
        
        print("-" * 50)
        print(f"User Query: {query_text}")
        print("-" * 50)
        print("Generated Answer:")
        print(generated_answer)
        print("-" * 50)
        
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        print("Please ensure Ollama is running and you have pulled the 'tinyllama' model.")
        print("Run 'ollama serve' in a terminal and then 'ollama run tinyllama' to download the model.")


if __name__ == "__main__":
    # File paths to your uploaded datasets
    symptoms_file = 'DiseaseAndSymptoms.csv'
    precautions_file = 'Disease precaution.csv'

    if not os.path.exists(symptoms_file) or not os.path.exists(precautions_file):
        print("Please ensure both 'DiseaseAndSymptoms.csv' and 'Disease precaution.csv' are in your project folder.")
    else:
        documents = load_data(symptoms_file, precautions_file)
        pc, index_name, model = setup_pinecone_and_model()

        # # The upsert function is commented out. UNCOMMENT THIS BLOCK AND RUN ONCE TO POPULATE THE DB,
        # # THEN COMMENT IT BACK OUT BEFORE RUNNING QUERIES.
        # # upsert_data_to_pinecone(pc, index_name, model, documents)

        # Start an interactive loop to accept user queries
        print("\nMedical RAG Chatbot is ready. Type 'exit' to quit.")
        while True:
            user_query = input("Enter your symptoms or a medical question: ")
            if user_query.lower() == 'exit':
                break
            
            rag_query_with_ollama(pc, index_name, model, user_query)
