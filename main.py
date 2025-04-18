from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field # Used for data validation and defining request/response models
from sentence_transformers import SentenceTransformer, util # The core library for semantic search
import numpy as np
import torch # Required by sentence-transformers (or tensorflow)
from typing import List, Dict # For type hinting

# --- Configuration ---
# Choosing a pre-trained model. 'all-MiniLM-L6-v2' is chosen here as it's a good starting point:
# fast, good quality, and relatively small.
# Other options: 'msmarco-distilbert-base-v4', 'paraphrase-MiniLM-L6-v2'
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Data Store (Replace with your actual data source) ---
# For this example, we'll use a simple list of strings.
# In a real application, this could come from a database, file, etc.
documents_store = [
    "The weather today is sunny and warm.",
    "Artificial intelligence is transforming many industries.",
    "Python is a versatile programming language.",
    "How does climate change affect polar bears?",
    "The new electric car has an impressive range.",
    "Learning about machine learning can be challenging but rewarding.",
    "What are the best practices for API design?",
    "The capital of France is Paris.",
    "Data science involves statistics and computer science.",
    "Semantic search helps find information based on meaning, not just keywords."
]

# --- Global Variables (Initialized on Startup) ---
# We load the model and compute embeddings once when the app starts
# to avoid reloading them on every request.
model = None
document_embeddings = None

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Simple Semantic Search API",
    description="An API that finds documents semantically similar to a query.",
    version="0.1.0",
)

# --- Pydantic Models (for Request and Response) ---
class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, description="The search text")
    top_k: int = Field(3, ge=1, le=100, description="Number of top results to return")

class SearchResult(BaseModel):
    text: str
    score: float = Field(..., ge=0, le=1, description="Similarity score (0 to 1)")

class SearchResponse(BaseModel):
    results: List[SearchResult]

# --- Application Startup Event ---
# This function runs once when FastAPI starts up.
# Ideal place to load models and pre-compute embeddings.
@app.on_event("startup")
async def startup_event():
    global model, document_embeddings
    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    # Load the Sentence Transformer model
    # device='cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
    model = SentenceTransformer(MODEL_NAME, device='cpu') # Start with CPU for simplicity
    print("Model loaded.")

    print("Encoding documents...")
    # Encode all documents in the store into embeddings (vectors)
    # This might take a moment depending on the number of documents and model size.
    document_embeddings = model.encode(documents_store, convert_to_tensor=True)
    print(f"Encoded {len(documents_store)} documents.")
    print("Application startup complete.")

# --- API Endpoint ---
@app.post("/search", response_model=SearchResponse)
async def perform_semantic_search(search_request: SearchQuery):
    """
    Performs semantic search based on the input query.

    - Encodes the query text into an embedding.
    - Computes cosine similarity between the query embedding and all document embeddings.
    - Returns the top_k documents with the highest similarity scores.
    """
    if model is None or document_embeddings is None:
        # This should ideally not happen if startup event completes successfully
        raise HTTPException(status_code=503, detail="Model or document embeddings not ready.")

    print(f"Received search query: '{search_request.query}', top_k={search_request.top_k}")

    # 1. Encode the search query
    query_embedding = model.encode(search_request.query, convert_to_tensor=True)

    # 2. Compute Cosine Similarity
    # 'util.cos_sim' computes similarity between the query and all document embeddings.
    # It returns a tensor of scores.
    cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0] # We take [0] as we have one query

    # 3. Find the top_k most similar documents
    # We use torch.topk to find the indices and scores of the highest scores.
    # 'top_k' is clamped to the number of documents available.
    actual_top_k = min(search_request.top_k, len(documents_store))
    top_results = torch.topk(cosine_scores, k=actual_top_k)

    # 4. Format the results
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append(
            SearchResult(
                text=documents_store[idx],
                score=round(score.item(), 4) # .item() gets the Python number from tensor
            )
        )
        print(f"  - Found: Score={score.item():.4f}, Index={idx}, Text='{documents_store[idx]}'")

    return SearchResponse(results=results)

# --- Health Check Endpoint (Good Practice) ---
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    # Check if model is loaded as a simple check
    model_loaded = model is not None
    return {"status": "ok", "model_loaded": model_loaded}

# --- Running the App (for local development) ---
# If you run this script directly (python main.py), it won't start the server.
# You need to use uvicorn: `uvicorn main:app --reload`
# The following block is usually for debugging or simple script execution,
# but `uvicorn` is the standard way to run FastAPI apps.
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting Uvicorn server...")
#     # Note: --reload is great for development but should be off in production.
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)