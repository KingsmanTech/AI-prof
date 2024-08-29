import sys
import json
from pinecone import Pinecone, ServerlessSpec
import requests
import time

# API Keys (Replace with your actual keys)
PINECONE_API_KEY= "Ypur API key"
HUGGINGFACE_API_KEY= "Your API key"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create a Pinecone index
try:
    pc.create_index(
        name="rag",
        dimension=384,  # Changed to match the output of the Hugging Face model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index 'rag' created successfully")
except Exception as e:
    print(f"Note: {e}")
    print("If the index already exists, you can ignore this message and continue.")

# Load the review data
try:
    with open("reviews.json", "r") as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: reviews.json file not found")
    sys.exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON in reviews.json")
    sys.exit(1)

processed_data = []

# Function to get embedding from Hugging Face
def get_embedding(text):
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": text})
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()[0]  # Return the first (and only) embedding
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Error getting embedding after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff

# Create embeddings for each review
for review in data["reviews"]:
    embedding = get_embedding(review['review'])
    if embedding is not None:
        processed_data.append(
            {
                "id": review["professor"],
                "values": embedding,
                "metadata": {
                    "review": review["review"],
                    "subject": review["subject"],
                    "stars": review["stars"],
                }
            }
        )
    else:
        print(f"Skipping review for {review['professor']} due to embedding error")

# Insert the embeddings into the Pinecone index
index = pc.Index("rag")
try:
    if processed_data:
        upsert_response = index.upsert(
            vectors=processed_data,
            namespace="ns1",
        )
        print(f"Upserted count: {upsert_response['upserted_count']}")
    else:
        print("No data to upsert")
except Exception as e:
    print(f"Error upserting vectors: {e}")

# Print index statistics
try:
    stats = index.describe_index_stats()
    print("Index statistics:")
    print(stats)
except Exception as e:
    print(f"Error fetching index statistics: {e}")