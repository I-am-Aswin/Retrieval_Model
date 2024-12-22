import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from preprocess import load_data_from_csv
from faiss_search import search_faiss
from a_star import a_star_search

def retreive(query):
    # Load sentences and embeddings
    sentences = load_data_from_csv("/home/aswin/Projects/Edu_AI/data/data.csv")
    embeddings = np.load("/home/aswin/Projects/Edu_AI/embeddings/embeddings.npy")

    # Load FAISS index
    index = faiss.read_index("/home/aswin/Projects/Edu_AI/embeddings/faiss_index")

    # Query for search
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    query_embedding = model.encode([query])

    # FAISS Search
    indices, _ = search_faiss(query_embedding, index, top_k=10)
    candidate_embeddings = embeddings[indices]
    candidate_sentences = [sentences[i] for i in indices]

    # A* Search
    results = a_star_search(query_embedding[0], candidate_embeddings, candidate_sentences, top_k=3)
    
    return results
    # Output results
    

if __name__ == "__main__":
    
    query = "What is the function of an operating system?"
    print("\nQuery:", query)
    print("\nTop Results:")
    for idx, result in enumerate(retreive( query )):
        print(f"{idx + 1}. {result}")
    
    
