import faiss
import numpy as np

def build_faiss_index(embeddings):
    """
    Build a FAISS index from embeddings and save it.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, "/home/aswin/Projects/Edu_AI/embeddings/faiss_index")
    return index

def search_faiss(query_embedding, index, top_k=5):
    """
    Search the FAISS index for the top-k nearest neighbors.
    """
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]

if __name__ == "__main__":
    # Load embeddings
    embeddings = np.load("/home/aswin/Projects/Edu_AI/embeddings/embeddings.npy")

    # Build FAISS index
    index = build_faiss_index(embeddings)
    print("FAISS index built and saved.")
