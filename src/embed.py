from sentence_transformers import SentenceTransformer
import numpy as np

from preprocess import load_data_from_csv

def generate_embeddings(sentences, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings for a list of sentences.
    """
    model = SentenceTransformer(model_name, device='cpu')
    embeddings = model.encode(sentences)
    return embeddings

if __name__ == "__main__":

    # Load preprocessed data
    data = load_data_from_csv("/home/aswin/Projects/Edu_AI/data/data.csv")
    embeddings = generate_embeddings(data)

    # Save embeddings to a file
    np.save("../embeddings/embeddings.npy", embeddings)
    print("Embeddings saved.")
