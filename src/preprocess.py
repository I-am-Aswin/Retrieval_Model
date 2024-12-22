import re
import pandas as pd

def preprocess(text):
    """
    Clean and preprocess text.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def load_data_from_csv(file_path):
    """Load and preprocess data from a CSV file."""
    df = pd.read_csv(file_path)
    return [preprocess(str(line)) for line in df["Data"]]

if __name__ == "__main__":
    # Example usage
    data = load_data_from_csv("/home/aswin/Projects/Edu_AI/data/data.csv")
    print(*data[:5], sep='\n')
