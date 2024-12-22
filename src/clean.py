import re
import pandas as pd

def extract_sentence_data(text):
    pattern = r"Sentence (\d+):(.*?)(?:\n|$)"
    matches = re.findall(pattern, text, re.DOTALL)

    final = []
    for match in matches:

        tmp_data = match[1].strip()
        final.append( [match[0], tmp_data])

    return final  # Return list of tuples (sentence_number, data)

def create_dataframe(data):
    if not data:
        return pd.DataFrame(columns=["Sentence", "Data"])

    try:
        df = pd.DataFrame(data, columns=["Sentence", "Data"])
        #Convert Sentence Number to numeric
        df["Sentence"] = pd.to_numeric(df["Sentence"])
        return df
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return pd.DataFrame(columns=["Sentence", "Data"])


if __name__ == '__main__':
    
    with open('/home/aswin/Projects/Edu_AI/data/chap_2.txt', 'r', encoding='utf-8') as f:
        content = f.read()
        
        cleaned_data = create_dataframe( extract_sentence_data(content) )

        cleaned_data.to_csv("/home/aswin/Projects/Edu_AI/data/data.csv", index=False, encoding='utf-8')