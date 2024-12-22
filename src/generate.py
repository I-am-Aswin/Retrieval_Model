from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the pipeline for text generation
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # Use CPU

def generate_content(content, query):
    """
    Generate a structured paragraph based on the retrieved content and query.
    """
    try:
        # Improved and explicit prompt
        prompt = (
            f"Query: '{query}'\n"
            f"Details retrieved:\n"
            f"{content}\n\n"
            f"Based on the above, provide a cohesive and clear explanation:"
        )

        # Generate response
        response = generator(
            prompt,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,  # Adjust for randomness
            top_k=50,  # Use top-k sampling
            top_p=0.9  # Use nucleus sampling
        )

        return response[0]["generated_text"]
    except Exception as e:
        return f"Error generating content: {e}"

if __name__ == "__main__":
    # Example content and query
    content = (
        "An operating system is a control program. "
        "An operating system is similar to a government. "
        "Finally, we describe how operating systems are created and how a computer starts its operating system."
    )
    query = "What is an operating system?"
    
    result = generate_content(content, query)
    print(result)
