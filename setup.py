import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_models():
    print("Downloading embedding model (all-MiniLM-L6-v2)...")
    try:
        # Download and save embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_model.save('models/all-MiniLM-L6-v2')
        print("✓ Embedding model saved successfully")
    except Exception as e:
        print(f"✗ Error downloading embedding model: {e}")
        return False
    
    print("Downloading TinyLlama model and tokenizer...")
    try:
        # Download and save TinyLlama model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        
        tokenizer.save_pretrained('models/tinyllama')
        model.save_pretrained('models/tinyllama')
        print("✓ TinyLlama model and tokenizer saved successfully")
    except Exception as e:
        print(f"✗ Error downloading TinyLlama model: {e}")
        return False

    print("Setup complete! You can now run the RAG system.")
    return True

if __name__ == "__main__":
    setup_models()