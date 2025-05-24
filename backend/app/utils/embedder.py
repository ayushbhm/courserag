from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    """
    Convert text to embedding vector using sentence-transformers
    """
    return model.encode(text).tolist() 