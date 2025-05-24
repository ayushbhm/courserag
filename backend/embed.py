import os
import uuid
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize Chroma client using new PersistentClient API
client = chromadb.PersistentClient(path="chroma_db")  

collection = client.get_or_create_collection(name="video_chunks")


model = SentenceTransformer("all-MiniLM-L6-v2")


chunks_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chunks"))

def embed_and_store_chunks():
    for filename in os.listdir(chunks_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(chunks_folder, filename)
            
            with open(file_path, "r", encoding="utf-8") as file:
                chunk_text = file.read()

            # Embed chunk
            embedding = model.encode(chunk_text)

            # Create unique ID for Chroma
            chunk_id = str(uuid.uuid4())

            # Extract video title from filename
            video_title = filename.replace("_", " ").replace(".txt", "")

            # Store in Chroma
            collection.add(
                ids=[chunk_id],
                documents=[chunk_text],
                embeddings=[embedding],
                metadatas=[{"video_title": video_title}]
            )

            print(f"✅ Stored chunk from: {filename}")

if __name__ == "__main__":
    embed_and_store_chunks()
