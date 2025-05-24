from flask import Blueprint, request, jsonify

import chromadb
from transformers import pipeline



from sentence_transformers import SentenceTransformer   
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text( text: str):
    return model.encode(text)


# Setup Blueprint
chat_bp = Blueprint("chat", __name__)

# Initialize global variables
client = None
collection = None
answer_model = pipeline("text2text-generation", model="google/flan-t5-base")


def init_chat():
    global client, collection, answer_model
    # Setup ChromaDB client and collection
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("video_chunks")
    # Setup answer generation model
    answer_model = pipeline("text2text-generation", model="google/flan-t5-base")

@chat_bp.route("/chat", methods=["POST"])
def chat():
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
            
        question = data.get("question")
        video = data.get("video_title")  # Optional
        
        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Embed the question
        query_embedding = embed_text(question)
        print(f"Query embedding created. Type: {type(query_embedding)} Length: {len(query_embedding)}")
        
        count = collection.count()
        print(f"Number of documents in collection: {count}")
        
        # Prepare search parameters
        search_params = {
            "query_embeddings": [query_embedding], 
            "n_results": 3
        }
        
        # Filter by video if specified
        

        
        # Search in ChromaDB
        results = collection.query(**search_params)
        print(f"Raw results keys: {results.keys()}")
        
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        print(f"Returned docs (no filter): {docs}")
        print(f"Returned metadatas (no filter): {metas}")
        if video:
           video_norm = video.strip().lower().replace(" ", "_")
           filtered_docs = []
           filtered_metas = []
           for doc, meta in zip(docs, metas):
              stored_title = meta.get("video_title", "").lower()
              if video_norm in stored_title:
                 filtered_docs.append(doc)
                 filtered_metas.append(meta)
        
        if not docs:
            return jsonify({
                "answer": "No relevant content found.",
                "sources": []
            }), 200
        
        # Generate answer
        context = "\n\n".join(docs)
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        answer = answer_model(prompt, max_length=200, do_sample=False)[0]["generated_text"]
        
        # Get source information
        sources = []
        for meta in results.get("metadatas", [[]])[0]:
            if meta and "video_title" in meta:
                sources.append(meta["video_title"])
        
        return jsonify({
            "answer": answer,
            "sources": list(set(sources)),  # Remove duplicates
            "chunks_used": len(docs)
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@chat_bp.route("/videos", methods=["GET"])
def get_videos():
    """Get list of available videos"""
    try:
        data = collection.get()
        videos = set()
        
        for meta in data['metadatas']:
            if meta and 'video_title' in meta:
                videos.add(meta['video_title'])
        
        return jsonify({
            "videos": sorted(list(videos)),
            "count": len(videos)
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@chat_bp.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    try:
        # Test database connection
        collection.count()
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500