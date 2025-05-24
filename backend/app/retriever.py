import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Setup
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="video_chunks")
search_model = SentenceTransformer("all-MiniLM-L6-v2")
answer_model = pipeline("text2text-generation", model="google/flan-t5-small")

def get_videos():
    """Get available video titles"""
    data = collection.get()
    videos = {meta['video_title'] for meta in data['metadatas'] if meta}
    return sorted(videos)

def ask(question, video=None):
    """Ask a question about videos"""
    # Search for relevant content
    query_embedding = search_model.encode(question)
    search_params = {"query_embeddings": [query_embedding], "n_results": 3}
    
    if video:
        search_params["where"] = {"video_title": {"$eq": video}}
    
    results = collection.query(**search_params)
    
    if not results['documents'][0]:
        return "No relevant content found."
    
    # Generate answer
    context = "\n\n".join(results['documents'][0])
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    answer = answer_model(prompt, max_length=200)[0]['generated_text']
    return answer

# Interactive mode
if __name__ == "__main__":
    print("Available videos:", get_videos())
    
    while True:
        video = input("\nVideo (or Enter for all): ").strip() or None
        question = input("Question: ").strip()
        
        if question.lower() in ['quit', 'exit']:
            break
            
        print("Answer:", ask(question, video))