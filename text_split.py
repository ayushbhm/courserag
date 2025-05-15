import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 👇 Step 1: Light segmentation of raw dumped transcript
def lightly_segment_transcript(text, every_n_words=20):
    words = text.split()
    segments = [
        " ".join(words[i:i+every_n_words]) + "."
        for i in range(0, len(words), every_n_words)
    ]
    return "\n".join(segments)

# 👇 Step 2: Set up the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 👇 Step 3: Input/output directories
transcript_dir = "transcripts"
output_dir = "chunks"
os.makedirs(output_dir, exist_ok=True)

# 👇 Step 4: Loop through .txt files
all_chunks = []
for filename in os.listdir(transcript_dir):
    if filename.endswith("_text.txt"):
        filepath = os.path.join(transcript_dir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Derive video title from filename
        video_title = filename.replace("_text.txt", "").replace("_", " ")

        # Preprocess and split
        segmented_text = lightly_segment_transcript(raw_text)
        documents = splitter.create_documents(
    texts=[segmented_text],
    metadatas=[{"video_title": video_title}]
)



        # Save individual chunks if needed
        for i, doc in enumerate(documents):
            chunk_filename = f"{video_title.replace(' ', '_')}_chunk_{i}.txt"
            with open(os.path.join(output_dir, chunk_filename), 'w', encoding='utf-8') as out_f:
                out_f.write(doc.page_content)

        # Save metadata for embedding step
        for doc in documents:
            all_chunks.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        print(f"✅ Processed: {video_title} → {len(documents)} chunks")

# 👇 Save all chunks to a JSONL file for later use
with open("all_chunks.jsonl", "w", encoding='utf-8') as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print("🎉 All transcripts split and saved.")
