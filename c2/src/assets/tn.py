from youtube_transcript_api import YouTubeTranscriptApi
import json
import json5
import os
import re

# Helper function to extract YouTube video ID from URL
def extract_video_id(url):
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError(f"Cannot extract video ID from URL: {url}")
    return video_id

# Load the JavaScript file and convert it to JSON
with open('c2/src/assets/deepLearning.js', 'r', encoding='utf-8') as f:
    js_content = f.read()

# Remove "export default" to get raw JSON-like content
json_like = re.sub(r'export\s+default\s+', '', js_content).strip()

# Parse it with json5
course_data = json5.loads(json_like)

# Output folders
os.makedirs('transcripts', exist_ok=True)

# Track metadata
video_metadata = {}

# Loop through all videos
for week in course_data:
    for video in week['videos']:
        video_title = video['title']
        video_url = video['url']
        try:
            video_id = extract_video_id(video_url)
            video_metadata[video_id] = {
                'title': video_title,
                'url': video_url,
                'week': week['title']
            }

            # Fetch transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)

            # Convert transcript to plain text
            transcript_text = " ".join([seg["text"] for seg in transcript])

            # Clean filename
            safe_title = re.sub(r'[^\w\-_ ]', '', video_title).replace(" ", "_")

            # Save files
            with open(f'transcripts/{safe_title}_raw.json', 'w', encoding='utf-8') as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)

            with open(f'transcripts/{safe_title}_text.txt', 'w', encoding='utf-8') as f:
                f.write(transcript_text)

            print(f"✅ Saved transcript for: {video_title}")

        except Exception as e:
            print(f"❌ Error fetching transcript for {video_title}: {e}")

# Save metadata
with open('video_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(video_metadata, f, ensure_ascii=False, indent=2)

print("🎉 All done!")
