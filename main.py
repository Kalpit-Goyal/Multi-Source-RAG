from transcript import get_video_id, get_transcript
from chunking import create_chunks

url="https://www.youtube.com/watch?v=FDEgZzJlY7s"

video_id=get_video_id(url)

text=get_transcript(video_id)

chunks=create_chunks(text)

print(len(chunks))

print(chunks[1])