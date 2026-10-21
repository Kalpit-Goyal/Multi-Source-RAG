from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import re
load_dotenv()

def get_video_id(url):
    parsed=urlparse(url)

    if parsed.hostname=="youtu.be":
        return parsed.path[1:]

    if parsed.hostname in ("www.youtube.com","youtube.com"):
        return parse_qs(parsed.query)["v"][0]

    raise ValueError("Invalid Youtube URL")

def get_transcript(video_id):
    transcript=YouTubeTranscriptApi().fetch(video_id,languages=["en"],)
    text=" ".join(chunk.text for chunk in transcript)
    text = " ".join(text.split())
    return text

