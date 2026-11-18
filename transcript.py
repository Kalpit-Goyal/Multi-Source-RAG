from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


def get_video_id(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host not in YOUTUBE_HOSTS:
        raise ValueError("Invalid YouTube URL")

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    else:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
            video_id = parts[1]
        else:
            values = parse_qs(parsed.query).get("v")
            video_id = values[0] if values else ""

    if not video_id:
        raise ValueError("Invalid YouTube URL: missing video id")
    return video_id


def get_transcript(video_id):
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    text = " ".join(chunk.text for chunk in transcript)
    return " ".join(text.split())
