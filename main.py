import argparse

from chunking import create_chunks
from rag import answer_question
from transcript import get_transcript, get_video_id
from vector_store import store_transcript


def index_video(url: str) -> str:
	video_id = get_video_id(url)
	chunks = create_chunks(get_transcript(video_id))
	count = store_transcript(video_id, chunks)
	print(f"Indexed {count} chunks for video {video_id}.")
	return video_id


def main() -> None:
	parser = argparse.ArgumentParser(description="Chat with YouTube transcripts.")
	parser.add_argument("url", help="YouTube video URL to index")
	parser.add_argument("--question", help="Ask one question and exit")
	args = parser.parse_args()
	video_id = index_video(args.url)

	if args.question:
		answer, _ = answer_question(args.question, video_id=video_id)
		print(f"\n{answer}")
		return

	print("Transcript indexed. Ask questions, or type 'exit' to quit.")
	while True:
		question = input("\nYou: ").strip()
		if question.lower() in {"exit", "quit"}:
			break
		if question:
			answer, _ = answer_question(question, video_id=video_id)
			print(f"AI: {answer}")


if __name__ == "__main__":
	main()