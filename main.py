import argparse
import sys
from pathlib import Path

from chunking import chunk_documents
from loaders import load_file, load_youtube
from rag import answer_question
from vector_store import index_source


if hasattr(sys.stdout, "reconfigure"):
	sys.stdout.reconfigure(encoding="utf-8")


def index_target(target: str) -> None:
	"""Index a YouTube URL or a local file path."""
	if target.startswith(("http://", "https://")):
		loaded = load_youtube(target)
	else:
		path = Path(target)
		if not path.is_file():
			raise FileNotFoundError(target)
		loaded = load_file(path.name, path.read_bytes())

	chunks = chunk_documents(loaded.documents, loaded.source_id)
	count = index_source(loaded, chunks)
	print(f"Indexed {loaded.source_name}: {count} chunks.")


def main() -> None:
	parser = argparse.ArgumentParser(description="Chat across indexed sources.")
	parser.add_argument(
		"sources",
		nargs="+",
		help="YouTube URLs and/or file paths (PDF, Excel, CSV, Word, text, images).",
	)
	parser.add_argument("--question", help="Ask one question and exit")
	args = parser.parse_args()

	for target in args.sources:
		index_target(target)

	history: list[dict] = []

	if args.question:
		answer, _ = answer_question(args.question, history=history)
		print(f"\n{answer}")
		return

	print("Sources indexed. Ask questions, or type 'exit' to quit.")
	while True:
		question = input("\nYou: ").strip()
		if question.lower() in {"exit", "quit"}:
			break
		if not question:
			continue
		answer, _ = answer_question(question, history=history)
		history += [
			{"role": "user", "content": question},
			{"role": "assistant", "content": answer},
		]
		print(f"AI: {answer}")


if __name__ == "__main__":
	main()
