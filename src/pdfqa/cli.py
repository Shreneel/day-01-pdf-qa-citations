"""Command line: python -m pdfqa.cli data/sample/northwind_handbook.pdf "How many PTO days do I get?" """

from __future__ import annotations

import argparse
import sys

from pdfqa.pipeline import PDFQA


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions about a PDF and get page-cited answers.")
    parser.add_argument("pdf", help="Path to a PDF file")
    parser.add_argument("question", nargs="?", help="Question to ask. Omit for interactive mode.")
    parser.add_argument("-k", type=int, default=None, help="Number of chunks to retrieve")
    args = parser.parse_args(argv)

    qa = PDFQA.from_pdf(args.pdf)
    print(f"Indexed {len(qa.pages)} pages / {len(qa.chunks)} chunks · retriever={qa.retriever.name} · model={qa.model.name}\n")

    questions = [args.question] if args.question else iter(lambda: input("Question (blank to quit): ").strip(), "")
    for question in questions:
        answer = qa.ask(question, k=args.k)
        print(answer.text)
        for c in answer.citations:
            print(f"  · p.{c.page}: {c.snippet}")
        if answer.invalid_pages:
            print(f"  ! cited pages not in retrieved context: {answer.invalid_pages}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
