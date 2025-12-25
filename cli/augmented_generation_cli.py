import argparse

from lib.rag_search import rag_command, summarize_command, citation_command, question_answer_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize RAG results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument("--limit", type=int, help="Number of results to fetch", default=5)

    citation_parser = subparsers.add_parser(
        "citations", help="Summarize RAG results with citations"
    )
    citation_parser.add_argument("query", type=str, help="Search query for RAG")
    citation_parser.add_argument("--limit", type=int, help="Number of results to fetch", default=5)

    question_parser = subparsers.add_parser(
        "question", help="Answer the question based on movies dataset"
    )
    question_parser.add_argument("query", type=str, help="Question to answer")
    question_parser.add_argument("--limit", type=int, help="Number of results to fetch", default=5)


    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_command(args.query)
        case "summarize":
            summarize_command(args.query, args.limit)
        case "citations":
            citation_command(args.query, args.limit)
        case "question":
            question_answer_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()