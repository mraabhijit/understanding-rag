import argparse


from lib.search_utils import (
    load_golden_data,
)
from lib.hybrid_search import (
    rrf_search_command
)


def evaluate_command(rrf_results: list[dict], data: dict, limit: int):
    retrieved_list = [doc['title'] for doc in rrf_results]
    relevant_retrieved_set = set(data['relevant_docs']).intersection(retrieved_list)
    precision = len(relevant_retrieved_set) / len(retrieved_list)
    recall = len(relevant_retrieved_set) / len(data['relevant_docs'])
    f1_score = (2 * precision * recall) / (precision + recall)
    print(f"\n\n- Query: {data['query']}")
    print(f"  - Precision@{limit}: {precision:.4f}")
    print(f"  - Recall@{limit}: {recall:.4f}")
    print(f"  - F1 Score: {f1_score:.4f}")
    print(f"  - Retrieved: {', '.join(retrieved_list)}")
    print(f"  - Relevant: {', '.join(data['relevant_docs'])}")


def main():
    parser = argparse.ArgumentParser(description="search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    test_data = load_golden_data()
    print(f"k={limit}")
    for data in test_data:
        rrf_results = rrf_search_command(query=data['query'], limit=limit)
        evaluate_command(rrf_results, data, limit)


if __name__ == '__main__':
    main()
