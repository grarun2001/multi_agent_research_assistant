from app.workflow.graph import build_graph


def main():

    print("=" * 60)
    print("       MULTI-AGENT RESEARCH ASSISTANT")
    print("=" * 60)

    graph = build_graph()

    while True:

        query = input("\nEnter your query (or type 'exit'): ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        result = graph.invoke({
            "query": query,
            "results": [],
            "final_answer": ""
        })

        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)

        print(result["final_answer"])


if __name__ == "__main__":
    main()