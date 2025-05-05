from langgraph.graph import StateGraph, END
from slack_bot.utils import (
    intent_node,
    paper_search_node,
    llm_node,
    rag_node,
    refine_node,
    implementation_node,
)

# Function to build a workflow
def build_research_flow():
    graph = StateGraph(dict)

    graph.add_node("intent", intent_node)
    graph.add_node("arxiv_search", paper_search_node)
    graph.add_node("llm_answer", llm_node)
    graph.add_node("rag_answer", rag_node)
    graph.add_node("refine_answer", refine_node)
    graph.add_node("implementation", implementation_node)

    graph.set_entry_point("intent")

    def route(state: dict) -> str:
        return state.get("intent", "general_question")

    graph.add_conditional_edges("intent", route, {
        "search_papers": "arxiv_search",
        "general_question": "llm_answer",
        "refine_question": "refine_answer",
    })

    graph.add_edge("arxiv_search", "rag_answer")
    graph.add_edge("rag_answer", "implementation")
    graph.add_edge("llm_answer", "implementation")
    graph.add_edge("refine_answer", "implementation")

    graph.add_edge("implementation", END)

    return graph.compile()

research_flow = build_research_flow()