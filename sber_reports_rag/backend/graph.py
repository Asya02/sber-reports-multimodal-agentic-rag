from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sber_reports_rag.backend.nodes import (
    document_search,
    finalize_response,
    generate,
    transform_query,
    web_search,
)
from sber_reports_rag.backend.validate_answer import (
    grade_generation_v_documents_and_question,
)
from sber_reports_rag.utils.helpers import GraphConfig, GraphState


def workflow_compiler() -> CompiledStateGraph:

    workflow = StateGraph(GraphState, config_schema=GraphConfig)

    # Define the nodes
    workflow.add_node("document_search", document_search)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("finalize_response", finalize_response)

    # Build graph
    workflow.set_entry_point("document_search")
    workflow.add_edge("document_search", "generate")
    workflow.add_edge("transform_query", "document_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("finalize_response", END)

    workflow.add_conditional_edges(
        "generate", grade_generation_v_documents_and_question
    )

    # Compile
    graph = workflow.compile()
    return graph


# graph = workflow_compiler()
# inputs = {"messages": [("human", "Как сбер адаптировался к новым условиям?")]}
# res = graph.invoke(inputs)
# print(res)

# inputs2 = {"messages": [("human", "Сколько сотрудников в Сбере")]}
# res2 = graph.invoke(inputs2)
# print(res2)
