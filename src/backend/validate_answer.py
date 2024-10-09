from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from utils.helpers import LLM, MAX_RETRIES, GraphState
from utils.templates import CHECK_HALLUCINATIONS_TEMPLATE


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Ответ основан на фактах: 'да' или 'нет'")


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(description="Ответ на вопрос - 'да' или 'нет'.")


def grade_generation_v_documents_and_question(
    state: GraphState, config
) -> Literal["generate", "transform_query", "web_search", "finalize_response"]:
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["candidate_answer"]
    web_fallback = state["web_fallback"]
    retries = state["retries"] if state.get("retries") is not None else -1
    max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

    # this means we've already gone through web fallback and can return to the user
    if not web_fallback:
        return "finalize_response"

    print("---CHECK HALLUCINATIONS---")

    check_hallucinations_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CHECK_HALLUCINATIONS_TEMPLATE),
            (
                "human",
                "Вопрос пользователя: \n\n {question} \n\n Ответ LLM: {generation}",
            ),
        ]
    )
    hallucination_grader = check_hallucinations_prompt | LLM.with_structured_output(
        GradeHallucinations
    )
    hallucination_grade: GradeHallucinations = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )  # type: ignore

    # Check hallucination
    if hallucination_grade.binary_score == "нет":
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "generate" if retries < max_retries else "web_search"

    print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
    print("---GRADE GENERATION vs QUESTION---")

    # Check question-answering
    answer_grader = check_hallucinations_prompt | LLM.with_structured_output(
        GradeAnswer
    )
    answer_grade: GradeAnswer = answer_grader.invoke(
        {"question": question, "generation": generation}
    )  # type: ignore
    if answer_grade.binary_score == "да":
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "finalize_response"
    else:
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "transform_query" if retries < max_retries else "web_search"
