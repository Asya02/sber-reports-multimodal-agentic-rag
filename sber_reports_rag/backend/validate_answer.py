from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from sber_reports_rag.utils.helpers import LLM, MAX_RETRIES, GraphState
from sber_reports_rag.utils.templates import (
    CHECK_HALLUCINATIONS_TEMPLATE,
    CHECK_HALLUCINATIONS_TEMPLATE_SECOND,
)


class GradeHallucinations(BaseModel):
    """Бинарная оценка, указывающая, основан ли ответ на контексте."""

    binary_score: str = Field(description="Ответ основан на фактах: 'да' или 'нет'")


class GradeAnswer(BaseModel):
    """Бинарная оценка, указывающая, дан ли ответ на вопрос."""

    binary_score: str = Field(description="Ответ на вопрос - 'да' или 'нет'.")


def grade_generation_v_documents_and_question(
    state: GraphState, config
) -> Literal["generate", "transform_query", "web_search", "finalize_response"]:
    """
    Оценка соответствия сгенерированного ответа документам и вопросу, и принятие решения о \
следующем шаге.

    Эта функция проверяет, соответствует ли сгенерированный ответ документам, из которых он \
был создан,
    и правильно ли он отвечает на исходный вопрос. В зависимости от результата, функция решает,
    какой следующий шаг выполнить: повторить генерацию, трансформировать запрос, выполнить \
веб-поиск,
    или завершить процесс.

    Args:
        state (GraphState): Текущее состояние графа

    Returns:
        str: Решение о следующем шаге.
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["candidate_answer"]
    web_fallback = state["web_fallback"]
    retries = state["retries"] if state.get("retries") is not None else -1
    max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

    # Если веб-поиск уже был выполнен и возвращаться к нему не нужно
    if not web_fallback:
        return "finalize_response"

    print("Текущее действие: Проверка на галлюцинации - основан ли ответ на контексте.")

    check_hallucinations_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CHECK_HALLUCINATIONS_TEMPLATE),
            (
                "human",
                "Набор фактов: \n\n {documents} \n\n Ответ LLM: {generation}",
            ),
        ]
    )
    hallucination_grader = check_hallucinations_prompt | LLM.with_structured_output(
        GradeHallucinations
    )
    hallucination_grade: GradeHallucinations = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )  # type: ignore

    # Проверка, основан ли ответ на документах
    if hallucination_grade.binary_score == "нет":
        print("Итоги проверки: Ответ модели не основан на документах. Новая попытка.")
        return "generate" if retries < max_retries else "web_search"

    print("Итоги проверки: Ответ модели основан на документах.")
    print("Текущее действие: Проверка на галлюцинации - дан ли ответ на вопрос.")

    # Оценка соответствия ответа вопросу
    check_hallucinations_prompt_second = ChatPromptTemplate.from_messages(
        [
            ("system", CHECK_HALLUCINATIONS_TEMPLATE_SECOND),
            (
                "human",
                "Вопрос пользователя: \n\n {question} \n\n Ответ LLM: {generation}",
            ),
        ]
    )

    answer_grader = check_hallucinations_prompt_second | LLM.with_structured_output(
        GradeAnswer
    )
    answer_grade: GradeAnswer = answer_grader.invoke(
        {"question": question, "generation": generation}
    )  # type: ignore
    if answer_grade.binary_score == "да":
        print("Итоги проверки: Ответ на вопрос дан.")
        return "finalize_response"
    else:
        print("Итоги проверки: Ответ на вопрос не дан.")
        return "transform_query" if retries < max_retries else "web_search"
