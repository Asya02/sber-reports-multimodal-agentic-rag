from langchain_core.documents import Document
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from sber_reports_rag.backend.rag import RETRIVER
from sber_reports_rag.utils.helpers import LLM, TAVILY_SEARCH_TOOL, GraphState
from sber_reports_rag.utils.templates import QUERY_REWRITING_TEMPLATE, RAG_TEMPLATE


def document_search(state: GraphState) -> GraphState:
    """
    Извлекает документы на основе последнего сообщения в состоянии графа.

    Функция извлекает последний вопрос из состояния, использует ретривер для поиска документов,
    а затем возвращает обновлённое состояние с найденными документами.

    Args:
        state (GraphState): Текущее состояние графа.

    Returns:
        GraphState: Обновлённое состояние с добавленным ключом "documents", содержащим извлечённые \
документы.
    """
    print("Текущее действие: Поиск релевантных документов")

    # Получаем последний вопрос из сообщений
    question = convert_to_messages(state["messages"])[-1].content

    # Извлекаем документы с помощью ретривера
    documents = RETRIVER.invoke(question)  # type: ignore

    # Возвращаем обновлённое состояние с документами
    return {"documents": documents, "question": question, "web_fallback": True}  # type: ignore


def generate(state: GraphState) -> GraphState:
    """
    Генерирует ответ с использованием модели LLM на основе предоставленных документов и вопроса.

    Функция использует контекст извлечённых документов и вопроса для создания запроса к модели
    и генерирует ответ. Количество попыток (retries) увеличивается на 1 при каждом вызове.

    Args:
        state (GraphState): Текущее состояние графа, содержащее вопрос, документы и другие данные.

    Returns:
        GraphState: Обновлённое состояние с добавленным ключом "candidate_answer", который содержит
        сгенерированный ответ, и ключом "retries", который увеличивается на 1 при каждом вызове.
    """
    print("Текущее действие: Генерация")

    # Получаем вопрос и документы из состояния
    question = state["question"]
    documents = state["documents"]

    # Получаем количество попыток (если есть), иначе ставим -1
    retries = state["retries"] if state.get("retries") is not None else -1

    # Создаём шаблон запроса для RAG
    rag_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    # Строим chain
    rag_chain = rag_prompt | LLM | StrOutputParser()

    # Генерируем ответ с использованием контекста документов и вопроса
    generation = rag_chain.invoke({"context": documents, "question": question})

    # Возвращаем обновлённое состояние с увеличенным количеством попыток и сгенерированным ответом
    return {"retries": retries + 1, "candidate_answer": generation}


def transform_query(state: GraphState) -> GraphState:
    """
    Преобразует исходный вопрос, чтобы получить более точную и улучшенную формулировку запроса.

    Функция использует шаблон запроса и модель LLM для переписывания исходного вопроса в более
    качественной форме. Обновлённый вопрос сохраняется в ключе "question" в состоянии графа.

    Args:
        state (GraphState): Текущее состояние графа, содержащее исходный вопрос и другие данные.

    Returns:
        GraphState: Обновлённое состояние с обновлённым ключом "question", содержащим
        улучшенную версию исходного вопроса.
    """
    print("Текущее действие: Переписывание запроса")

    # Получаем исходный вопрос из состояния
    question = state["question"]

    # Создаём шаблон запроса для переписывания вопроса
    query_rewriting_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUERY_REWRITING_TEMPLATE),
            (
                "human",
                "Вот исходный вопрос: \n\n {question} \n Сформулируйте улучшенный вопрос.",
            ),
        ]
    )

    # Переписываем вопрос
    query_rewriter = query_rewriting_prompt | LLM | StrOutputParser()

    # Генерируем улучшенный вопрос
    better_question = query_rewriter.invoke({"question": question})

    # Возвращаем обновлённое состояние с улучшенным вопросом
    return {"question": better_question}


def web_search(state: GraphState) -> GraphState:
    """
    Выполняет веб поиск по вопросу и добавляет результаты поиска к списку документов.

    Функция использует веб-инструмент для поиска по вопросу, полученному из состояния графа,
    извлекает контент из результатов поиска, и добавляет их в список документов с указанием
    источника как "websearch". Обновлённое состояние возвращается с обновлёнными документами
    и ключом "web_fallback", установленным в False.

    Args:
        state (GraphState): Текущее состояние графа, содержащее вопрос и документы.

    Returns:
        GraphState: Обновлённое состояние с добавленными результатами поиска в ключе "documents",
        и ключом "web_fallback", установленным в False.
    """
    print("Текущее действие: Web поиск с Tavily")

    # Получаем вопрос и текущие документы из состояния
    question = state["question"]
    documents = state["documents"]

    # Выполняем поиск с использованием веб-инструмента
    search_results = TAVILY_SEARCH_TOOL.invoke(question)

    # Извлекаем контент из результатов поиска и объединяем его в строку
    search_content = "\n".join([d["content"] for d in search_results])

    # Добавляем новый документ с результатами поиска в список документов
    documents.append(
        Document(page_content=search_content, metadata={"source": "websearch"})
    )

    # Возвращаем обновлённое состояние с документами и устанавливаем web_fallback в False
    return {"documents": documents, "web_fallback": False}


def finalize_response(state: GraphState) -> GraphState:
    """
    Функция извлекает сгенерированный ответ из состояния графа, создаёт сообщение от имени ИИ
    с этим ответом и возвращает обновлённое состояние с этим сообщением.

    Args:
        state (GraphState): Текущее состояние графа, содержащее ключ "candidate_answer",
        который представляет сгенерированный ответ.

    Returns:
        GraphState: Обновлённое состояние, содержащее список сообщений с ответом от ИИ в ключе \
"messages".
    """
    print("Текущее действие: Вывод финального ответа")

    return {"messages": [AIMessage(content=state["candidate_answer"])]}
