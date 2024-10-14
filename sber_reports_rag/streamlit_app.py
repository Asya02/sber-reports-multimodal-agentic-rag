import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from sber_reports_rag.backend.graph import workflow_compiler

st.set_page_config(page_title="sber_reports_rag")
st.title("Чат-бот, отвечающий на вопросы по отчёту Сбера за 2023 год")


@st.cache_resource
def get_graph_to_streamlit() -> CompiledStateGraph:
    """
    Возвращает скомпилированный граф.

    Returns:
        CompiledStateGraph: Скомпилированный объект графа состояний.
    """
    graph = workflow_compiler()
    return graph


if "graph" not in st.session_state:
    st.session_state.graph = get_graph_to_streamlit()


def show_ui(prompt_to_user="Спроси к меня что-нибудь") -> None:
    """
    Отображает пользовательский интерфейс чата.

    Args:
        prompt_to_user (str): Первоначальный приветственный текст для пользователя.

    Returns:
        None
    """
    # Инициализация списка сообщений в состоянии сессии, если он отсутствует
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Отображение всех сообщений в чате
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Обработка пользовательского ввода
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Генерация нового ответа, если последнее сообщение не от ассистента
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                inputs = {"messages": [("human", prompt)]}
                response = st.session_state.graph.invoke(inputs)["candidate_answer"]
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


show_ui()
