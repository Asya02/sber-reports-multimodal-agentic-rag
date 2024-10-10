import streamlit as st

from sber_reports_rag.backend.graph import workflow_compiler

st.set_page_config(page_title="sber_reports_rag")
st.title("Мультимодальный RAG на годовом отчёте Сбера")


@st.cache_resource
def get_graph_to_streamlit():
    graph = workflow_compiler()
    return graph


graph = get_graph_to_streamlit()


def show_ui(prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                inputs = {"messages": [("human", prompt)]}
                response = graph.invoke(inputs)["candidate_answer"]
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


show_ui()
