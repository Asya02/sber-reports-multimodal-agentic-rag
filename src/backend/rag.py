import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import TOKENAZER


def load_all_documents_from_folder(folder_path):
    documents = []

    # Проходим по каждому файлу в указанной папке
    for filename in os.listdir(folder_path):
        # Полный путь к файлу
        file_path = os.path.join(folder_path, filename)

        # Проверяем, что это текстовый файл
        if filename.lower().endswith(".txt"):
            try:
                # Используем TextLoader для загрузки содержимого файла
                loader = TextLoader(
                    file_path, encoding="utf-8"
                )  # Задаем кодировку, если нужно
                documents.extend(
                    loader.load()
                )  # Загружаем содержимое и добавляем в список
                print(f"Загружен файл: {filename}")
            except Exception as e:
                print(f"Ошибка при загрузке файла {filename}: {e}")

    return documents


def count_tokens(text):
    tokens = TOKENAZER.encode(text)
    return len(tokens)


def get_retriever():
    # Пример использования функции
    folder_path = "../data/interim/texts"
    docs = load_all_documents_from_folder(folder_path)

    print(f"Загружено документов: {len(docs)}")

    chunk_size = 512
    chunk_overlap = 0
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
    )
    splits = text_splitter.split_documents(docs)
    db = Chroma.from_documents(
        splits,
        HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
        collection_name="rag-sber-reports",
    )
    retriever = db.as_retriever()
    return retriever
