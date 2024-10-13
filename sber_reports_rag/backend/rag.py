import os

from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sber_reports_rag.utils.helpers import TOKENIZER


def load_all_documents_from_folder(folder_path: str) -> list[Document]:
    """
    Загружает все текстовые файлы из указанной папки и возвращает список документов.

    Проходит по всем файлам в папке, проверяет, что файл имеет расширение `.txt`, и загружает
    его содержимое с использованием TextLoader. Файлы, которые не являются текстовыми, игнорируются.
    В случае ошибки при загрузке файла выводит сообщение с описанием ошибки.

    Args:
        folder_path (str): Путь к папке, содержащей текстовые файлы.

    Returns:
        List[Document]: Список загруженных документов. Каждый документ представлен в виде объекта \
`Document`.
    """
    documents = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith(".txt"):
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
                print(f"Загружен файл: {filename}")
            except Exception as e:
                print(f"Ошибка при загрузке файла {filename}: {e}")

    return documents


def count_tokens(text: str) -> int:
    """
    Подсчитывает количество токенов в заданном тексте с использованием токенизатора.

    Функция использует глобальный токенизатор (TOKENIZER) для кодирования текста и
    возвращает количество токенов. Поддерживаются текстовые строки или байтовые строки.

    Args:
        text (str): Текст для подсчета токенов.

    Returns:
        int: Количество токенов, закодированных из текста.
    """
    tokens = TOKENIZER.encode(text)
    return len(tokens)


def create_retriever() -> BaseRetriever:
    """
    Создаёт и возвращает объект `Retriever` для извлечения данных на основе текстов из указанной \
папки.

    Функция загружает все документы из папки `texts`, разбивает их на текстовые чанки с заданными \
параметрами,
    затем сохраняет эти документы в векторное хранилище `Chroma`. Используется модель эмбеддингов
    "intfloat/multilingual-e5-large". Созданный `Retriever` возвращается для дальнейшего \
использования.

    Returns:
        BaseRetriever: Объект `Retriever`
    """
    # Путь к папке с документами
    dir = os.path.dirname(__file__)
    folder_path = os.path.join(dir, r"..\..\data\interim\texts")
    docs = load_all_documents_from_folder(folder_path)

    print(f"Загружено документов: {len(docs)}")

    # Параметры для разбиения текста на чанки
    chunk_size = 512
    chunk_overlap = 0
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
    )

    # Разбиваем документы на чанки
    splits = text_splitter.split_documents(docs)

    # Путь для хранения векторного хранилища
    vectorstore_path = os.path.join(dir, r"..\..\data\vectorstore")

    # Создаём векторное хранилище и сохраняем документы
    db = Chroma.from_documents(
        persist_directory=vectorstore_path,
        documents=splits,
        embedding=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
        collection_name="sber-reports",
    )
    retriever = db.as_retriever()
    return retriever


def get_retriever() -> BaseRetriever:
    """
    Загружает векторное хранилище и возвращает объект `Retriever` для извлечения данных.

    Функция подключается к уже существующему векторному хранилищу Chroma по указанному пути
    и использует модель эмбеддингов "intfloat/multilingual-e5-large". Возвращает объект `Retriever`
    для дальнейшего использования в извлечении релевантной информации.

    Returns:
        BaseRetriever: Объект `Retriever`
    """
    # Определяем путь к векторному хранилищу
    dir = os.path.dirname(__file__)
    vectorstore_path = os.path.join(dir, r"..\..\data\vectorstore")

    # Загружаем векторное хранилище с использованием Chroma
    db = Chroma(
        collection_name="sber-reports",
        persist_directory=vectorstore_path,
        embedding_function=HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large"
        ),
    )
    retriever = db.as_retriever()
    return retriever


RETRIVER = get_retriever()
# print(RETRIVER.get_relevant_documents("сотрудников сколько"))
