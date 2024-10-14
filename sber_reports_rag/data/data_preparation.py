import base64
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path
from PIL import Image

from sber_reports_rag.utils.templates import DESCRIBING_SLIDE_PROMPT_TEMPLATE


def save_pdf_pages_as_images(pdf_path: str, output_dir: str) -> None:
    """
    Сохраняет страницы PDF файла в виде изображений в формате PNG.

    Args:
        pdf_path (str): Путь к PDF файлу.
        output_dir (str): Директория, куда сохранять изображения.

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = convert_from_path(pdf_path)

    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        print(f"Сохранена страница {i + 1} как {image_path}")


def encode_image(image_path: str) -> str:
    """
    Кодирует изображение в формат base64.

    Args:
        image_path (str): Путь к изображению.

    Returns:
        str: Закодированное в формате base64 изображение.
    """
    with open(image_path, "rb") as image_file:
        # Читаем файл изображения в бинарном режиме и кодируем в base64
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64: str, prompt: str) -> str:
    """
    Получает суммаризацию изображения с использованием GPT модели.

    Args:
        img_base64 (str): Закодированное в формате base64 изображение.
        prompt (str): Запрос для GPT модели, инструкция для суммаризации.

    Returns:
        str: Суммаризация изображения, возвращенная моделью.
    """
    chat = ChatOpenAI(model="gpt-4o-mini")
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},  # Запрос для модели
                    {
                        "type": "image_url",  # Тип содержимого - изображение
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    # Возвращаем содержимое ответа от модели
    return msg.content  # type: ignore


def prepare_text_from_image(img_path: str) -> str:
    """
    Подготавливает текстовое описание изображения, закодированного в base64.

    Args:
        img_path (str): Путь к изображению.

    Returns:
        Optional[str]: Суммаризация изображения.
    """
    base64_image = encode_image(img_path)
    return image_summarize(base64_image, DESCRIBING_SLIDE_PROMPT_TEMPLATE)


def create_texts_from_images(input_dir: str, output_dir: str) -> None:
    """
    Генерирует текстовые файлы на основе изображений из директории.

    Args:
        input_dir (str): Папка, содержащая изображения.
        output_dir (str): Папка, куда будут сохранены текстовые файлы.

    Returns:
        None
    """
    # Проверяем, существует ли папка для текстовых файлов, если нет, создаем её
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Проходим по каждому файлу в папке с изображениями
    for image_filename in os.listdir(input_dir):
        # Полный путь к изображению
        image_path = os.path.join(input_dir, image_filename)

        # Проверяем, что это изображение (например, файл с расширением PNG или JPG)
        if image_filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            # Открываем изображение, если оно действительно является изображением
            try:
                with Image.open(image_path):
                    result = prepare_text_from_image(image_path)

                    # Генерируем имя файла для текстового результата
                    output_filename = os.path.splitext(image_filename)[0] + ".txt"
                    output_path = os.path.join(output_dir, output_filename)

                    # Записываем результат функции в текстовый файл
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(result)
                    print(
                        f"Результат для {image_filename} сохранен в {output_filename}"
                    )
            except Exception as e:
                print(f"Ошибка при обработке {image_filename}: {e}")


dir = os.path.dirname(__file__)
# pdf_path = os.path.join(dir, r"../../data/raw/Сбер 2023.pdf")
# output_dir = os.path.join(dir, r"../../data/interim/images/")
# save_pdf_pages_as_images(pdf_path, output_dir)
input_dir = os.path.join(dir, r"../../data/interim/images/")
output_dir = os.path.join(dir, r"../../data/interim/texts/")
create_texts_from_images(input_dir, output_dir)
# if __name__ == "main":
#     input_dir = "../../data/interim/images/"
#     output_dir = "../../data/interim/texts/"

#     create_texts_from_images(input_dir, output_dir)
