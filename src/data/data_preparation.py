import base64
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path
from PIL import Image

from utils.templates import DESCRIBING_SLIDE_PROMPT_TEMPLATE


def save_pdf_pages_as_images(pdf_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = convert_from_path(pdf_path)

    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        print(f"Сохранена страница {i + 1} как {image_path}")


def encode_image(image_path):
    """
    Функция для кодирования изображения в формат base64.

    Аргументы:
    image_path: Строка, путь к изображению, которое нужно закодировать.

    Возвращает:
    Закодированное в формате base64 изображение в виде строки.
    """
    with open(image_path, "rb") as image_file:
        # Читаем файл изображения в бинарном режиме и кодируем в base64
        return base64.b64encode(image_file.read()).decode("utf-8")


# Функция для суммаризации изображения с использованием модели GPT
def image_summarize(img_base64, prompt):
    """
    Функция для получения суммаризации изображения с использованием GPT модели.

    Аргументы:
    img_base64: Строка, изображение закодированное в формате base64.
    prompt: Строка, запрос для модели GPT, содержащий инструкцию для суммаризации \
изображения.

    Возвращает:
    Суммаризация изображения, возвращенная моделью GPT.
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
    return msg.content


def prepare_text_from_image(img_path):
    base64_image = encode_image(img_path)
    return image_summarize(base64_image, DESCRIBING_SLIDE_PROMPT_TEMPLATE)


def create_texts_from_images(input_dir, output_dir):
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


if __name__ == "main":
    input_dir = "../../data/interim/images/"
    output_dir = "../../data/interim/texts/"

    create_texts_from_images(input_dir, output_dir)
