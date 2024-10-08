import base64
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path


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
    prompt = """Опиши ключевую информацию, которая представлена на изображении. \
Описание должно быть конкретным и точным. Обрати особое внимание на графики, диаграммы \
или визуальные элементы, которые можно проанализировать.
    Очень важно не упустить детали. Ответь в формате Markdown."""
    base64_image = encode_image(img_path)
    return image_summarize(base64_image, prompt)
