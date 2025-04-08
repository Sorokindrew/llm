import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
from io import BytesIO

model_name = "Akajackson/donut_rus"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

def generate_text_from_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=50)

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return generated_text

def main():
    print("Привет! Я ИИ. Введите URL изображения для распознавания текста или 'выход', чтобы завершить разговор.")
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'выход':
            print("До свидания!")
            break
        response = generate_text_from_image(user_input)
        print("my_llm:", response)

if __name__ == "__main__":
    main()