import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BertModel
import re
import pickle
from classifier import predict_1, predict_2  # Импорт функций предсказания из модуля classifier


# Символы, которые удаляются при предобработке текста
p_d = './,][-")(~!#@^%$;*?&№∙^:<:>=_+\|`1°234}{567890'

# Функция для предобработки текста: удаление символов и ссылок
def preprocess(text):
    output = text.replace('\n', ' ').replace('\t', ' ').replace('\u200c', ' ')
    # Удаление ссылок из текста
    output = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", output)
    output = re.sub(r'^https?:\/\/.*[\r\n]*', '', output, flags=re.MULTILINE)
    # Замена всех символов из p_d на пробелы
    for k in p_d:
        output = output.replace(k,' ')
    # Удаление двойных пробелов
    output = output.replace('  ', ' ')
    return output.strip()

# Чтение данных из CSV файла для создания классов
df = pd.read_csv('./data/train_normal.csv')
# Уникальные классы для первой задачи классификации
classes_1 = df['class_1'].unique()
# Словарь для преобразования классов в индексы и наоборот
classes_1_str2int = {classes_1[i]:i for i in range(len(classes_1))}
classes_1_int2str = {i:classes_1[i] for i in range(len(classes_1))}
# Аналогичная структура для второй задачи классификации
classes_2 = df['class_2'].unique()
classes_2_str2int = { classes_2[i]:i for i in range(len(classes_2))}
classes_2_int2str = { i:classes_2[i] for i in range(len(classes_2))}

# Определение устройства для вычислений (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Импорт предобученной модели и токенизатора
model_name = "deepvk/USER-bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Импорт модели RAG для генерации ответа
from fullrag import llmmodel
rag = llmmodel()

# Определение структуры для запросов от пользователя
class Request(BaseModel):
    question: str  # Вопрос, который будет задаваться пользователем

# Определение структуры для ответа
class Response(BaseModel):
    answer: str  # Ответ, сгенерированный моделью
    class_1: str  # Предсказанный класс первой классификации
    class_2: str  # Предсказанный класс второй классификации

# Инициализация FastAPI приложения
app = FastAPI()

# Простая начальная точка, которая выводит сообщение
@app.get("/")
def index():
    return {"text": "Интеллектуальный помощник оператора службы поддержки."}

# Основной API-эндпоинт для предсказания классов и генерации ответа
@app.post("/predict")
async def predict_sentiment(request: Request):
    text = request.question  # Получаем вопрос из запроса
    # Поиск в базе данных (БЗ) с помощью модели RAG
    bz = rag.search_db(rag.db_main, text)
    # Извлекаем 5 вопросов из базы знаний и тему
    quests, theme = [bz[i].dict()['metadata']['Вопрос из БЗ'] for i in range(5)], bz[0].dict()['metadata']['Тема']
    
    # Подготавливаем текст для классификации, добавляя тему и вопросы из БЗ
    text_class = text + ' | ' + theme
    for i in range(len(quests)):
        text_class += ' | ' + quests[i]
    
    # Предсказание классов для вопроса
    class_1, class_2 = predict_1(text), predict_2(text)
    
    # Формируем ответ на запрос
    response = Response(
        answer=rag.generate_answer(text),  # Генерация ответа моделью RAG
        class_1=class_1,  # Первый класс
        class_2=class_2,  # Второй класс
    )
    return response


