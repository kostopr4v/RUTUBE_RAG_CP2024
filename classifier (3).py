import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BertModel
import re
import pickle

# Символы, которые будут удаляться при предобработке текста
p_d = './,][-")(~!#@^%$;*?&№∙^:<:>=_+\|`1°234}{567890'

# Функция предобработки текста, очищает текст от специальных символов и ссылок
def preprocess(text):
    output = text.replace('\n', ' ').replace('\t', ' ').replace('\u200c', ' ')
    # Удаление URL из текста
    output = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", output)
    output = re.sub(r'^https?:\/\/.*[\r\n]*', '', output, flags=re.MULTILINE)
    # Замена всех символов из p_d на пробелы
    for k in p_d:
        output = output.replace(k,' ')
    # Удаление двойных пробелов
    output = output.replace('  ', ' ')
    return output.strip()

# Определение класса BERT для классификации
class BertCLS(nn.Module):
    def __init__(self, model, n_classes):
        super(BertCLS, self).__init__()
        self.model = model
        # Линейный слой для классификации с количеством выходов, равным числу классов
        self.fc = nn.Linear(1024, n_classes)
    
    # Прямой проход модели, возвращает выход классификатора
    def forward(self, batch):
        return self.fc(self.model(**batch).pooler_output)

# Чтение данных из CSV для получения классов
df = pd.read_csv('./data/train_normal.csv')
# Уникальные классы для первой задачи
classes_1 = df['class_1'].unique()
# Словарь для преобразования строковых классов в индексы
classes_1_str2int = {classes_1[i]:i for i in range(len(classes_1))}
# Словарь для обратного преобразования индексов в строки
classes_1_int2str = {i:classes_1[i] for i in range(len(classes_1))}
# Аналогичные операции для второй задачи
classes_2 = df['class_2'].unique()
classes_2_str2int = {classes_2[i]:i for i in range(len(classes_2))}
classes_2_int2str = {i:classes_2[i] for i in range(len(classes_2))}

# Выбор устройства (CUDA или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализация токенизатора
model_name = "deepvk/USER-bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Инициализация первой модели BERT с классами первой задачи
bert_model = BertModel.from_pretrained(
            model_name, 
            ignore_mismatched_sizes=True,  # Игнорирование несовпадения размеров модели
            num_labels=len(classes_1)  # Количество классов для первой задачи
        )

# Инициализация второй модели BERT с классами второй задачи
bert_model2 = BertModel.from_pretrained(
            model_name, 
            ignore_mismatched_sizes=True, 
            num_labels=len(classes_2)
        )

# Создание экземпляра модели классификации для первой задачи
model_1 = BertCLS(bert_model, n_classes=len(classes_1))

# Загрузка обученной модели для первой задачи
model_1.load_state_dict(torch.load('./models-all-classes/deepvk_mge_class_1.pth', map_location=device))

# Создание экземпляра модели классификации для второй задачи
model_2 = BertCLS(bert_model2, n_classes=len(classes_2))

# Загрузка обученной модели для второй задачи
model_2.load_state_dict(torch.load('./models-all-classes/deepvk_mge_class_2.pth', map_location=device))

# Функция токенизации входных данных
token = lambda model_input: tokenizer(model_input, padding=True,
                    max_length=512, truncation=True,
                    return_tensors='pt')

# Функция предсказания для первой задачи
def predict_1(inputs):
    model_1.eval().cuda()  # Перевод модели в режим оценки и на GPU
    data = token(inputs)  # Токенизация входных данных
    data = data.to(device)  # Перенос данных на устройство (CUDA/CPU)
    embeddings = model_1(data)  # Получение предсказаний
    res = embeddings.argmax(-1).detach().cpu().numpy()[0]  # Индекс класса с наибольшей вероятностью
    return classes_1_int2str[res]  # Возвращение строкового значения класса

# Функция предсказания для второй задачи
def predict_2(inputs):
    model_2.eval().cuda()  # Перевод модели в режим оценки и на GPU
    data = token(inputs)  # Токенизация входных данных
    data = data.to(device)  # Перенос данных на устройство
    embeddings = model_2(data)  # Получение предсказаний
    res = embeddings.argmax(-1).detach().cpu().numpy()[0]  # Индекс класса с наибольшей вероятностью
    return classes_2_int2str[res]  # Возвращение строкового значения класса