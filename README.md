# Интеллектуальный помощник оператора службы поддержки

Этот проект предоставляет готовую реализацию **Retrieval-Augmented Generation** (RAG) системы, которая использует языковую модель для генерации ответов на основе поиска по базе данных (retriever). В основе системы лежат предварительно обученные модели LLM и FAISS для поиска релевантных документов.

## Установка

Перед началом работы убедитесь, что у вас установлен Python версии 3.8 и выше. Необходимые библиотеки можно установить с помощью `pip`:

```bash
pip install torch transformers langchain langchain_community faiss-cpu pandas tqdm
```


### 1. **Использование LLM**

1. Инициализация модели с указанием языка и пути к данным:

```python
from fullrag import llmmodel

# Инициализируем объект RAG-модели
rag = llmmodel(llm_model_name="Qwen/Qwen2.5-7B-Instruct")
```

2. Поиск и генерация ответа на запрос:

```python
question = "Как восстановить пароль на платформе?"

# Генерация ответа на запрос
answer = rag.generate_answer(question)
print(answer)
```

### 2. **Обработка текста**
Метод `preprocess` используется для очистки входного текста от ненужных символов и ссылок.

```python
clean_text = preprocess("Пример текста с лишними символами: https://example.com!")
print(clean_text)  # Вывод: Пример текста с лишними символами
```

### 3. **Использование API**
```python
import requests
url = 'correctly-awake-dogfish.ngrok-free.app/predict'
def get_ans(text)
	data_json = {'question':text}
	return requests.post(url, json=data_json).json()
```

### 4. **Использование TG-бота**
- **Ссылка на телеграм бота**: https://t.me/theboyscpbot
- ![Демо](https://raw.githubusercontent.com/kostopr4v/RUTUBE_RAG_CP2024/main/цп.gif)

## Параметры и конфигурация

- **Модель LLM**: Вы можете использовать любую предобученную модель LLM из библиотеки HuggingFace, передав её название в `llm_model_name`. Например, `"Qwen/Qwen2.5-7B-Instruct"` или `"IlyaGusev/saiga_llama3_8b"`.
- **Устройство вычислений**: По умолчанию используется `"cuda"`, если у вас есть GPU. Если вы работаете на CPU, установите `device="cpu"`.

## Требования

- Python 3.8 и выше
- `torch` — для запуска языковой модели.
- `transformers` — для работы с моделями LLM и embeddings.
- `faiss-cpu` — для поиска по базе данных.
- `pandas` и `tqdm` — для работы с данными и отображения прогресса.
