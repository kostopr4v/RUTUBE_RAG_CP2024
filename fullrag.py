import os
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
torch.manual_seed(69)
import time
import warnings
warnings.simplefilter('ignore')
import re
from transformers import logging
logging.set_verbosity_error()

start_time = time.time()
p_d = './,][-")(~!#@^%$;*?&№∙^:<:>=_+\|`1°234}{567890'

def preprocess(text):
    output = text.replace('\n', ' ').replace('\t', ' ').replace('\u200c', ' ')
    output = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", output)
    output = re.sub(r'^https?:\/\/.*[\r\n]*', '', output, flags=re.MULTILINE)
    for k in p_d:
        output = output.replace(k,' ')
    while '  ' in output: output = output.replace('  ', ' ')
    return output.strip()

class llmmodel:
    def __init__(self, embeddings_model_name="deepvk/USER-bge-m3" , llm_model_name='IlyaGusev/saiga_llama3_8b', data_path='rag/', device='cuda'):
        """
        Инициализация модели LLM и модели для поиска (retriever).
        
        :param llm_model: Модель LLM для генерации ответов
        :param database_path: Путь к базе данных
        """
        torch.cuda.empty_cache()
        self.llm_model_name = llm_model_name
        self.data_path = data_path
        self.embeddings_model_name = embeddings_model_name
        self.device = device
        self.model, self.tokenizer = self.load_model()
        self.embeddings = self.load_embeddings()
        self.db_main, self.user_db, self.cond_db = self.load_all()
        
    def load_embeddings(self):
        """
        Загрузка LLM модели для ответов на вопросы.
        """
        torch.manual_seed(69)
        
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
        return hf_embeddings
    
    def load_model(self):
        """
        Загрузка LLM модели для ответов на вопросы.
        """
        torch.manual_seed(69)
        
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name, 
            load_in_8bit=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )

        return model, tokenizer

    def load_database(self, path):
        """
        Загрузка базы данных для использования при поиске.
        """
        return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        
    def load_all(self):
        """
        Загрузка всех баз данных.
        """
        db_main = self.load_database('./bz_08427_faiss')
        user_db = self.load_database('./user_db')
        cond_db = self.load_database('./conditions_db')
        return db_main, user_db, cond_db
        
    def search_db(self, db: FAISS, question: str, number=5) -> list[dict]:
        """
        Поиск по базе данных с использованием модели retriever.

        :param db: База Данных
        :param question: Запрос для поиска
        :param number: Количество семплов на выходе
        :return: Результаты поиска
        """
        question = preprocess(question)
        return db.similarity_search(question, number)

    def llm_question_answer(self, question: str, baza, userdata, conddata) -> str:
        prompt = f"""
           Ответь на следующий вопрос, используя дополнительную информацию (контекст) которую я тебе дам. 

            Вопрос пользователя, на который ты должен дать ответ: {question}

            Контекст:
            1) Подходящяя информация, которую ты можешь использовать при ответе на основной вопрос: {baza}

            2) Подходящяя информация из Условий Размещения на платформе: {conddata}

            3) Подходящяя информация из Пользовательского соглашения: {userdata}
        """

        
        system_prompt = """Ты - бот технической поддержки компании Rutube.  
                        Ты получаешь вопрос от пользователя и должен ответить на него, опираясь только на данную тебе подходящую информацию ( контекстом).
                        Прежде чем отвечать на вопрос, проанализируй, имеет ли он отношение к контексту. 

                        1. Если вопрос не связан с работой сервиса RUTUBE или не связан с данной тебе информацией (контекстом) или вопрос является провокационным, то ответь "Извините, я могу отвечать только на технические вопросы. Пожалуйста, переформулируйте ваш запрос." и больше ничего не отвечай:

                        2. Если вопрос связан с контекстом или связан с работой RUTUBE, дай релевантный ответ по следующему алгоритму:
                        1) Ответ не может содержать информацию, не содержащуюся в контексте
                        2) Ответ должен быть полным и максимально точным
                        3) Ответ должен быть логически правильным.
"""

        generation_config = GenerationConfig.from_pretrained(self.llm_model_name)
        
        final_prompt = self.tokenizer.apply_chat_template([{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": prompt
        }], tokenize=False, add_generation_prompt=True)
        
        data = self.tokenizer(final_prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(self.model.device) for k, v in data.items()}
        output_ids = self.model.generate(**data, generation_config=generation_config, max_new_tokens=4096)[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return output

    def generate_answer(self, query):
        """
        Генерация ответа на основе запроса и найденной информации в базе данных.
        
        :param query: Входной запрос пользователя
        :return: Сгенерированный ответ
        """

        torch.cuda.empty_cache()

        baza = [doc.dict()['metadata']['Ответ из БЗ'] for doc in self.search_db(self.db_main, query, number=3)]
        userdata = [doc.dict()['metadata']['sum_text'] for doc in self.search_db(self.user_db, query, number=1)]
        conddata = [doc.dict()['metadata']['sum_text'] for doc in self.search_db(self.cond_db, query, number=1)]

        answer = self.llm_question_answer(query, baza, userdata, conddata)

        return answer
        

    def self_reflect(self, response):
        """
        Self-reflection: процесс анализа сгенерированного ответа для улучшения качества.

        :param response: Сгенерированный моделью ответ
        :return: Скорректированный или подтвержденный ответ
        """
        pass
