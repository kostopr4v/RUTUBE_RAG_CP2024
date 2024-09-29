import random
from aiogram import Bot, Dispatcher, types
from aiogram.types import  ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils import executor
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import os
from pydub import AudioSegment

from fullrag import llmmodel
from whisper import translate_audio


API_TOKEN = '7830435119:AAFjmuG6OZAPv2fmCUHVB_9nRDmmPsTa2Rc'
# Правильные логин и пароль для модератора
MODERATOR_LOGIN = "moderator"
MODERATOR_PASSWORD = "password123"

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
dp.middleware.setup(LoggingMiddleware())

moderator_id = None  # Идентификатор модератора
user_id = None       # Идентификатор пользователя
question_data = {}   # Хранение информации о вопросе, включая сгенерированное число

rag_model = llmmodel()

def get_llm_answer(question):
    print('START GENERATING ANSWER')
    an = rag_model.generate_answer(question).strip()
    print('FINISHED GENERATING ANSWER')
    return 'Здравствуйте!' + '\n' + an

if not os.path.exists("voices"):
    os.makedirs("voices")

# Состояния для регистрации модератора
class ModeratorForm(StatesGroup):
    waiting_for_login = State()
    waiting_for_password = State()

class EditNumberForm(StatesGroup):
    waiting_for_custom_number = State()

# Кнопка для входа модератора
def get_moderator_keyboard():
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(KeyboardButton("👮 Войти как оператор"))
    keyboard.add(KeyboardButton("❓ Задать вопрос"))
    return keyboard

def get_emoji_keyboard():
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(KeyboardButton("👍"))
    keyboard.add(KeyboardButton("👎"))
    return keyboard

# Кнопки для модератора: редактировать или оставить число
def get_moderator_choice_keyboard():
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("Редактировать", callback_data="edit_number"))
    keyboard.add(InlineKeyboardButton("Оставить", callback_data="leave_number"))
    return keyboard

async def save_voice_message(voice: types.Voice, file_name: str):
    file = await bot.get_file(voice.file_id)
    await bot.download_file(file.file_path, file_name)

    # Конвертация OGG в MP3
    audio = AudioSegment.from_file(file_name, format="ogg")
    audio.export(file_name, format="mp3")

# Функция обработки голосового сообщения (для примера всегда возвращает "Привет")
def process_voice_message_to_text(file_name: str) -> str:
    # Заглушка для обработки голосового сообщения
    result = translate_audio(file_name)
    print(result)
    return result


# Хендлер для старта бота с приветственным сообщением
@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    global user_id
    user_id = message.from_user.id
    welcome_text = (
        "👋 Добро пожаловать в QnA ассистент от Rutube!\n\n"
        "Задайте свой вопрос, и оператор технической поддержки ответит на него как можно скорее.\n"
        "оператор может войти через специальную кнопку."
    )
    await message.reply(welcome_text, reply_markup=get_moderator_keyboard())

# Хендлер для нажатия на кнопку "Войти как модератор"
@dp.message_handler(lambda message: message.text == "👮 Войти как оператор")
async def moderator_login_start(message: types.Message):
    await message.reply("Введите логин:")
    await ModeratorForm.waiting_for_login.set()  # Устанавливаем состояние ожидания логина

# Хендлер для ввода логина
@dp.message_handler(state=ModeratorForm.waiting_for_login)
async def process_login(message: types.Message, state: FSMContext):
    if message.text == MODERATOR_LOGIN:
        await state.update_data(login=message.text)
        await message.reply("Теперь введите пароль:")
        await ModeratorForm.waiting_for_password.set()  # Переходим к состоянию ожидания пароля
    else:
        await message.reply("Неверный логин. Попробуйте снова.")

# Хендлер для ввода пароля
@dp.message_handler(state=ModeratorForm.waiting_for_password)
async def process_password(message: types.Message, state: FSMContext):
    if message.text == MODERATOR_PASSWORD:
        global moderator_id
        moderator_id = message.from_user.id
        await message.reply("Вы успешно вошли как оператор.\nСюда будут приходить вопросы от пользователей, а также ответы сгенерированные LLM.\nВы сможете редактировать сообщение, предложенное нейросетью и ответить пользователю", reply_markup=types.ReplyKeyboardRemove())
        await state.finish()  # Заканчиваем состояние FSM
    else:
        await message.reply("Неверный пароль. Попробуйте снова.")
        await state.finish()

# Хендлер для пользователя "Задать вопрос"
@dp.message_handler(lambda message: message.text == "❓ Задать вопрос")
async def user_question_start(message: types.Message):
    await message.reply("Пожалуйста, введите ваш вопрос.")

@dp.message_handler(content_types=types.ContentType.VOICE)
async def handle_voice_message(message: types.Message):
    global moderator_id
    if moderator_id:

        voice = message.voice
        file_name = f"voices/voice_{message.from_user.id}_{message.message_id}.mp3"
        
        # Сохраняем голосовое сообщение
        await save_voice_message(voice, file_name)

        # Обрабатываем голосовое сообщение
        processed_text = str(process_voice_message_to_text(file_name))
        question_data['question'] = processed_text
        question_data['llm_answer'] = get_llm_answer(processed_text)


        # Отправляем обработанный текст модератору
        msg = await bot.send_message(moderator_id, f"Вопрос от пользователя: {question_data['question']}\nСгенерированное число: {question_data['llm_answer']}",
                               reply_markup=get_moderator_choice_keyboard())
        question_data['moderator_message_id'] = msg.message_id
        await message.reply("Ваш голосовой вопрос был отправлен оператору.")
    else:
        await message.reply("Извините, оператор не доступен в данный момент.")


# Хендлер для получения вопросов от пользователя
@dp.message_handler(lambda message: message.from_user.id == user_id and moderator_id is not None)
async def user_question(message: types.Message):
    if message.text not in ["👍", "👎"]:
        if moderator_id:
            await message.reply("Ваш вопрос отправлен оператору.")
            question_data['question'] = message.text
            question_data['llm_answer'] = get_llm_answer(message.text)

            # Отправляем обработанный текст модератору
            msg = await bot.send_message(moderator_id, f"Вопрос от пользователя: {question_data['question']}\nСгенерированное ответ: {question_data['llm_answer']}",
                                reply_markup=get_moderator_choice_keyboard())
            question_data['moderator_message_id'] = msg.message_id

        else:
            await message.reply("Извините, оператор не доступен в данный момент.")
    else:pass

# Callback хендлер для обработки выбора модератора
@dp.callback_query_handler(lambda c: c.data in ["edit_number", "leave_number"])
async def process_moderator_choice(callback_query: types.CallbackQuery):
    # Удаляем кнопки после нажатия
    await bot.edit_message_reply_markup(
        chat_id=moderator_id,
        message_id=question_data['moderator_message_id'],
        reply_markup=None
    )

    if callback_query.data == "edit_number":
        # Модератор выбрал редактировать число
        await bot.send_message(moderator_id, "Введите новое сообщение для пользователя.")
        await EditNumberForm.waiting_for_custom_number.set()  # Устанавливаем состояние ожидания нового сообщения
    elif callback_query.data == "leave_number":
        # Модератор выбрал оставить сгенерированное число
        await bot.send_message(user_id, f"Ответ от оператора: {question_data['llm_answer']}", reply_markup=get_emoji_keyboard())
        await bot.send_message(moderator_id, "Ответ отправлен пользователю.")  # Уведомляем модератора
        await bot.answer_callback_query(callback_query.id)

# Хендлер для ввода нового сообщения модератором
@dp.message_handler(state=EditNumberForm.waiting_for_custom_number)
async def process_custom_message(message: types.Message, state: FSMContext):
    await bot.send_message(user_id, f"Ответ от оператора: {message.text}", reply_markup=get_emoji_keyboard())
    await message.reply("Ваш ответ отправлен пользователю.")
    await state.finish()  # Завершаем состояние

# Хендлер для обработки неизвестных команд
@dp.message_handler()
async def handle_message(message: types.Message):
    if message.text not in ["👍", "👎"]:
        await message.reply("Неизвестная команда. Используйте /start для начала.")
    else:
        pass

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
