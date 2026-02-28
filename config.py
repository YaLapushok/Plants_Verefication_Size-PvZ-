import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("Пожалуйста, укажите TELEGRAM_BOT_TOKEN в файле .env или переменных окружения.")
