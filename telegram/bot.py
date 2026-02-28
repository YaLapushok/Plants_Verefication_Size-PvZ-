import sys
import os
import asyncio
import logging
import io

# Add the root directory to sys.path so we can import 'core' and 'config'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, BufferedInputFile, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart
from config import BOT_TOKEN
from core.ml import process_image

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

user_settings = {}

def get_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔲 Включить рамки (BBox)"), KeyboardButton(text="🔳 Выключить рамки (BBox)")]
        ],
        resize_keyboard=True
    )

@dp.message(CommandStart())
async def handle_start(message: Message):
    await message.reply(
        "👋 Привет! Я бот для классификации и сегментации растений.\n\n"
        "Отправь мне фото растения (пшеница или рукола), и я:\n"
        "1) Определю класс растения.\n"
        "2) Выполню сегментацию по соответствующей модели.\n"
        "3) Пришлю результат тебе!\n\n"
        "Вы можете переключать отображение рамок используя кнопки ниже:",
        reply_markup=get_keyboard()
    )

@dp.message(F.text == "🔲 Включить рамки (BBox)")
async def enable_boxes(message: Message):
    user_settings[message.from_user.id] = True
    await message.reply("✅ Рамки (BBox) на сегментированных изображениях **включены**.", reply_markup=get_keyboard(), parse_mode="Markdown")

@dp.message(F.text == "🔳 Выключить рамки (BBox)")
async def disable_boxes(message: Message):
    user_settings[message.from_user.id] = False
    await message.reply("❌ Рамки (BBox) **выключены**. Вы будете видеть только маски сегментации.", reply_markup=get_keyboard(), parse_mode="Markdown")

@dp.message(F.photo)
async def handle_photo(message: Message):
    processing_msg = await message.reply("⏳ Обрабатываю изображение. Это может занять несколько секунд...")
    
    try:
        # 1. Download the largest photo
        photo = message.photo[-1]
        image_buffer = io.BytesIO()
        await bot.download(photo, destination=image_buffer)
        image_bytes = image_buffer.getvalue()
        
        # 2. Process via ML core
        show_boxes = user_settings.get(message.from_user.id, True)
        result = process_image(image_bytes, show_boxes=show_boxes)
        
        if result.get('error'):
            raise Exception(result['error'])
            
        plant_name_ru = result['class_name']
        metrics_text = result['metrics_text']
        
        # 3. Send back annotated image
        result_file = BufferedInputFile(result['annotated_image_bytes'], filename="result.jpg")
        await message.reply_photo(
            photo=result_file,
            caption=f"Классификация: **{plant_name_ru}**{metrics_text}",
            parse_mode="Markdown"
        )
        
        # 4. Send statistics chart separately if available
        if result.get('chart_bytes'):
            chart_file = BufferedInputFile(result['chart_bytes'], filename="chart.jpg")
            await message.reply_photo(
                photo=chart_file,
                caption="📊 Диаграмма соотношения площадей частей растения."
            )

    except Exception as e:
        logging.error(f"Error processing image in Telegram bot: {e}")
        await message.reply("❌ Произошла ошибка при обработке изображения. Возможно, оно повреждено или модель не смогла его обработать.")
    finally:
        await processing_msg.delete()

@dp.message()
async def handle_text(message: Message):
    await message.reply("Пожалуйста, отправь мне фото растения как обычное изображение (используйте кнопку-скрепку).")

async def main():
    logging.info("Starting bot...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
