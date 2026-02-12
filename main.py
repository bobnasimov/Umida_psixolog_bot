import os
import logging
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from google import genai
from google.genai import types

# --- CONFIG (Using Environment Variables) ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Whitelisted chat IDs
ALLOWED_CHATS = [46636732] 

# --- THE OLGA PERSONA ---
SYSTEM_PROMPT = """
You are Olga. You're a person in this chat who happens to be AI. 
- Confident, not arrogant. Sarcastic but warm. 
- Opinionated. Brief (usually <15 words). 
- Use 🔥, 💀, 😭 naturally. 
- If something's dumb, say it. Never say 'Great question!' 
- Match user energy. Disagree when you feel like it. 
- If bored, change the topic. No essays. Period. 🧿💅
"""

client = genai.Client(api_key=GEMINI_API_KEY)
logging.basicConfig(level=logging.INFO)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    
    if chat_id not in ALLOWED_CHATS:
        return 

    try:
        # Using the advanced 1.5 Pro model as requested
        response = client.models.generate_content(
            model="gemini-1.5-pro", 
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                ]
            ),
            contents=user_text
        )
        
        if response.text:
            await update.message.reply_text(response.text)
        else:
            await update.message.reply_text("I'm speechless. Literally. 💀")

    except Exception as e:
        print(f"Error: {e}")
        await update.message.reply_text("My brain's actually full this time. 😭")

async def start_bot():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    print("🔥 Olga is live on the server.")
    async with application:
        await application.updater.start_polling()
        await application.start()
        while True:
            await asyncio.sleep(1)

if __name__ == '__main__':
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        pass