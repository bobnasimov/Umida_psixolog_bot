import os, logging, asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from google import genai
from google.genai import types

# Pulls from Render Environment Variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALLOWED_CHATS = [46636732] 

client = genai.Client(api_key=GEMINI_API_KEY)
logging.basicConfig(level=logging.INFO)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id not in ALLOWED_CHATS: 
        return
    
    # Retry loop with a 60-second cooldown for Free Tier limits
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction="You are Olga. Sarcastic, brief, opinionated. Use 💀 naturally.",
                    safety_settings=[types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")]
                ),
                contents=update.message.text
            )
            return await update.message.reply_text(response.text or "💀")
            
        except Exception as e:
            err_msg = str(e)
            
            # If rate limited (429), wait 60 seconds before trying again
            if "429" in err_msg:
                logging.warning("Rate limit hit. Cooling down for 60s...")
                await asyncio.sleep(60)
                continue
            
            # Reports other real errors (Auth, etc.)
            logging.error(f"REAL ERROR: {err_msg}")
            return await update.message.reply_text(f"Error: {err_msg[:100]}... 💀")

async def start_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # Force kill webhooks to prevent "Conflict" errors
    await app.bot.delete_webhook(drop_pending_updates=True)
    
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    print("🔥 Olga is live. 60s cooldown logic active.")
    
    async with app:
        await app.updater.start_polling()
        await app.start()
        while True: await asyncio.sleep(1)

if __name__ == '__main__':
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        pass
