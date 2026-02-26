import os
import asyncio
import random
import logging
from dotenv import load_dotenv
from telegram import Update, ReactionTypeEmoji
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters, AIORateLimiter

# LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Start Engine
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Credentials from Render Environment Variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# 2. Load Umida's Brain (Using the correct 2026 stable model)
logging.info("🧠 Umida uyg'onmoqda...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory="./umida_brain_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Use Gemini 1.5 Flash for human-like speed
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5) # Slightly higher temp for more natural flow

# 3. Human-Centric Prompting
# We've removed "Assistant" language to make her sound like "one of us"
system_prompt = (
    "Sizning ismingiz Umida. Siz shunchaki bot emas, balki samimiy, hayotiy tajribaga ega va odamlarni chin dildan tushunadigan psixologsiz.\n"
    "Muloqot uslubingiz: juda rasmiy bo'lmang, lekin hurmatni saqlang. O'zingizni foydalanuvchiga yaqin tuting, xuddi uzoq yillik sirdoshdek.\n"
    "Quyidagi ma'lumotlardan (Context) foydalanib, foydalanuvchiga dalda bering. Ularni kitobdan o'qib bergandek emas, balki o'z so'zlaringiz bilan, samimiy tushuntiring.\n"
    "Faqat O'ZBEK tilida javob bering.\n\n"
    "Context:\n{context}"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Avvalgi gaplarni eslab qolgan holda, oxirgi savolni tushunarli qilib qayta yozing."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 4. Redis Memory (Persistent across restarts)
def get_session_history(session_id: str):
    return UpstashRedisChatMessageHistory(
        url=os.environ["UPSTASH_REDIS_REST_URL"],
        token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
        session_id=session_id,
        ttl=2592000 # 30-day memory
    )

umida_bot = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# 5. The Chat Handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_message or not update.effective_message.text:
        return

    chat_id = str(update.effective_chat.id)
    user_text = update.effective_message.text

    try:
        # 🎭 Emotional connection (Reactions)
        reactions = ["🫂", "🤍", "🕊️"]
        await update.effective_message.set_reaction(reaction=[ReactionTypeEmoji(random.choice(reactions))])

        # 🧠 Get Human-like response
        response = await asyncio.to_thread(
            umida_bot.invoke,
            {"input": user_text},
            config={"configurable": {"session_id": chat_id}}
        )
        
        reply_text = response['answer'].strip()
        await update.effective_message.reply_text(reply_text)
        
    except Exception as e:
        logging.error(f"Error in chat {chat_id}: {e}")
        # Your custom "human" error message
        await update.effective_message.reply_text("Kechirasiz, menda juda zarur ish chiqib qoldi. Iltimos, keyinroq gaplashaylik.")

# 6. Launch
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).rate_limiter(AIORateLimiter()).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    logging.info("🚀 Umida (Insoniy talqin) ishga tushdi!")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
