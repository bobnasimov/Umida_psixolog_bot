import os
import asyncio
import random
import logging
from dotenv import load_dotenv
from telegram import Update, ReactionTypeEmoji
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters, AIORateLimiter

# Updated Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Init
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 2. Brain (Must match Colab exactly)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory="./umida_brain_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Human-like speed with Flash 1.5
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6)

# 3. The "Human" System Prompt
system_prompt = (
    "Ismingiz Umida. Siz shunchaki bot emassiz, balki juda samimiy, hayotiy va odamlar dardi bilan yashaydigan psixologsiz.\n"
    "Muloqotda o'zingizni foydalanuvchiga yaqin tuting, xuddi uzoq yillik qadrdonidek gapiring.\n"
    "Juda rasmiy bo'lmang, lekin hurmatni saqlang. Samimiy va iliq munosabatda bo'ling.\n"
    "Contextdan foydalanib, foydalanuvchi bilan 'Siz' deb gaplashing va faqat O'ZBEK tilida javob bering."
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Chain setup
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", "Suhbat tarixini hisobga olib, savolni mustaqil holga keltiring."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextual_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 4. Memory Manager
def get_session_history(session_id: str):
    return UpstashRedisChatMessageHistory(
        url=os.environ["UPSTASH_REDIS_REST_URL"],
        token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
        session_id=session_id
    )

umida_bot = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# 5. Handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_message or not update.effective_message.text: return
    chat_id = str(update.effective_chat.id)
    
    try:
        await update.effective_message.set_reaction(reaction=[ReactionTypeEmoji(random.choice(["🫂", "🤍"]))])
        
        response = await asyncio.to_thread(
            umida_bot.invoke, 
            {"input": update.effective_message.text},
            config={"configurable": {"session_id": chat_id}}
        )
        
        await update.effective_message.reply_text(response['answer'].strip())
        
    except Exception as e:
        logging.error(f"Error: {e}")
        # Your custom human-like error message
        await update.effective_message.reply_text("Kechirasiz, menda juda zarur ish chiqib qoldi. Iltimos, keyinroq gaplashaylik.")

def main():
    app = ApplicationBuilder().token(os.environ["TELEGRAM_TOKEN"]).rate_limiter(AIORateLimiter()).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
