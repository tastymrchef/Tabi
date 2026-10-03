"""
Block E -- the Telegram interface.

Everything before this block only ran Tabi from a Python script talking
directly to the graph (test_fix.py, the __main__ in agent/workflow.py).
This turns it into an actual chat bot: one long-running process that
keeps the Postgres-backed checkpointer/store and the compiled agent
graph open for its whole lifetime, and feeds each incoming Telegram
message into that same graph.

Thread and user identity: a Telegram chat's id becomes the LangGraph
thread_id (conversation history is per-chat, same idea as a script
reusing one thread_id across calls), and the Telegram user's id becomes
user_id (so preferences saved via remember_preference are per-person,
not per-chat -- matters once this is a group chat, or the same person
talks to the bot from more than one device).

Requires python-telegram-bot>=21 (async-native, so it shares the same
asyncio loop as the rest of Tabi rather than needing its own thread) and
TELEGRAM_BOT_TOKEN in .env -- get one by messaging @BotFather on
Telegram and following its /newbot flow.
"""

import asyncio
import base64
import logging
import os

import pymupdf
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.agent.workflow import build_agent_graph
from src.memory.db import get_db_uri, postgres_backend

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("tabi.telegram")

# Telegram rejects any single message over this; split long replies
# instead of letting the send fail outright.
TELEGRAM_MESSAGE_LIMIT = 4000

# Vision models take images, not raw PDF bytes, so a PDF gets rasterized
# page-by-page first. Cap how many pages we'll convert/send -- a long
# itinerary PDF doesn't need every page to extract the facts that matter,
# and each page costs real vision tokens.
MAX_PDF_PAGES = 5

DEFAULT_DOCUMENT_CAPTION = (
    "I'm sending a travel document or photo -- take a look, tell me what "
    "it is, and remember any concrete trip facts from it."
)


def _get_bot_token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Add it to your .env file -- message "
            "@BotFather on Telegram to create a bot and get a token."
        )
    return token


async def _handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message:
        await message.reply_text(
            "Hi, I'm Tabi, your travel companion. Ask me about weather, nearby "
            "places, or currency conversions, or say something like 'plan a day "
            "for me in Kyoto' and I'll put together a real, researched itinerary."
        )


def _pdf_to_image_data_urls(pdf_bytes: bytes) -> list[str]:
    data_urls = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page in list(pdf)[:MAX_PDF_PAGES]:
            pix = page.get_pixmap(dpi=150)
            b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
            data_urls.append(f"data:image/png;base64,{b64}")
    return data_urls


async def _run_agent_turn(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    content,
) -> None:
    """Shared by the text and document/photo handlers: feed `content` (a
    plain string or a list of multimodal content blocks) into the agent
    graph on this chat's thread, and send back whatever it replies."""
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user

    agent_graph = context.application.bot_data["agent_graph"]
    config = {"configurable": {"thread_id": str(chat.id)}}

    await context.bot.send_chat_action(chat_id=chat.id, action="typing")

    try:
        result = await agent_graph.ainvoke(
            {
                "messages": [{"role": "user", "content": content}],
                "user_id": str(user.id) if user else str(chat.id),
            },
            config=config,
        )
    except Exception:
        logger.exception("agent_graph.ainvoke failed for chat %s", chat.id)
        await message.reply_text(
            "Something went wrong on my end processing that -- try again in a moment."
        )
        return

    reply = result["messages"][-1].content or (
        "I don't have a reply for that -- could you rephrase?"
    )

    for i in range(0, len(reply), TELEGRAM_MESSAGE_LIMIT):
        await message.reply_text(reply[i : i + TELEGRAM_MESSAGE_LIMIT])


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat

    if not message or not message.text or not chat:
        return

    await _run_agent_turn(update, context, message.text)


async def _handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles photos and uploaded files (Block B). Images go straight to
    the agent as vision input; PDFs get rasterized to images first since
    vision models take images, not raw PDF bytes. The main agent's system
    prompt is what actually decides to call remember_trip_fact -- this
    handler's only job is turning the Telegram attachment into the same
    kind of message content the agent already knows how to read."""
    message = update.effective_message
    chat = update.effective_chat

    if not message or not chat:
        return

    image_data_urls: list[str] = []

    if message.photo:
        tg_file = await context.bot.get_file(message.photo[-1].file_id)
        file_bytes = bytes(await tg_file.download_as_bytearray())
        b64 = base64.b64encode(file_bytes).decode("ascii")
        image_data_urls.append(f"data:image/jpeg;base64,{b64}")
    elif message.document:
        mime_type = message.document.mime_type or ""
        tg_file = await context.bot.get_file(message.document.file_id)
        file_bytes = bytes(await tg_file.download_as_bytearray())

        if mime_type == "application/pdf":
            try:
                image_data_urls = _pdf_to_image_data_urls(file_bytes)
            except Exception:
                logger.exception("Failed to rasterize PDF for chat %s", chat.id)
                await message.reply_text(
                    "I couldn't read that PDF -- it may be corrupted or password-protected."
                )
                return
        elif mime_type.startswith("image/"):
            b64 = base64.b64encode(file_bytes).decode("ascii")
            image_data_urls.append(f"data:{mime_type};base64,{b64}")
        else:
            await message.reply_text(
                "I can read images and PDFs right now -- try sending this as one of those."
            )
            return
    else:
        return

    if not image_data_urls:
        await message.reply_text("I couldn't find anything to read in that file.")
        return

    content = [{"type": "text", "text": message.caption or DEFAULT_DOCUMENT_CAPTION}]
    content.extend(
        {"type": "image_url", "image_url": {"url": url}} for url in image_data_urls
    )

    await _run_agent_turn(update, context, content)


async def main() -> None:
    token = _get_bot_token()

    async with postgres_backend(get_db_uri()) as (checkpointer, store):
        await checkpointer.setup()
        await store.setup()

        agent_graph = await build_agent_graph(checkpointer, store)

        application: Application = ApplicationBuilder().token(token).build()
        application.bot_data["agent_graph"] = agent_graph
        application.add_handler(CommandHandler("start", _handle_start))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message)
        )
        application.add_handler(
            MessageHandler(filters.PHOTO | filters.Document.ALL, _handle_document)
        )

        # Manual lifecycle (rather than Application.run_polling(), which
        # calls asyncio.run() internally and isn't meant to be nested
        # inside code that already owns the loop -- we need this coroutine
        # to keep running inside the same postgres_backend context the
        # whole time the bot is up).
        async with application:
            await application.start()
            await application.updater.start_polling()
            logger.info("Tabi is running -- press Ctrl+C to stop.")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
            finally:
                await application.updater.stop()
                await application.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
