"""
Telegram Handler - Floating cursor and robust sequential results.
"""

import asyncio
import os
import re
import subprocess
import sys
import html
from telegram import (
    Update,
    Message,
    ReplyKeyboardMarkup,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand
)
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)

from .config import TOKEN, MY_ID, TMP_DIR, WORKSPACE_DIR, TASKS_FILE
from .memory import set_current_session
from .engine import call_gemini_stream, get_short_summary, ACTIVE_SUBPROCESSES, STOP_SIGNAL, LIVE_BUFFERS, clear_live_buffer
from .logger import logger, conv_logger

import whisper
from gtts import gTTS

# Load Whisper model
logger.info("Loading Whisper model...")
WHISPER_MODEL = whisper.load_model("tiny")

CHAT_LOCKS = {}
GLOBAL_APPLICATION = None

TOOL_MAPPING = {
    "list_directory": "ReadFolder",
    "read_file": "ReadFile",
    "write_file": "WriteFile",
    "replace": "EditFile",
    "run_shell_command": "Bash",
    "grep_search": "Search",
    "google_web_search": "WebSearch",
    "web_fetch": "WebRead",
    "invoke_agent": "SubAgent"
}

TOOL_ICONS = {
    "invoke_agent": "👥",
    "run_shell_command": "🛠️",
    "google_web_search": "🌐",
    "web_fetch": "📖"
}


def is_not_user(update: Update) -> bool:
    user_id = update.effective_user.id
    if user_id != MY_ID:
        logger.warning(f"Unauthorized: {user_id}")
        return True
    return False


async def _handle_attachments(message: Message) -> str:
    user_input = message.text or ""
    caption = message.caption or ""
    
    if message.voice or message.audio:
        try:
            audio_obj = await (message.voice or message.audio).get_file()
            audio_path = os.path.join(TMP_DIR, audio_obj.file_path.split('/')[-1])
            await audio_obj.download_to_drive(audio_path)
            result = WHISPER_MODEL.transcribe(audio_path, language="fr")
            transcription = result.get("text", "").strip()
            
            # Combine transcription with existing input or caption
            parts = []
            if transcription: parts.append(f"{transcription}\n[VOICE_TRANSCRIPTION]")
            if caption: parts.append(caption)
            if user_input and user_input != caption: parts.append(user_input)
            user_input = "\n\n".join(parts)
        except Exception as e: logger.error(f"STT Error: {e}")
        return user_input

    if not (message.photo or message.document): 
        return user_input or caption

    try:
        is_photo = bool(message.photo)
        file_obj = await (message.photo[-1] if is_photo else message.document).get_file()
        
        # Try to get original filename for documents
        orig_name = message.document.file_name if message.document else file_obj.file_path.split('/')[-1]
        file_path = os.path.join(TMP_DIR, orig_name)
        
        await file_obj.download_to_drive(file_path)
        user_input = f"{caption or 'Analysis'}\n[FILE: {file_path}]"
    except Exception as e: logger.error(f"File Error: {e}")
    return user_input


def _get_clean_user_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\[(ACTION_INDEX|VOICE_TRANSCRIPTION|FILE|SEND_IMAGE):.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "<thinking" in text.lower(): text = re.split(r"<thinking", text, flags=re.IGNORECASE)[0]
    return text.strip()


def _format_html_response(text: str) -> str:
    """Safe formatting for Telegram HTML, protecting code blocks."""
    if not text: return ""
    
    # Protect code blocks
    placeholders = []
    def save_code(match):
        content = match.group(1)
        safe_content = html.escape(content)
        placeholders.append(f"<code>{safe_content}</code>")
        return f"[[CODE_BLOCK_{len(placeholders)-1}]]"
    
    # Blocks and inline code
    text = re.sub(r"```(?:[\w]+)?\n?(.*?)```", save_code, text, flags=re.DOTALL)
    text = re.sub(r"`(.*?)`", save_code, text)

    # Escape everything else
    text = html.escape(text)

    # Basic formatting
    text = re.sub(r"^[ \t]*#{1,6}\s*(.*?)[ \t]*$", r"<b>\1</b>", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)

    # Restore code
    for i, placeholder in enumerate(placeholders):
        text = text.replace(f"[[CODE_BLOCK_{i}]]", placeholder)

    return text.strip()


async def _reply_with_voice(chat_id, text: str, reply_to_message_id=None):
    try:
        clean = re.sub(r'<[^>]+>', '', _get_clean_user_text(text))
        if not clean.strip(): return
        path = os.path.join(TMP_DIR, f"reply_{int(asyncio.get_event_loop().time())}.mp3")
        gTTS(text=clean, lang='fr').save(path)
        with open(path, 'rb') as v:
            await GLOBAL_APPLICATION.bot.send_voice(chat_id=chat_id, voice=v, reply_to_message_id=reply_to_message_id)
    except: pass


async def trigger_scheduled_task(prompt: str):
    if not GLOBAL_APPLICATION: return
    await _process_request(MY_ID, prompt)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or is_not_user(update): return
    chat_id = update.effective_chat.id
    if chat_id not in CHAT_LOCKS: CHAT_LOCKS[chat_id] = asyncio.Lock()
    async with CHAT_LOCKS[chat_id]:
        user_input = await _handle_attachments(update.message)
        conv_logger.info(f"USER [{chat_id}]: {user_input}")
        await _process_request(chat_id, user_input, update.message)


async def _process_request(chat_id, user_input, origin_message=None):
    STOP_SIGNAL[chat_id] = False
    
    async def send_msg(text, silent=True, noisy=False):
        """Robust permanent delivery with retry."""
        if not text or not text.strip(): return None
        disable_notif = silent if not noisy else False
        for attempt in range(3):
            try:
                if origin_message: return await origin_message.reply_text(text, parse_mode="HTML", disable_notification=disable_notif)
                return await GLOBAL_APPLICATION.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML", disable_notification=disable_notif)
            except Exception as e:
                if attempt == 2:
                    logger.error(f"TG Send Error: {e}")
                    # Fallback to plain text if HTML fails
                    plain = re.sub(r'<[^>]+>', '', text)
                    if origin_message: return await origin_message.reply_text(plain[:4000])
                    return await GLOBAL_APPLICATION.bot.send_message(chat_id=chat_id, text=plain[:4000])
                await asyncio.sleep(1)

    # Thinking cursor
    cursor_msg = await send_msg("🤔 <b>Thinking...</b>", silent=True)

    full_response = ""
    sent_upto = 0
    active_tools = {} # tid -> {header, output}
    final_stats = None

    async def rotate_cursor():
        """Refreshes the bottom cursor."""
        nonlocal cursor_msg
        old = cursor_msg
        cursor_msg = await send_msg("🤔 <b>Thinking...</b>", silent=True)
        try: await old.delete()
        except: pass

    async def flush_text():
        """Sends unsent part of full_response."""
        nonlocal sent_upto
        to_send = full_response[sent_upto:].strip()
        if to_send:
            await send_msg(_format_html_response(to_send))
            sent_upto = len(full_response)
            await rotate_cursor()

    async def handle_images(text):
        matches = list(re.finditer(r"\[SEND_IMAGE:\s*(.*?)\]", text, re.IGNORECASE))
        for m in matches:
            path = m.group(1).strip()
            if not os.path.exists(path): path = os.path.join(WORKSPACE_DIR, os.path.basename(path))
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as p:
                        if origin_message: await origin_message.reply_photo(p, caption=f"📸 {os.path.basename(path)}", disable_notification=True)
                        else: await GLOBAL_APPLICATION.bot.send_photo(chat_id=chat_id, photo=p, caption=f"📸 {os.path.basename(path)}", disable_notification=True)
                    await rotate_cursor()
                except: pass

    async def callback(e_type, e_data):
        nonlocal full_response, cursor_msg, active_tools, final_stats
        if STOP_SIGNAL.get(chat_id): return

        if e_type == "message" and e_data.get("role") == "assistant":
            content = e_data.get("content", "")
            if content: full_response += content

        elif e_type == "tool_use":
            await flush_text()
            clear_live_buffer(chat_id)
            
            name = e_data.get("tool_name") or e_data.get("name") or "tool"
            tid = e_data.get("tool_id") or e_data.get("id") or f"idx_{len(active_tools)}"
            params = e_data.get("parameters") or e_data.get("args") or {}
            
            if "HISTORY.md" in str(params):
                active_tools[tid] = {"silent": True}
                return

            nick, icon = TOOL_MAPPING.get(name, name), TOOL_ICONS.get(name, "🛠️")
            p_val = params.get("command") or params.get("file_path") or params.get("dir_path") or ""
            p_disp = html.escape((str(p_val)[:150] + "...") if len(str(p_val)) > 150 else str(p_val))
            header = f"{icon} {nick} <code>{p_disp}</code>"
            
            await send_msg(header)
            await rotate_cursor()
            active_tools[tid] = {"header": header, "output": "", "silent": False}
            conv_logger.info(f"TOOL [{chat_id}] {name}: {p_val}")

        elif e_type == "raw_stdout":
            content = e_data.get("content", "")
            if not active_tools: return
            last_tid = list(active_tools.keys())[-1]
            if not active_tools[last_tid].get("silent"):
                active_tools[last_tid]["output"] += content + "\n"

        elif e_type == "tool_result":
            tid = e_data.get("tool_id") or e_data.get("id")
            t_info = active_tools.get(tid)
            if not t_info and active_tools: t_info = list(active_tools.values())[-1]
            if not t_info or t_info.get("silent"): return
            
            raw_out = e_data.get("output")
            output = str(raw_out).strip() if (raw_out is not None and str(raw_out).strip()) else t_info.get("output", "").strip()
            
            conv_logger.info(f"RESULT [{chat_id}] {tid}: {output[:200]}...")
            await handle_images(output)
            clean_output = re.sub(r"\[SEND_IMAGE:.*?\]", "", output, flags=re.IGNORECASE)
            
            if clean_output:
                safe = html.escape("\n".join(clean_output.split("\n")[-50:]))
                await send_msg(f"<pre>{safe}</pre>")
                await rotate_cursor()

        elif e_type == "result":
            final_stats = e_data.get("stats")

        elif e_type == "error":
            severity, msg = e_data.get("severity", "error"), e_data.get("message", "Unknown error")
            icon = "⚠️" if severity == "warning" else "❌"
            await send_msg(f"{icon} <b>Quota/API {severity.capitalize()}:</b>\n<code>{html.escape(msg)}</code>")
            await rotate_cursor()

    # Run loop
    exit_code, stats_from_stream = await call_gemini_stream(user_input, chat_id, callback)
    final_stats = final_stats or stats_from_stream

    # Final text flush
    # await flush_text()
    
    conv_logger.info(f"AGENT [{chat_id}]: {full_response}")
    if "[VOICE_TRANSCRIPTION]" in user_input and full_response:
        summary = await get_short_summary(full_response)
        await _reply_with_voice(chat_id, summary, reply_to_message_id=origin_message.message_id if origin_message else None)

    try: await cursor_msg.delete()
    except: pass
    
    if final_stats:
        tokens, dur = final_stats.get("total_tokens", 0), final_stats.get("duration_ms", 0) / 1000
        if tokens > 0:
            txt = f"<code>💎 {tokens} tokens | ⏱️ {dur:.1f}s</code>"
            await send_msg(txt, noisy=True)
            conv_logger.info(f"STATS [{chat_id}]: {tokens} tokens")


async def stop_cmd(update, context):
    if is_not_user(update): return
    STOP_SIGNAL[update.effective_chat.id] = True
    for p in ACTIVE_SUBPROCESSES.get(update.effective_chat.id, []):
        try: p.kill()
        except: pass
    await update.message.reply_text("🛑 Stopped.", parse_mode="HTML")

async def status_cmd(update, context):
    if is_not_user(update): return
    procs = ACTIVE_SUBPROCESSES.get(update.effective_chat.id, [])
    if not procs: return await update.message.reply_text("😴 Idle")
    s = f"🚀 <b>Agent Status</b>\n━━━━━━━━━━━━━━━\n👥 <b>Active:</b> {len(procs)}\n"
    if update.effective_chat.id in LIVE_BUFFERS:
        logs = html.escape("\n".join(LIVE_BUFFERS[update.effective_chat.id]))
        s += f"\n📝 <b>Logs:</b>\n<pre>{logs[-1000:]}</pre>"
    await update.message.reply_text(s, parse_mode="HTML")

async def tasks_cmd(update, context):
    if is_not_user(update): return
    if not os.path.exists(TASKS_FILE): return await update.message.reply_text("Empty")
    import json
    with open(TASKS_FILE, 'r') as f: tasks = json.load(f)
    kb = [[InlineKeyboardButton(f"❌ {t['name']}", callback_data=f"del_task_{i}")] for i, t in enumerate(tasks)]
    await update.message.reply_text("📋 <b>Tasks:</b>", parse_mode="HTML", reply_markup=InlineKeyboardMarkup(kb))

async def task_callback(update, context):
    q = update.callback_query
    await q.answer()
    if q.data.startswith("del_task_"):
        idx = int(q.data.split("_")[-1])
        import json
        with open(TASKS_FILE, 'r') as f: tasks = json.load(f)
        if 0 <= idx < len(tasks):
            t = tasks.pop(idx); json.dump(tasks, open(TASKS_FILE, 'w'), indent=4)
            await q.edit_message_text(f"✅ Deleted: {t['name']}")

async def update_cmd(update, context):
    if is_not_user(update): return
    msg = await update.message.reply_text("🔄 Updating...")
    p = await asyncio.create_subprocess_shell("npm install -g @google/gemini-cli@nightly")
    await p.wait()
    await msg.edit_text("✅ Updated! Restarting...")
    os.execv(sys.executable, ['python'] + sys.argv)

async def post_init(app):
    global GLOBAL_APPLICATION
    GLOBAL_APPLICATION = app
    commands = [BotCommand("start", "Menu principal"), BotCommand("clear", "Nouvelle session"), BotCommand("chat", "Sessions"), BotCommand("tasks", "Tasks"), BotCommand("status", "Status"), BotCommand("stop", "Stop")]
    await app.bot.set_my_commands(commands)
    if hasattr(app, "scheduler_func"): asyncio.create_task(app.scheduler_func(trigger_scheduled_task))

async def clear_cmd(update, context):
    if is_not_user(update): return
    set_current_session(update.effective_chat.id, "fresh_session")
    await update.message.reply_text("✨ <b>Nouvelle session démarrée !</b>", parse_mode="HTML")

async def chat_cmd(update, context):
    if is_not_user(update): return
    try:
        res = subprocess.check_output(["gemini", "--list-sessions"], text=True, stderr=subprocess.STDOUT)
        await update.message.reply_text(f"📋 <b>Sessions :</b>\n<pre>{html.escape(res)}</pre>", parse_mode="HTML")
    except: await update.message.reply_text("Aucune session trouvée.")

async def error_handler(update, context): logger.error(f"TG Error: {context.error}")

def run_bot(scheduler_func=None):
    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()
    app.scheduler_func = scheduler_func
    menu = ReplyKeyboardMarkup([['/clear', '/chat'], ['/tasks', '/status'], ['/stop']], resize_keyboard=True)
    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("<b>Ready!</b>", parse_mode="HTML", reply_markup=menu)))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("chat", chat_cmd))
    app.add_handler(CommandHandler("tasks", tasks_cmd))
    app.add_handler(CommandHandler("update", update_cmd))
    app.add_handler(CallbackQueryHandler(task_callback))
    app.add_handler(MessageHandler(filters.ALL & (~filters.COMMAND), handle_message, block=False))
    app.add_error_handler(error_handler)
    app.run_polling()
