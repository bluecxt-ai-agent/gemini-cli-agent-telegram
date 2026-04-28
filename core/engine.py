"""
Gemini Engine - Core reasoning loop with real-time logs buffering.
"""

import json
import asyncio
import os
from collections import deque
from .memory import get_current_session, set_current_session
from .logger import logger

ACTIVE_SUBPROCESSES = {}
STOP_SIGNAL = {}
LIVE_BUFFERS = {}  # chat_id: deque of strings
CURRENT_COMMANDS = {}  # chat_id: str

SYSTEM_INSTRUCTIONS = """
You are the Gemini CLI Agent, a high-performance autonomous engineering assistant.
You operate within a Docker container. 

HIERARCHY & PERSISTENCE RULES:
1. USE `/app/workspace` for ALL long-term data, source code, and project files.
   This folder is persistent and survives container resets (docker-compose down).
2. USE `/app/tmp` for temporary working notes or transient data.
3. You have root access inside this container. You can install packages (pip, apt)
   if needed, but they will be lost on container reset.
4. ALWAYS prioritize working within `/app/workspace` for important project elements.

MANDATORY FORMATTING:
- ALL internal reasoning must be inside <thinking> tags. NEVER use [Thought: ...] format.
- Use <b>bold</b>, <i>italic</i>, <code>inline code</code> and <pre>blocks of code</pre> for Telegram formatting.
- Speak French (Français) to the user, but keep internal thoughts and code in English.
"""


def register_process(chat_id, proc):
    if chat_id not in ACTIVE_SUBPROCESSES:
        ACTIVE_SUBPROCESSES[chat_id] = []
    ACTIVE_SUBPROCESSES[chat_id].append(proc)


def unregister_process(chat_id, proc):
    if chat_id in ACTIVE_SUBPROCESSES:
        try:
            ACTIVE_SUBPROCESSES[chat_id].remove(proc)
        except ValueError:
            pass


async def _read_stream(stream, chat_id, is_stderr=False):
    """ Reads a stream line by line and updates the live buffer. """
    if chat_id not in LIVE_BUFFERS:
        LIVE_BUFFERS[chat_id] = deque(maxlen=20)
        
    while True:
        line = await stream.readline()
        if not line:
            break
        
        text = line.decode().strip()
        if text:
            # If it's stderr or non-JSON stdout, add to live buffer
            if is_stderr or not (text.startswith("{") and text.endswith("}")):
                LIVE_BUFFERS[chat_id].append(text)
                if is_stderr:
                    logger.warning(f"Gemini Stderr [{chat_id}]: {text}")


async def call_gemini_stream(prompt, chat_id, callback):
    """
    Calls Gemini CLI and processes JSON events while buffering raw output.
    """
    session_id = get_current_session(chat_id) or "latest"
    LIVE_BUFFERS[chat_id] = deque(maxlen=20)
    CURRENT_COMMANDS.pop(chat_id, None)

    format_reminder = (
        "\n\nIMPORTANT FORMATTING RULES:\n"
        "1. ALL internal monologue, technical analysis, and tool reasoning MUST be in English and inside <thinking> tags.\n"
        "2. Any text OUTSIDE <thinking> tags MUST be in French (Français) for the user.\n"
        "3. NEVER use [Thought: ...] or markdown headers for reasoning.\n"
        "4. To send an image or screenshot to the user, use the syntax: [SEND_IMAGE: /path/to/image.png] outside thinking tags."
    )
    full_prompt = f"{SYSTEM_INSTRUCTIONS}{format_reminder}\n\nUSER REQUEST: {prompt}"

    args = [
        "gemini", "--prompt", "-",
        "--output-format", "stream-json",
        "--approval-mode", "yolo"
    ]

    if session_id and session_id != "fresh_session":
        args.extend(["--resume", session_id])

    env = os.environ.copy()
    env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )
    register_process(chat_id, proc)

    # Start background task to read stderr in real-time
    stderr_task = asyncio.create_task(_read_stream(proc.stderr, chat_id, is_stderr=True))

    try:
        proc.stdin.write(full_prompt.encode())
        await proc.stdin.drain()
        proc.stdin.close()

        while True:
            line = await proc.stdout.readline()
            if not line:
                break

            raw_line = line.decode().strip()
            if not raw_line:
                continue

            if not raw_line.startswith("{"):
                LIVE_BUFFERS[chat_id].append(raw_line)
                continue

            try:
                event = json.loads(raw_line)
                e_type = event.get("type")

                if e_type == "init":
                    sid = event.get("session_id")
                    if sid:
                        set_current_session(chat_id, sid)

                await callback(e_type, event)

            except json.JSONDecodeError:
                LIVE_BUFFERS[chat_id].append(raw_line)

        await proc.wait()
        await stderr_task
        return proc.returncode, ""

    except Exception as e:
        logger.exception("ENGINE ERROR")
        return -1, str(e)
    finally:
        unregister_process(chat_id, proc)
        if proc.returncode is None:
            try: proc.kill()
            except: pass
