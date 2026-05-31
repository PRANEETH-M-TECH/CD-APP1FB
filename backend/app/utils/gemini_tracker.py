import inspect
import os
import datetime
from typing import List, Dict

# Global list to store Gemini call details during this run
gemini_calls: List[Dict] = []

def get_caller_function() -> str:
    """
    Traverses the call stack to find the first function outside of the tracker/SDK
    that initiated the Gemini API call.
    """
    stack = inspect.stack()
    for frame in stack:
        filename = frame.filename
        # Skip our own tracker frames, python inspect frames, and site-packages/SDK internals
        if "gemini_tracker.py" in filename or "inspect.py" in filename or "google\\genai" in filename.lower() or "google/genai" in filename.lower():
            continue
        func_name = frame.function
        basename = os.path.basename(filename)
        return f"{func_name} ({basename}:{frame.lineno})"
    return "Unknown"

def get_prompt_text(contents) -> str:
    """
    Extracts plain text string from Gemini contents argument.
    Supports str, list of content objects, etc.
    """
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        text_parts = []
        for part in contents:
            if isinstance(part, str):
                text_parts.append(part)
            elif hasattr(part, "text"):
                text_parts.append(part.text)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            else:
                text_parts.append(str(part))
        return "\n".join(text_parts)
    return str(contents)

def instrument_client(client):
    """
    Monkeypatches gemini_client.models.generate_content and generate_content_stream
    to measure, log, and print detailed info for every call.
    """
    if hasattr(client.models, "_is_instrumented"):
        return client

    original_generate_content = client.models.generate_content
    original_generate_content_stream = client.models.generate_content_stream

    def wrapped_generate_content(model, contents, **kwargs):
        caller = get_caller_function()
        prompt_str = get_prompt_text(contents)
        prompt_size = len(prompt_str)
        estimated_tokens = round(prompt_size / 4)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        print("\n[CALL START]")
        print(f"Function: {caller}")
        print(f"Prompt Size (chars): {prompt_size}")
        print(f"Estimated Tokens: {estimated_tokens}")
        print(f"Timestamp: {timestamp}")
        print("[CALL END]\n", flush=True)

        gemini_calls.append({
            "function": caller,
            "prompt_size": prompt_size,
            "estimated_tokens": estimated_tokens,
            "timestamp": timestamp,
            "type": "unary"
        })

        return original_generate_content(model=model, contents=contents, **kwargs)

    def wrapped_generate_content_stream(model, contents, **kwargs):
        caller = get_caller_function()
        prompt_str = get_prompt_text(contents)
        prompt_size = len(prompt_str)
        estimated_tokens = round(prompt_size / 4)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        print("\n[CALL START]")
        print(f"Function: {caller}")
        print(f"Prompt Size (chars): {prompt_size}")
        print(f"Estimated Tokens: {estimated_tokens}")
        print(f"Timestamp: {timestamp}")
        print("[CALL END]\n", flush=True)

        gemini_calls.append({
            "function": caller,
            "prompt_size": prompt_size,
            "estimated_tokens": estimated_tokens,
            "timestamp": timestamp,
            "type": "streaming"
        })

        return original_generate_content_stream(model=model, contents=contents, **kwargs)

    client.models.generate_content = wrapped_generate_content
    client.models.generate_content_stream = wrapped_generate_content_stream
    client.models._is_instrumented = True
    return client
