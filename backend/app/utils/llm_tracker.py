"""Provider-neutral request timing and token-estimate logging."""

import contextvars
import datetime
import inspect
import os
import time


request_stats = contextvars.ContextVar("request_stats", default=None)
llm_calls = []


def _caller() -> str:
    for frame in inspect.stack():
        filename = frame.filename
        if "llm_tracker.py" in filename or "inspect.py" in filename:
            continue
        return f"{frame.function} ({os.path.basename(filename)}:{frame.lineno})"
    return "Unknown"


def _prompt(contents) -> str:
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        return "\n".join(str(item) for item in contents)
    return str(contents)


def instrument_client(client):
    if getattr(client.models, "_is_instrumented", False):
        return client

    original = client.models.generate_content
    original_stream = client.models.generate_content_stream

    def wrapped(model, contents, **kwargs):
        caller = _caller()
        prompt = _prompt(contents)
        started = time.time()
        print(f"\n[LLM CALL START] {caller} | model={model} | input_chars={len(prompt)}", flush=True)
        response = original(model=model, contents=contents, **kwargs)
        output = getattr(response, "text", "") or ""
        total = round((len(prompt) + len(output)) / 4)
        duration = round((time.time() - started) * 1000)
        print(f"[LLM CALL END] {caller} | output_chars={len(output)} | estimated_tokens={total} | duration_ms={duration}", flush=True)
        stats = request_stats.get()
        if stats is not None:
            stats.setdefault("calls", []).append({"function": caller, "total_tokens": total, "duration_ms": duration})
        return response

    def wrapped_stream(model, contents, **kwargs):
        caller = _caller()
        prompt = _prompt(contents)
        started = time.time()
        print(f"\n[LLM STREAM START] {caller} | model={model} | input_chars={len(prompt)}", flush=True)
        stream = original_stream(model=model, contents=contents, **kwargs)

        def iterator():
            pieces = []
            try:
                for chunk in stream:
                    if getattr(chunk, "text", ""):
                        pieces.append(chunk.text)
                    yield chunk
            finally:
                total = round((len(prompt) + len("".join(pieces))) / 4)
                duration = round((time.time() - started) * 1000)
                print(f"[LLM STREAM END] {caller} | estimated_tokens={total} | duration_ms={duration}", flush=True)
                stats = request_stats.get()
                if stats is not None:
                    stats.setdefault("calls", []).append({"function": caller, "total_tokens": total, "duration_ms": duration})
        return iterator()

    client.models.generate_content = wrapped
    client.models.generate_content_stream = wrapped_stream
    client.models._is_instrumented = True
    return client


def print_query_performance_report():
    stats = request_stats.get()
    if not stats:
        return
    calls = stats.get("calls", [])
    total = sum(call.get("total_tokens", 0) for call in calls)
    elapsed = round((time.time() - stats.get("start_time", time.time())) * 1000)
    print(f"\nQUERY PERFORMANCE REPORT\nQuestion: {stats.get('query', 'Unknown')}\nTotal Estimated Tokens: {total}\nTotal LLM Calls: {len(calls)}\nTotal Execution Time: {elapsed} ms\n", flush=True)
