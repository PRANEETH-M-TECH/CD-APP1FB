import inspect
import os
import datetime
import time
import contextvars
from typing import List, Dict

# Global list to store Gemini call details during this run
gemini_calls: List[Dict] = []

# Thread-safe ContextVar to trace calls made in a single web request context
request_stats = contextvars.ContextVar("request_stats", default=None)

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

def print_query_performance_report():
    """
    Retrieves the accumulated session logs and outputs the consolidated
    QUERY PERFORMANCE REPORT to standard output.
    """
    stats = request_stats.get()
    if not stats:
        return
    
    query = stats.get("query", "Unknown Query")
    calls = stats.get("calls", [])
    
    reformulation_tokens = 0
    answer_tokens = 0
    followup_tokens = 0
    total_tokens = 0
    total_calls = len(calls)
    
    for call in calls:
        func = call["function"]
        tok = call["total_tokens"]
        
        # Categorize calls by function name in call stack
        if "reformulate" in func.lower():
            reformulation_tokens += tok
        elif "event_generator" in func.lower() or "generate_conversational_answer" in func.lower() or "generate_answer" in func.lower():
            answer_tokens += tok
        elif "followup" in func.lower():
            followup_tokens += tok
        elif "determine_next_action" in func.lower():
            # Intent classifier LLM routing (Tier 5) is also part of reformulation/classification overhead
            reformulation_tokens += tok
        else:
            # Fallback
            answer_tokens += tok
            
        total_tokens += tok
        
    elapsed_time_ms = round((time.time() - stats["start_time"]) * 1000)
    
    report = f"""
====================================

QUERY PERFORMANCE REPORT

Question: {query}

Reformulation Tokens:
{reformulation_tokens}

Answer Generation Tokens:
{answer_tokens}

Followup Tokens:
{followup_tokens}

Total Estimated Tokens:
{total_tokens}

Total Gemini Calls:
{total_calls}

Total Execution Time:
{elapsed_time_ms} ms

====================================
"""
    print(report, flush=True)

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
        input_tokens = round(prompt_size / 4)
        start_time = time.time()
        start_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Existing logs preserved
        print("\n[CALL START]")
        print(f"Function: {caller}")
        print(f"Prompt Size (chars): {prompt_size}")
        print(f"Estimated Tokens: {input_tokens}")
        print(f"Timestamp: {start_timestamp}")
        print("[CALL END]\n", flush=True)

        # Phase 3 Log: LOG START
        print(f"\nLOG START\nFunction Name: {caller}\nTimestamp: {start_timestamp}\nPrompt Characters: {prompt_size}\nEstimated Input Tokens: {input_tokens}\nLOG END\n", flush=True)

        gemini_calls.append({
            "function": caller,
            "prompt_size": prompt_size,
            "estimated_tokens": input_tokens,
            "timestamp": start_timestamp,
            "type": "unary"
        })

        # Execute call
        response = original_generate_content(model=model, contents=contents, **kwargs)

        end_time = time.time()
        duration_ms = round((end_time - start_time) * 1000)
        end_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        response_text = ""
        try:
            if response and hasattr(response, "text"):
                response_text = response.text
        except Exception:
            pass

        output_tokens = round(len(response_text) / 4)
        total_tokens = input_tokens + output_tokens

        # Phase 3 Log: LOG COMPLETE / After Gemini returns
        print(f"\nFunction Name: {caller}\nTimestamp: {end_timestamp}\nEstimated Output Tokens: {output_tokens}\nEstimated Total Tokens: {total_tokens}\nExecution Duration (ms): {duration_ms}\n", flush=True)

        # Record to ContextVar tracking
        stats = request_stats.get()
        if stats is not None:
            stats["calls"].append({
                "function": caller,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "duration_ms": duration_ms
            })

        return response

    def wrapped_generate_content_stream(model, contents, **kwargs):
        caller = get_caller_function()
        prompt_str = get_prompt_text(contents)
        prompt_size = len(prompt_str)
        input_tokens = round(prompt_size / 4)
        start_time = time.time()
        start_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Existing logs preserved
        print("\n[CALL START]")
        print(f"Function: {caller}")
        print(f"Prompt Size (chars): {prompt_size}")
        print(f"Estimated Tokens: {input_tokens}")
        print(f"Timestamp: {start_timestamp}")
        print("[CALL END]\n", flush=True)

        # Phase 3 Log: LOG START
        print(f"\nLOG START\nFunction Name: {caller}\nTimestamp: {start_timestamp}\nPrompt Characters: {prompt_size}\nEstimated Input Tokens: {input_tokens}\nLOG END\n", flush=True)

        gemini_calls.append({
            "function": caller,
            "prompt_size": prompt_size,
            "estimated_tokens": input_tokens,
            "timestamp": start_timestamp,
            "type": "streaming"
        })

        # Execute call stream
        response_stream = original_generate_content_stream(model=model, contents=contents, **kwargs)

        def generator_wrapper():
            full_output = []
            try:
                for chunk in response_stream:
                    try:
                        if chunk.text:
                            full_output.append(chunk.text)
                    except Exception:
                        pass
                    yield chunk
            finally:
                # On stream completion/close
                end_time = time.time()
                duration_ms = round((end_time - start_time) * 1000)
                end_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                output_text = "".join(full_output)
                output_tokens = round(len(output_text) / 4)
                total_tokens = input_tokens + output_tokens

                # Phase 3 Log: LOG COMPLETE / After Gemini returns
                print(f"\nFunction Name: {caller}\nTimestamp: {end_timestamp}\nEstimated Output Tokens: {output_tokens}\nEstimated Total Tokens: {total_tokens}\nExecution Duration (ms): {duration_ms}\n", flush=True)

                # Record to ContextVar tracking
                stats = request_stats.get()
                if stats is not None:
                    stats["calls"].append({
                        "function": caller,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "duration_ms": duration_ms
                    })

        return generator_wrapper()

    client.models.generate_content = wrapped_generate_content
    client.models.generate_content_stream = wrapped_generate_content_stream
    client.models._is_instrumented = True
    return client
