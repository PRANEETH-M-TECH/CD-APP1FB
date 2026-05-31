# Gemini API Calls Measurement Report

This report shows the Gemini API usage metrics for a single user query execution under the **CHADUVU-GURU** backend architecture.

## 📊 Summary Metrics

| Metric | Value |
| :--- | :--- |
| **Total Gemini Calls** | 3 |
| **Largest Prompt Size (chars)** | 2,427 chars |
| **Total Estimated Tokens** | 1,473 tokens |

## 📞 Call Inventory & Tokens Per Call

Below is the chronological sequence of Gemini API calls triggered by the query: **"What is photosynthesis?"**

| Call # | Function (File) | Call Type | Prompt Size (chars) | Estimated Tokens | Timestamp |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `reformulate_with_llm (answer_service.py:352)` | Unary | 1,516 | 379 | `2026-05-31 16:30:14.286` |
| 2 | `event_generator (chat.py:497)` | Streaming | 1,949 | 487 | `2026-05-31 16:30:30.580` |
| 3 | `generate_smart_followups (answer_service.py:522)` | Unary | 2,427 | 607 | `2026-05-31 16:30:40.386` |

## 🔬 Detailed Breakdown of Calls

### Call 1: `reformulate_with_llm (answer_service.py:352)`
* **Type**: unary
* **Prompt Size**: 1,516 characters
* **Estimated Tokens**: 379 tokens
* **Timestamp**: 2026-05-31 16:30:14.286

---
### Call 2: `event_generator (chat.py:497)`
* **Type**: streaming
* **Prompt Size**: 1,949 characters
* **Estimated Tokens**: 487 tokens
* **Timestamp**: 2026-05-31 16:30:30.580

---
### Call 3: `generate_smart_followups (answer_service.py:522)`
* **Type**: unary
* **Prompt Size**: 2,427 characters
* **Estimated Tokens**: 607 tokens
* **Timestamp**: 2026-05-31 16:30:40.386

---
