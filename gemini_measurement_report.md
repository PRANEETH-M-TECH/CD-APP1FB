# Gemini API Calls Measurement Report

This report shows the Gemini API usage metrics for a single user query execution under the **CHADUVU-GURU** backend architecture.

## 📊 Summary Metrics

| Metric | Value |
| :--- | :--- |
| **Total Gemini Calls** | 3 |
| **Largest Prompt Size (chars)** | 57,104 chars |
| **Total Estimated Tokens** | 16,808 tokens |

## 📞 Call Inventory & Tokens Per Call

Below is the chronological sequence of Gemini API calls triggered by the query: **"What is photosynthesis?"**

| Call # | Function (File) | Call Type | Prompt Size (chars) | Estimated Tokens | Timestamp |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `reformulate_with_llm (answer_service.py:351)` | Unary | 57,104 | 14,276 | `2026-05-31 14:54:59.184` |
| 2 | `event_generator (chat.py:490)` | Streaming | 7,926 | 1,982 | `2026-05-31 14:55:18.830` |
| 3 | `generate_smart_followups (answer_service.py:521)` | Unary | 2,199 | 550 | `2026-05-31 14:55:22.403` |

## 🔬 Detailed Breakdown of Calls

### Call 1: `reformulate_with_llm (answer_service.py:351)`
* **Type**: unary
* **Prompt Size**: 57,104 characters
* **Estimated Tokens**: 14,276 tokens
* **Timestamp**: 2026-05-31 14:54:59.184

---
### Call 2: `event_generator (chat.py:490)`
* **Type**: streaming
* **Prompt Size**: 7,926 characters
* **Estimated Tokens**: 1,982 tokens
* **Timestamp**: 2026-05-31 14:55:18.830

---
### Call 3: `generate_smart_followups (answer_service.py:521)`
* **Type**: unary
* **Prompt Size**: 2,199 characters
* **Estimated Tokens**: 550 tokens
* **Timestamp**: 2026-05-31 14:55:22.403

---
