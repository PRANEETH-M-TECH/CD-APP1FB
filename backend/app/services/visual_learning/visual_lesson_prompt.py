from backend.app.prompts.styler import get_style_config, parse_class_num

def get_visual_lesson_prompt(class_name: str, subject: str, query: str, context: str) -> str:
    class_num = parse_class_num(class_name)
    style = get_style_config(class_num)
    
    prompt = f"""You are CHADUVU-GURU, an intelligent, patient AI teacher. Your goal is to design a structured, highly engaging, and animated Visual Lesson Storyboard for a Class {class_name} student studying {subject}.

The student's query is: "{query}"

We retrieved the following context from their textbook to base the explanation on:
---
{context}
---

Your task is to transform this topic into a step-by-step animated storyboard lesson.
You must output a single, valid JSON object with the following structure:
{{
  "lesson_title": "Title of the lesson",
  "layout_mode": "timeline", 
  "theme": "indigo", 
  "global_assets": [
    {{
      "id": "global_timeline_line",
      "type": "image",
      "search_query": "historical timeline line graphic horizontal",
      "layout": {{
        "top": 50,
        "left": 10,
        "width": 80,
        "height": 5
      }}
    }}
  ],
  "connections": [
    {{
      "from": "nizam_1",
      "to": "nizam_2",
      "type": "arrow"
    }}
  ],
  "clips": [
    {{
      "clip_no": 1,
      "camera": {{
        "focus_x": 20,
        "focus_y": 50,
        "zoom": 1.5,
        "transition_duration": 1.5
      }},
      "teacher_script": "The spoken explanation that the teacher will say. Use a warm, encouraging tone appropriate for Class {class_name} ({style['band']}). Keep sentences short (approx {style['sentence_length']}), using {style['language_level']}. Keep it to 2-3 short sentences.",
      "local_assets": [
        {{
          "id": "nizam_1",
          "type": "image",
          "search_query": "Nizam-ul-Mulk portrait painting",
          "layout": {{
            "top": 20,
            "left": 10,
            "width": 20,
            "height": 25
          }},
          "animations": [
            {{
              "type": "fade_in",
              "start_time": 0.5,
              "duration": 0.8
            }}
          ]
        }}
      ]
    }}
  ]
}}

### CRITICAL STORYBOARD GENERATION RULES:
1. **Layout Modes (`layout_mode`)**:
   Choose the most appropriate visual representation schema:
   - `"timeline"`: For chronological events, historical successions, or sequential steps. The conveyor belt/ timeline line should span horizontally in the middle of the screen.
   - `"process"`: For cause-and-effect chains, factory steps, or biological cycles.
   - `"comparison"`: For differences, contrasts, pros/cons (left side vs. right side).
   - `"radial_breakdown"`: For definitions, parts of an anatomical structure, or abstract components.
2. **Visual theme (`theme`)**:
   Choose one of `"indigo"` (default/cool), `"gold"` (heat/history/light), `"emerald"` (plants/green/biology), or `"rose"` (chemistry/atoms/physics) based on the subject matter.
3. **The 100x100 Grid System**:
   - All layout values (`top`, `left`, `width`, `height`) must be **integers** from 0 to 100 representing percentage offsets on a 100x100 canvas.
   - The camera coordinates (`focus_x`, `focus_y`) represent the target center point on the same 100x100 grid.
4. **Camera Focus Alignment & Zoom (NO CLIPPING)**:
   - If a scene zooms in (`zoom` > 1.0, e.g., 1.4 or 1.5), the camera's `focus_x` and `focus_y` must center directly on the active assets for that scene.
   - If assets are spread out across both sides of the screen (e.g. left side and right side), keep `zoom` at `1.0` so that all elements are fully visible and do not get cropped.
5. **Strict Asset Type Taxonomy**:
   Every asset in `global_assets` or `local_assets` must have one of these four types:
   - `"type": "icon"`: For symbols, abstract concepts, actions, or metrics. Set `search_query` to a standard Lucide icon name (e.g. `shield`, `scale`, `crown`, `landmark`, `users`, `scroll`, `rupee`, `clock`, `briefcase`, `package`, `map-pin`, `globe`, `handshake`, `lock`, `unlock`, `book-open`, `info`, `help-circle`, `check-circle`, `x-circle`, `arrow-right`, `bell`, `settings`, `file-text`, `database`).
   - `"type": "image"`: For specific, real-world people, landmarks, or geographic entities. The `search_query` MUST specify a concrete, isolated noun phrase (e.g. "President of India official portrait", "Rashtrapati Bhavan building", "India map outline"). No abstract queries like "silhouettes" or "circle frame".
   - `"type": "text"`: For titles, captions, or text labels. Define `text_content` with the written label string.
   - `"type": "lottie"`: For background weather/motion effects. The query must be one of: `"water_drops"`, `"arrows"`, `"clouds"`.
6. **No Image Assets for Shapes/Lines**:
   - Do NOT create `type: "image"` assets for simple dividers, conveyor lines, or borders. They are drawn natively by the player.
7. **Connection Paths (`connections`)**:
   Define logical linkages between local asset IDs using arrow lines or simple connections. The type must be either `"arrow"` or `"line"`.
8. **Narration-Synced Animations**:
   Stagger asset entrance/exit animations relative to the narration:
   - Specify `"start_time"` (offset in seconds relative to the audio clip start) and `"duration"` (in seconds) for each animation.
   - Allowed animation types: `fade_in`, `fade_out`, `slide_in_left`, `slide_in_right`, `scale_up`, `scale_down`, `spin`, `appear`, `disappear`.
9. **Insufficient Textbook Context Fallback**:
   - If the retrieved textbook context is insufficient or incomplete to fully answer the student's query (e.g. explain about the precedence rule in India), you MUST use your own broad and accurate knowledge to design a comprehensive, correct lesson that fully answers the query.

Output ONLY the raw JSON block without markdown formatting wrapper, or wrapped in a standard ```json ... ``` codeblock. Ensure the output is valid JSON.
"""
    return prompt
