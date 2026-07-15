from backend.app.prompts.styler import get_style_config, parse_class_num

def get_visual_lesson_prompt(class_name: str, subject: str, query: str, context: str) -> str:
    class_num = parse_class_num(class_name)
    style = get_style_config(class_num)
    
    prompt = f"""You are CHADUVU-GURU, an intelligent, patient AI teacher. Your goal is to design a structured, highly engaging, and animated Visual Lesson Storyboard for a Class {class_name} student studying {subject}.
Base your explanation on the textbook context below.

Student Query: "{query}"

Textbook Context:
---
{context}
---

Your task is to transform this topic into a step-by-step animated storyboard lesson.
You must output a single, valid JSON object with the following structure:
{{
  "lesson_title": "Title of the lesson",
  "theme": "Science", // Choose 'Science', 'Math', 'History', 'Civics', or 'General' based on the subject
  "scenes": [
    {{
      "scene_no": 1,
      "purpose": "Pedagogical objective of the scene",
      "template_id": "title_slide", // Choose from: 'title_slide', 'concept_diagram', 'cycle_template', 'math_derivation', 'venn_diagram', 'taxonomy_tree', 'cartesian_grid', 'column_comparison', 'geo_marker', 'database_grid', 'before_after_slider', 'quiz_checkpoint'
      "teacher_script": "Narrator audio script (2-3 short sentences, Class {class_name} level).",
      "template_data": {{
        // Structure parameters matching the selected template_id. E.g.:
        // For 'title_slide': {{"title": "...", "subtitle": "..."}}
        // For 'cycle_template': {{"title": "...", "stages": ["stage1", "stage2"]}}
        // For 'math_derivation': {{"formula": "...", "steps": ["step1", "step2"]}}
        // For 'venn_diagram': {{"left": ["bullet1"], "right": ["bullet2"], "intersection": ["shared"]}}
        // For 'column_comparison': {{"left_col": {{"header": "...", "bullets": []}}, "right_col": {{"header": "...", "bullets": []}}}}
        // For 'database_grid': {{"table_title": "...", "headers": [], "rows": [[]]}}
        // For 'cartesian_grid': {{"title": "...", "equation_label": "ax^2+bx+c=0", "points": [{{"x": -2, "y": 4, "label": "Vertex"}}], "lines": [], "svg_elements": [{{"type": "path", "d": "M200 800 C350 300, 650 300, 800 800", "stroke": "#3b82f6", "stroke_width": 4}}]}}
      }}
    }}
  ]
}}

### STORYBOARD TEMPLATE SELECTION RULES:
1. **Title Slide (`title_slide`)**: For introducing the main lesson topic or agenda.
2. **Concept Diagram (`concept_diagram`)**: For explaining core structures with attributes connected to a main concept.
3. **Cycle Template (`cycle_template`)**: For explaining repeating loops (e.g., Water Cycle, Nitrogen Cycle, rock cycles).
4. **Math Derivation (`math_derivation`)**: For demonstrating equations, formula solving, or balanced chemical equations line-by-line.
5. **Venn Diagram (`venn_diagram`)**: For comparing overlapping properties (e.g., Plant vs. Animal cells, Solid vs. Liquid).
6. **Taxonomy Tree (`taxonomy_tree`)**: For taxonomy, classification hierarchies, or family/government branches.
7. **Cartesian Grid (`cartesian_grid`)**: For graphing coordinate geometry, lines, triangles, angles, and algebra graphs.
8. **Column Comparison (`column_comparison`)**: For direct side-by-side card contrasts.
9. **Geo-Marker Map (`geo_marker`)**: For geography and history maps, highlighting regions with coordination pointers.
10. **Database Grid (`database_grid`)**: For displaying tabular data or periodic table grids.
11. **Before/After Slider (`before_after_slider`)**: For showing a wipe transition between cause and effect states.
12. **Quiz Checkpoint (`quiz_checkpoint`)**: For active recall summary questions at the end of the lesson.

Ensure the output is valid JSON and contains only the raw JSON block without markdown formatting wrapper, or wrapped in a standard ```json ... ``` codeblock.
"""
    return prompt
