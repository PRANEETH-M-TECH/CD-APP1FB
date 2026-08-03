"""
Single-source-of-truth loader for the HyperFrames template registry.

The registry itself lives at hyperframes_engine/shared/template-registry.json
(plain JSON so the Node engine can require() it and this module can json.load()
it with zero new dependencies on either side). Do not hardcode template ids,
"best for" descriptions, or selection constraints anywhere else - edit the JSON
file and every consumer (LLM prompt text, post-hoc audit/repair pass) updates
automatically.
"""
import os
import json
import logging

logger = logging.getLogger(__name__)

_MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MAIN_DIR, "..", "..", "..", ".."))
REGISTRY_PATH = os.path.join(_PROJECT_ROOT, "hyperframes_engine", "shared", "template-registry.json")
ICONS_PATH = os.path.join(_PROJECT_ROOT, "hyperframes_engine", "shared", "icons.js")

_cache = None
_icon_names_cache = None


def get_icon_names() -> list:
    """
    Names of the curated icons available to the LLM (see
    hyperframes_engine/shared/icons.js). Extracted from the JS source by
    regex instead of duplicating the list here, so the two stay in sync
    automatically when an icon is added/removed.
    """
    global _icon_names_cache
    if _icon_names_cache is not None:
        return _icon_names_cache
    import re
    try:
        with open(ICONS_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        names = re.findall(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*):\s*'<", text, re.MULTILINE)
        _icon_names_cache = [n for n in names if n != "dot"]
    except Exception as e:
        logger.error(f"[TemplateRegistry] Failed to load icon names from {ICONS_PATH}: {e}")
        _icon_names_cache = []
    return _icon_names_cache


def load_registry() -> dict:
    """Returns the {template_id: {...}} mapping, cached after first read."""
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _cache = data.get("templates", {})
    except Exception as e:
        logger.error(f"[TemplateRegistry] Failed to load {REGISTRY_PATH}: {e}")
        _cache = {}
    return _cache


def get_active_template_ids() -> list:
    """Template ids the LLM is allowed to choose from (status == 'active')."""
    registry = load_registry()
    return [tid for tid, meta in registry.items() if meta.get("status") == "active"]


def get_constraint_map() -> dict:
    """{template_id: [constraint_strings]} for active templates only."""
    registry = load_registry()
    return {
        tid: meta.get("constraints", [])
        for tid, meta in registry.items()
        if meta.get("status") == "active"
    }


def build_template_choice_line() -> str:
    """e.g. "'title_slide', 'concept_diagram', ..." for embedding in the prompt."""
    return ", ".join(f"'{tid}'" for tid in get_active_template_ids())


def build_icon_guidance_text() -> str:
    """
    Instructs the LLM to attach an 'icon' name (from the curated icon set) to
    every labeled item it can - center concepts, leaf nodes, cycle stages,
    taxonomy branches, comparison bullets, venn items, before/after bullets,
    and map markers. This is what turns those templates from plain text-in-
    boxes into something actually visual: each icon is a real rendered SVG
    picked from hyperframes_engine/shared/icons.js, not LLM-generated art, so
    an unrecognized/omitted icon name always degrades gracefully to a plain
    dot rather than breaking the scene.
    """
    icon_names = get_icon_names()
    if not icon_names:
        return ""
    icon_list = ", ".join(f"'{n}'" for n in icon_names)
    return (
        "### VISUAL ICONS (make every scene actually visual, not just text):\n"
        "For concept_diagram, cycle_template, taxonomy_tree, column_comparison, "
        "venn_diagram, before_after_slider, and geo_marker: give EVERY labeled item "
        "(the central concept, each leaf/branch/stage/bullet/marker) an `\"icon\"` "
        "field alongside its text, e.g. {\"text\": \"Evaporation\", \"icon\": \"sun\"}. "
        f"Choose the closest matching icon name from this exact list: {icon_list}. "
        "Pick the most semantically relevant icon for each concept (e.g. 'sun' for "
        "heat/energy/day, 'water_drop' for liquids, 'leaf' for plants, 'brain' for "
        "thinking/biology, 'factory' for industry, 'book' for learning). Never invent "
        "an icon name outside this list - if nothing fits well, omit the icon field "
        "for that item rather than guessing."
    )


def build_template_data_hints_block(indent: str = "        ") -> str:
    """Multi-line '// For <id>: <hint>' comment block for the prompt's template_data example."""
    registry = load_registry()
    lines = []
    for tid in get_active_template_ids():
        hint = registry[tid].get("template_data_hint", "")
        lines.append(f"{indent}// For '{tid}': {hint}")
    return "\n".join(lines)


def build_selection_rules_text() -> str:
    """Numbered prose rules derived from each active template's best_for + constraints."""
    registry = load_registry()
    lines = []
    n = 1
    for tid in get_active_template_ids():
        meta = registry[tid]
        rule = f"{n}. **{tid}**: {meta.get('best_for', '')}."
        constraints = meta.get("constraints", [])
        if "scene_1_only" in constraints:
            rule += " MUST be used ONLY for Scene 1."
        for c in constraints:
            if c.startswith("max_uses:"):
                rule += f" Use at most {c.split(':', 1)[1]} time(s) per lesson."
            if c == "last_scene_only":
                rule += " Use ONLY as the final scene."
        lines.append(rule)
        n += 1

    banned = [tid for tid, meta in registry.items() if meta.get("status") == "banned"]
    if banned:
        banned_list = ", ".join(f"`{tid}`" for tid in banned)
        lines.append(f"{n}. **Do NOT use** these template ids under any circumstances: {banned_list}.")
    return "\n".join(lines)


def _find_scene_1_template(valid: list, constraints: dict) -> str:
    for tid in valid:
        if "scene_1_only" in constraints.get(tid, []):
            return tid
    return valid[0] if valid else "concept_diagram"


def _pick_replacement(valid: list, constraints: dict, avoid: set, use_counts: dict) -> str:
    """First valid, non-scene-1-only, non-last-scene-only template not in `avoid`
    and not already at its max_uses limit."""
    for tid in valid:
        if tid in avoid:
            continue
        tid_constraints = constraints.get(tid, [])
        if "scene_1_only" in tid_constraints or "last_scene_only" in tid_constraints:
            continue
        max_uses = None
        for c in tid_constraints:
            if c.startswith("max_uses:"):
                max_uses = int(c.split(":", 1)[1])
        if max_uses is not None and use_counts.get(tid, 0) >= max_uses:
            continue
        return tid
    # Nothing else fits - fall back to the first valid template even if imperfect,
    # rather than leaving an invalid/banned template_id in place.
    return valid[0] if valid else "concept_diagram"


def repair_scene_templates(clips: list, log=None) -> list:
    """
    Enforces registry-derived constraints on an LLM-produced scene list in place:
    scene-1-only / last-scene-only placement, max_uses caps, and no consecutive
    duplicate template_ids. Mutates and returns `clips` (each a dict with a
    'template_id' key). `log` is an optional callable(str) for audit output.
    """
    log = log or (lambda msg: None)
    valid = get_active_template_ids()
    constraints = get_constraint_map()
    use_counts = {}
    last_idx = len(clips) - 1

    for idx, clip in enumerate(clips):
        tid = clip.get("template_id")
        tid_constraints = constraints.get(tid, None)

        if idx == 0:
            forced = _find_scene_1_template(valid, constraints)
            if tid != forced:
                clip["template_id"] = forced
                log(f"   [AUDIT REPAIR] Scene 1 forced to '{forced}'")
            tid = clip["template_id"]
        else:
            invalid = tid_constraints is None
            misplaced = tid_constraints is not None and (
                "scene_1_only" in tid_constraints
                or ("last_scene_only" in tid_constraints and idx != last_idx)
            )
            if invalid or misplaced:
                avoid = {clips[idx - 1].get("template_id")}
                replacement = _pick_replacement(valid, constraints, avoid, use_counts)
                log(f"   [AUDIT REPAIR] Swapped disallowed/misplaced template '{tid}' in Scene {idx + 1} to '{replacement}'")
                clip["template_id"] = replacement
                tid = replacement

            if tid == clips[idx - 1].get("template_id"):
                avoid = {tid}
                replacement = _pick_replacement(valid, constraints, avoid, use_counts)
                log(f"   [AUDIT REPAIR] Swapped consecutive duplicate '{tid}' in Scene {idx + 1} to '{replacement}'")
                clip["template_id"] = replacement
                tid = replacement

        # Enforce max_uses after any swap above may have changed tid
        tid_constraints = constraints.get(tid, [])
        max_uses = None
        for c in tid_constraints:
            if c.startswith("max_uses:"):
                max_uses = int(c.split(":", 1)[1])
        if max_uses is not None and use_counts.get(tid, 0) >= max_uses:
            avoid = {tid, clips[idx - 1].get("template_id")} if idx > 0 else {tid}
            replacement = _pick_replacement(valid, constraints, avoid, use_counts)
            log(f"   [AUDIT REPAIR] '{tid}' exceeded max_uses in Scene {idx + 1}, swapped to '{replacement}'")
            clip["template_id"] = replacement
            tid = replacement

        use_counts[tid] = use_counts.get(tid, 0) + 1

    return clips
