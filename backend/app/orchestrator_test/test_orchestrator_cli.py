import os
import sys
import json
import time

# Ensure project root is in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.orchestrator_test.test_runner import (
    authenticate_student_by_email,
    run_orchestrator_pipeline,
    OUTPUTS_DIR
)


def print_banner():
    print("\n" + "=" * 65)
    print("      CHADUVU GURU - ORCHESTRATOR LAYER TERMINAL CLI HARNESS     ")
    print("==================================================")
    print(" Real Backend Test Harness | No Sarvam Audio | Real RAG & Firebase")
    print("=" * 65 + "\n")


def display_report_console(report: dict):
    """Displays a clean, structured inspection trace in terminal."""
    out = report.get("orchestrator_output", {})
    student = report.get("authenticated_student", {})
    rag_chunks = report.get("retrieved_top10_chunks", [])

    print("\n" + "=" * 65)
    print(" [REPORT] ORCHESTRATOR EXECUTION REPORT (INSPECTION TRACE)")
    print("=" * 65)

    print(f"\n[1. STUDENT SESSION]")
    print(f" * Name  : {student.get('name')} ({student.get('email')})")
    print(f" * Grade : Class {student.get('class')} | Board: {student.get('board')}")

    print(f"\n[2. RAW QUERY & REFORMULATION]")
    print(f" * Raw User Input       : \"{report.get('raw_user_query')}\"")
    print(f" * Reformulated Query   : \"{out.get('reformulated_query')}\"")
    print(f" * Matched Subject      : {out.get('matched_subject') or 'N/A'}")
    print(f" * Matched Chapter      : {out.get('matched_chapter') or 'N/A'}")

    print(f"\n[3. SAFETY & CLASSIFICATION]")
    auth_status = "YES (ALLOWED)" if out.get('is_authorized') else "NO (REFUSED)"
    print(f" * Is Authorized?       : {auth_status}")
    if not out.get('is_authorized'):
        print(f" * Refusal Message      : \"{out.get('refusal_reason')}\"")
    print(f" * Classification       : {out.get('classification')}")
    print(f" * Format Decision      : {out.get('format_decision')} (Complexity Level {out.get('complexity_level')})")

    if report.get("rag_retrieval_executed"):
        print(f"\n[4. RAG VECTOR RETRIEVAL ({len(rag_chunks)} Chunks Retrieved)]")
        for chunk in rag_chunks[:3]:  # Display top 3 snippets in console
            print(f"  - Chunk #{chunk.get('chunk_index')} [Score: {chunk.get('score')}]: {chunk.get('content_snippet')}")

    print(f"\n[5. FINAL TEXT NARRATION SCRIPT]")
    print(f" {out.get('text_narration') or 'None'}")

    storyboard = out.get("video_storyboard")
    if storyboard:
        # Support both new dictionary format and old list format
        scenes = storyboard.get("scenes", []) if isinstance(storyboard, dict) else storyboard
        print(f"\n[6. HYPERFRAMES VIDEO STORYBOARD ({len(scenes)} Scenes)]")
        for scene in scenes:
            scene_no = scene.get('scene_no') or scene.get('scene_number') or 0
            scene_purpose = scene.get('purpose') or scene.get('scene_title') or 'N/A'
            template_id = scene.get('template_id') or 'concept_diagram'
            duration = scene.get('estimated_duration_seconds') or 10
            script = scene.get('teacher_script') or scene.get('narration_text') or ''
            
            print(f"  * Scene #{scene_no} [Template: {template_id}] ({duration}s)")
            print(f"     Purpose   : {scene_purpose}")
            print(f"     Script    : {script}\n")
    else:
        print(f"\n[6. HYPERFRAMES VIDEO STORYBOARD]")
        print(" * Null (QUICK_ANSWER format decision - No video rendering required)")

    print(f"\n[SAVED AUDIT REPORT]")
    print(f" - Report Saved To : {report.get('saved_report_path')}")
    print("=" * 65 + "\n")


def main():
    print_banner()

    current_student = None

    while True:
        if not current_student:
            print("\n[AUTH] STUDENT AUTHENTICATION REQUIRED")
            email_input = input("Enter Student Email (e.g. student7@cg.com) [or 'exit' to quit]: ").strip()

            if email_input.lower() == "exit":
                print("\n[INFO] Exiting Orchestrator CLI Test Harness. Goodbye!")
                break

            if not email_input:
                email_input = "student7@cg.com"

            current_student = authenticate_student_by_email(email_input)
            print(f"\n[SESSION LOCKED] Current Active User: {current_student['name']} (Class {current_student['class']})")
            print("Type 'logout' to switch users, or 'exit' to quit.\n")

        query_input = input(f"[{current_student['name']} | Class {current_student['class']}] Ask a question: ").strip()

        if not query_input:
            continue

        if query_input.lower() == "exit":
            print("\n[INFO] Exiting Orchestrator CLI Test Harness. Goodbye!")
            break

        if query_input.lower() == "logout":
            print(f"\n[INFO] Logging out {current_student['name']}...")
            current_student = None
            continue

        # Execute Orchestrator Pipeline
        try:
            report = run_orchestrator_pipeline(query_input, current_student)
            display_report_console(report)
        except Exception as e:
            print(f"\n[ERROR] Pipeline execution failed: {e}\n")


if __name__ == "__main__":
    main()
