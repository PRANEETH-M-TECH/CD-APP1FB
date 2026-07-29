import json
import os

def main():
    report_path = r"c:\Users\home\Desktop\CG-DEV\CD-APP1FB\backend\app\orchestrator_test\test_outputs\query_report_20260729_164748.json"
    output_dir = r"c:\Users\home\Desktop\CG-DEV\CD-APP1FB\hyperframes_engine\outputs\chola_dynasty"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    storyboard = data["orchestrator_output"]["video_storyboard"]
    storyboard["lesson_id"] = "chola_dynasty"
    
    output_json_path = os.path.join(output_dir, "lesson.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, indent=2)
        
    print(f"Extracted storyboard saved to {output_json_path}")

if __name__ == "__main__":
    main()
