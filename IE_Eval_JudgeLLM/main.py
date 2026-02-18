from pathlib import Path
from Custom_evaluation import C_evaluate_single_file
from JudgeLLM import G_evaluate_single_file 

# Config
TRANSCRIPTION_FOLDER = Path("/scratch/project_2010972/sabina/judgeLLM/data/transcription")  
GROUND_TRUTH_DIR = Path("/scratch/project_2010972/sabina/judgeLLM/data/GT")     
OUTPUT_DIR = Path("/scratch/project_2010972/sabina/judgeLLM/data/gpt4")    

# Reports
IE_DETAILED_REPORT = Path("IE_Detailed_ErrorReport.txt")
LLM_DETAILED_REPORT = Path("LLM_Detailed_ErrorReport.txt")
PASS_FAIL_REPORT = Path("IE_PassFail_Report.txt")

THRESHOLD = 0.85

if __name__ == "__main__":

    for output_file in OUTPUT_DIR.glob("*.json"):
        base_name = output_file.stem

        gt_file = GROUND_TRUTH_DIR / f"{base_name}.json"
        transcription_file = TRANSCRIPTION_FOLDER / f"{base_name}.txt"

        if not gt_file.exists():
            print(f"GT file missing for {base_name}")
            continue
        if not transcription_file.exists():
            print(f"Transcription file missing for {base_name}")
            continue

        print(f"\nProcessing file: {base_name}")

        # Custom evaluation 
        C_evaluate_single_file(
            gt_file=gt_file,
            pred_file=output_file, #json
            detailed_report_path=IE_DETAILED_REPORT,
            summary_report_path=PASS_FAIL_REPORT,
            threshold=THRESHOLD
        )

        # JudgeLLM evaluation 
        G_evaluate_single_file(
            gt_file_path=gt_file,    
            transcription_file_path=output_file,
            report_file_path=PASS_FAIL_REPORT,
            threshold=THRESHOLD
        )

    print("\nAll evaluations completed.")
    print(f"IE Detailed Report: {IE_DETAILED_REPORT}")
    print(f"LLM Detailed Report: {LLM_DETAILED_REPORT}")
    print(f"Pass/Fail Report: {PASS_FAIL_REPORT}")
