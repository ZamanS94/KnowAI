import json
import argparse
from pathlib import Path
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from custom_llm import CustomGeminiFlash
from custom_llm2 import CustomOpenAI

# Fields to validate
FIELDS_TO_CHECK = [
    "Raportin tyyppi",
    "Tarkkailijan nimi",
    "Tarkkailijaorganisaatio",
    "Tarkkailija on kesätyöntekijä",
    "Tapahtuma-aika",
    "Sijaintitiedot",
    "Kuva",
    "Tapahtuma oli vakava",
    "Tapahtuma-alueen kuvaus",
    "Mahdolliset seuraukset",
    "Toteutetut toimenpiteet",
    "Ehdotus"
]


# Single-file evaluation
def G_evaluate_single_file(
    gt_file_path: Path,
    transcription_file_path: Path,
    report_file_path: Path,
    threshold: float,
):

    if not gt_file_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file_path}")

    if not transcription_file_path.exists():
        raise FileNotFoundError(f"Transcription file not found: {transcription_file_path}")

    with gt_file_path.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)

    with transcription_file_path.open("r", encoding="utf-8") as f:
        pred_data = json.load(f)

    filtered_extracted = {k: pred_data.get(k, None) for k in FIELDS_TO_CHECK}

    # prompt
    prompt = f"""
You are an expert judge evaluating information extraction.

GROUND TRUTH JSON:
{json.dumps(gt_data, indent=2)}

EXTRACTED JSON:
{json.dumps(filtered_extracted, indent=2)}
"""
    custom_aiModel = CustomOpenAI() # Or CustomGeminiFlash()

    geval_metric = GEval(
        name="JSON Extraction Quality",
        criteria="""
    Evaluate whether the extracted JSON faithfully represents the source text.
    - All required fields must be present
    - Values must be present in the source text
    - No hallucinations. Halucinations should be penalized
    - Partial correctness is okay if not hallucinated
    - Complete ansers should be rewarded
    """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        model=custom_aiModel,
        threshold=threshold,
    )

    # test case
    test_case = LLMTestCase(
        input=prompt,
        actual_output=json.dumps(filtered_extracted, indent=2)
    )

    score = geval_metric.measure(test_case)
    reason = geval_metric.reason

    with report_file_path.open("a", encoding="utf-8") as report:
        report.write("From Judge LLM" + "\n")
        report.write(f"SCORE: {score}\n")
        report.write(f"THRESHOLD: {threshold}\n")
        report.write(f"PASSED: {score >= threshold}\n")
        report.write("REASONING: ")
        report.write(reason + "\n\n")
        report.write("=" * 80 + "\n")

    print(f"Evaluation complete. Report saved to: {report_file_path}")
