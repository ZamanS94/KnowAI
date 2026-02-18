import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

client = OpenAI(api_key="sk-proj")

def get_openai_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
    
EXACT_FIELDS = [
    "Raportin tyyppi", "Tarkkailijan nimi", "Tarkkailijaorganisaatio",
    "Tarkkailija on kesätyöntekijä", "Tapahtuma-aika",
    "Sijaintitiedot", "Kuva", "Tapahtuma oli vakava"
]

SEMANTIC_FIELDS = [
    "Tapahtuma-alueen kuvaus", "Mahdolliset seuraukset",
    "Toteutetut toimenpiteet", "Ehdotus"
]

ALL_FIELDS = EXACT_FIELDS + SEMANTIC_FIELDS

EXACT_THRESHOLD = 0.9
SEMANTIC_THRESHOLD = 0.8


def C_evaluate_single_file(
    gt_file: Path,
    pred_file: Path,
    detailed_report_path: Path,
    summary_report_path: Path,
    threshold: float = 0.85
):
    gt = json.loads(gt_file.read_text(encoding="utf-8"))
    pred = json.loads(pred_file.read_text(encoding="utf-8"))

    field_similarity_scores = defaultdict(list)
    tp = fp = fn = tn = 0
    matched_fields = evaluated_fields = 0
    exact_matched = exact_evaluated = 0
    semantic_matched = semantic_evaluated = 0

    # detailed report
    with detailed_report_path.open("a", encoding="utf-8") as detailed_report:
        detailed_report.write(f"FILE: {gt_file.name} vs {pred_file.name}\n")
        detailed_report.write("-" * 60 + "\n")

        for field in ALL_FIELDS:
            gt_val = str(gt.get(field, "")).strip()
            pred_val = str(pred.get(field, "")).strip()

            if gt_val and pred_val:
                tp += 1
                emb_gt = get_openai_embedding(gt_val)
                emb_pred = get_openai_embedding(pred_val)
                sim = cosine_similarity([emb_gt], [emb_pred])[0][0]
                field_similarity_scores[field].append(sim)

                standard_threshold = EXACT_THRESHOLD if field in EXACT_FIELDS else SEMANTIC_THRESHOLD
                status = "PASS" if sim >= standard_threshold else "ERROR"

                if status == "PASS":
                    matched_fields += 1
                    if field in EXACT_FIELDS:
                        exact_matched += 1
                    else:
                        semantic_matched += 1

                evaluated_fields += 1
                if field in EXACT_FIELDS:
                    exact_evaluated += 1
                else:
                    semantic_evaluated += 1

                detailed_report.write(f"{field} | similarity={sim:.3f} | threshold={standard_threshold} | {status}\n")

            elif not gt_val and pred_val:
                fp += 1
                detailed_report.write(f"{field} | FP (pred present, GT missing)\n")

            elif gt_val and not pred_val:
                fn += 1
                detailed_report.write(f"{field} | FN (GT present, pred missing)\n")

            else:
                tn += 1
                detailed_report.write(f"{field} | TN (both missing/empty)\n")

        # summary
        total_relevant = tp + fn + fp
        overall_match = (tp / total_relevant) if total_relevant else 0

        exact_match = (exact_matched / exact_evaluated) if exact_evaluated else 0
        semantic_match = (semantic_matched / semantic_evaluated) if semantic_evaluated else 0

        detailed_report.write("\nSUMMARY PER FILE\n")
        detailed_report.write(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
        detailed_report.write(f"Overall Match: {overall_match:.2%}\n")
        detailed_report.write(f"Exact Fields Match: {exact_match:.2%}\n")
        detailed_report.write(f"Semantic Fields Match: {semantic_match:.2%}\n\n")

    # overall summary
    with summary_report_path.open("a", encoding="utf-8") as summary_report:
        result = "PASS" if overall_match >= threshold else "FAIL"
        summary_report.write("From Custom Evaluation" + "\n")
        summary_report.write(f"{gt_file.name}: {result}\n")
        
        
        
