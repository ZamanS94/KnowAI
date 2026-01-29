from pathlib import Path
import os
import torch
from ErrorRateCalculation_sequenceMatching import calTechTermsError, calDiffErros
from ErrorRateCalculation_jiwer import calculate_diff_errors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# need to change the location based on user's folders' addresses

GroundTruth_Transcription_FOLDER = Path(
    "/scratch/project_2010972/sabina/QADental/data/TranscriptiongroundTruth/Finnish"
    )
new_transcription_folder_QADentalTool_Fi = Path(
    "/scratch/project_2010972/sabina/KnowAI/data/Transcription/QADentalTool/Fi"
    )
GroundTruth_TechnicalTerms_FOLDER_GPT5= Path(
    "/scratch/project_2010972/sabina/KnowAI/data/TechnicalTerms/GT/Finnish/GPT5"
    )
errorReport = Path("/scratch/project_2010972/sabina/KnowAI/HH/errorReport.txt")

'''
GroundTruth_Transcription_FOLDER_En = Path(
    "/scratch/project_2010972/sabina/QADental/data/TranscriptiongroundTruth/English"
    )
new_transcription_folder_QADentalTool_En = Path(
    "/scratch/project_2010972/sabina/KnowAI/data/Transcription/QADentalTool/En"
    )
'''

if __name__ == "__main__":
    
    transcription_language = "fi" #or en

    #For Finnish Clips
    # Spelling mistakes and standard Erros
    
    print("Starting WER and Spelling Error Calculation for Finnish Clips", flush=True)

    total_GTwords = 0
    total_Transcriptionwords = 0
    total_wer = 0
    total_deleted = 0
    total_added = 0
    total_spellingError = 0
    
    GTfiles_list_folder = GroundTruth_Transcription_FOLDER #change for english
    transcriptionFiles_Path = new_transcription_folder_QADentalTool_Fi #change for english

    with open(errorReport, "a", encoding="utf-8") as f:
        f.write("\n\n\nFinnish Clips - ***** Model WER & Spelling Mistakes\n\n")

    for file in os.listdir(GTfiles_list_folder):
        if file.endswith(".txt"):
            gtFile = os.path.join(GTfiles_list_folder, file)
            transcriptionFile = os.path.join(transcriptionFiles_Path, file)
            if not os.path.exists(transcriptionFile):
                print(f"!!!! Missing transcription file: {transcriptionFile}")
                continue

            try:
                _GT_words, _Transcription_words, wer, deleted_rate, added_rate, SpellingError_rate = calDiffErros(
                    gtFile, transcriptionFile, errorReport, transcription_language
                )
                '''
                _GT_words, _Transcription_words, wer, deleted_rate, added_rate, SpellingError_rate = calculate_diff_errors(
                    gtFile, transcriptionFile, errorReport, transcription_language
                )
                '''
                total_GTwords += _GT_words
                total_Transcriptionwords += _Transcription_words
                total_wer += wer * _GT_words / 100
                total_deleted += deleted_rate * _GT_words / 100
                total_added += added_rate * _GT_words / 100
                total_spellingError += SpellingError_rate * _Transcription_words / 100

            except Exception as e:
                print(f"Error processing {file}: {e}")

    print("avg", total_wer/total_GTwords)
   
    # Report average
    with open(errorReport, "a", encoding="utf-8") as f:
        f.write("AVERAGES FOR WER & SPELLING\n")
        f.write(f"Average WER: {total_wer/total_GTwords*100:.2f}%\n")
        f.write(f"Average Deleted Rate: {total_deleted/total_GTwords*100:.2f}%\n")
        f.write(f"Average Added Rate: {total_added/total_GTwords*100:.2f}%\n")
        f.write(f"Average Spelling Errors: {total_spellingError/total_Transcriptionwords*100:.2f}%\n")
        f.write("="*40 + "\n")
    print("\nAll standard eror calculataion done")


    #Tech Terms Error
    print("\nStarting Technical Terms Error Calculation", flush=True)

    GTterms_list_folder = GroundTruth_TechnicalTerms_FOLDER_GPT5
    Transcriptionterms_list_folder = new_transcription_folder_QADentalTool_Fi
    total_GTterms = 0
    GTterms_not_found = 0

    with open(errorReport, "a", encoding="utf-8") as f:
        f.write("\n\nFinnish Clips - **** Technical Terms Error Rate\n\n")

    for file in os.listdir(GTterms_list_folder):
        if file.endswith(".txt"):
            gt_terms_file = os.path.join(GTterms_list_folder, file)
            trans_terms_file = os.path.join(Transcriptionterms_list_folder, file)
            if not os.path.exists(trans_terms_file):
                print(f"!!!!!Missing transcription terms file: {trans_terms_file}")
                continue

            try:
                gt_count, missing_count = calTechTermsError(gt_terms_file, trans_terms_file, errorReport)
                total_GTterms += gt_count
                GTterms_not_found += missing_count
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Report average
    with open(errorReport, "a", encoding="utf-8") as f:
        if total_GTterms > 0:
            f.write(f"Total Technical Term: {total_GTterms}\n")
            f.write(f"Average Not found Technical Terms Error Rate: {GTterms_not_found/total_GTterms*100:.2f}%\n")
        else:
            f.write("Average Not found Technical Terms Error Rate: N/A\n")
        f.write("="*40 + "\n")

    print("\nTechnical terms error rate calculation done")
