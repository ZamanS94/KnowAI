from pathlib import Path
from difflib import SequenceMatcher
import pyvoikko as v
from spellchecker import SpellChecker
import re


def clean_text_transcription(text):
    text = text.lower()
    text = re.sub(r"[^a-zåäö0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


# 1 TECHNICAL TERMS ERROR

def calTechTermsError(groundtruth_file, transcription_file, errorReport: Path):
    
    with open(groundtruth_file, "r", encoding="utf-8") as f:
        gt_terms = [
            clean_text_transcription(line.replace("-", " "))
            for line in f if line.strip()
        ]
        
    with open(transcription_file, "r", encoding="utf-8") as f:
        trans_text = " ".join(line.strip() for line in f if line.strip())
        trans_words = clean_text_transcription(trans_text.replace("-", " ")) 

    not_found_terms = []
    for term_words in gt_terms:
        found = False
        for i in range(len(trans_words) - len(term_words) + 1):
            if trans_words[i:i + len(term_words)] == term_words:
                found = True
                break
        if not found:
            not_found_terms.append(" ".join(term_words))

    total_terms = len(gt_terms)
    not_found_count = len(not_found_terms)
    percentage_not_found = (not_found_count / total_terms * 100) if total_terms > 0 else 0

    file_name = Path(groundtruth_file).name
    with open(errorReport, "a", encoding="utf-8") as f:
        f.write(f"File: {file_name}\n")
        f.write(f"Total GT terms: {total_terms}\n")
        f.write(f"Not found rate: {percentage_not_found:.2f}%\n")
        if not_found_terms:
            f.write(f"Terms not found: {', '.join(sorted(not_found_terms))}\n")
        f.write("=" * 40 + "\n")

    return total_terms, not_found_count


# 2 WER / Delete error / Added error

def calSMatcherErros(words1, words2):

    s = SequenceMatcher(None, words1, words2)

    num_subs = 0
    num_del = 0
    num_ins = 0
    num_words = len(words1)

    for tag, i1, i2, j1, j2 in s.get_opcodes():

        if tag == 'replace':
            # substitutions = minimum number of words replaced
            subs = min(i2 - i1, j2 - j1)
            num_subs += subs

            # extra GT words → deletions
            num_del += max(0, (i2 - i1) - (j2 - j1))

            # extra TR words → insertions
            num_ins += max(0, (j2 - j1) - (i2 - i1))

        elif tag == 'delete':
            num_del += i2 - i1

        elif tag == 'insert':
            num_ins += j2 - j1

    wer = ((num_subs + num_del + num_ins) / num_words * 100) if num_words > 0 else 0
    deleted_rate = (num_del / num_words * 100) if num_words > 0 else 0
    added_rate = (num_ins / num_words * 100) if num_words > 0 else 0

    return wer, deleted_rate, added_rate



# 3 Spelling mistakes

def calSpellErros(words):

    errors = []
    for word in words:
        if not v.analyse(word):  # No analysis → likely misspelled
            errors.append(word)

    total_words = len(words)
    num_errors = len(errors)
    error_rate = (num_errors / total_words * 100) if total_words > 0 else 0

    return error_rate, errors

def calSpellErros_En(words): 
    
    spell = SpellChecker()
    misspelled = spell.unknown(words)
    total_words = len(words)
    num_errors = len(misspelled)
    error_rate = (num_errors / total_words * 100) if total_words > 0 else 0

    return error_rate, list(misspelled)


# main Error cal. method

def calDiffErros(groundtruth_file, transcription_file, errorReport: Path, language: str = "fi"):

    text1 = Path(groundtruth_file).read_text(encoding="utf-8")
    text2 = Path(transcription_file).read_text(encoding="utf-8")

    words1 = clean_text_transcription(text1)
    words2 = clean_text_transcription(text2)

    totalGT_words = len(words1)
    totalTranscription_words = len(words2)

    wer, deleted_rate, added_rate = calSMatcherErros(words1, words2)
    if language == "fi":
        spelling_rate, spelling_errors = calSpellErros(words2)
    if language == "en":
        spelling_rate, spelling_errors = calSpellErros_En(words2)
    file_name = Path(groundtruth_file).name
    print(file_name, wer)
    with open(errorReport, "a", encoding="utf-8") as f:
        f.write(f"File: {file_name}\n")
        f.write(f"Total words in GT: {totalGT_words}\n")
        f.write(f"Total words in Transcription: {totalTranscription_words}\n")
        f.write(f"WER: {wer:.2f}% | Deleted: {deleted_rate:.2f}% | Added: {added_rate:.2f}%\n")
        f.write(f"Spelling Error Rate: {spelling_rate:.2f}%\n")
        f.write(f"Misspelled words: {', '.join(spelling_errors)}\n")
        f.write("=" * 40 + "\n")
   
    return (
        totalGT_words,
        totalTranscription_words,
        wer,
        deleted_rate,
        added_rate,
        spelling_rate,
    )
