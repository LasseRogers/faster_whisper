#!/usr/bin/env python3

import jiwer
import sys
import difflib

# ANSI colors
RED = "\033[91m"
RESET = "\033[0m"

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def clean_words(text):
    transforms = jiwer.Compose([
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    return transforms(text).split()

def highlight_diff(ref_words, hyp_words):
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    ref_colored = []
    hyp_colored = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            ref_colored.extend(ref_words[i1:i2])
            hyp_colored.extend(hyp_words[j1:j2])
        elif tag == "replace":
            ref_colored.extend(f"{RED}{w}{RESET}" for w in ref_words[i1:i2])
            hyp_colored.extend(f"{RED}{w}{RESET}" for w in hyp_words[j1:j2])
        elif tag == "delete":
            ref_colored.extend(f"{RED}{w}{RESET}" for w in ref_words[i1:i2])
            hyp_colored.append(f"{RED}<MISSING>{RESET}")
        elif tag == "insert":
            ref_colored.append(f"{RED}<MISSING>{RESET}")
            hyp_colored.extend(f"{RED}{w}{RESET}" for w in hyp_words[j1:j2])

    return " ".join(ref_colored), " ".join(hyp_colored)

def main():
    if len(sys.argv) != 3:
        print("Usage: ./jiwer_compare.py <reference.txt> <hypothesis.txt>")
        sys.exit(1)

    ref_lines = read_file(sys.argv[1])
    hyp_lines = read_file(sys.argv[2])

    max_lines = max(len(ref_lines), len(hyp_lines))
    while len(ref_lines) < max_lines:
        ref_lines.append("\n")
    while len(hyp_lines) < max_lines:
        hyp_lines.append("\n")

    total_ref_words = []
    total_hyp_words = []

    for lineno, (ref_line, hyp_line) in enumerate(zip(ref_lines, hyp_lines), 1):
        ref_words = clean_words(ref_line)
        hyp_words = clean_words(hyp_line)

        total_ref_words.extend(ref_words)
        total_hyp_words.extend(hyp_words)

        if ref_words != hyp_words:
            ref_colored, hyp_colored = highlight_diff(ref_words, hyp_words)
            print(f"Line {lineno}:")
            print("Reference: ", ref_colored)
            print("Hypothesis:", hyp_colored)

    wer = jiwer.wer(" ".join(total_ref_words), " ".join(total_hyp_words))
    print(f"\nWER: {wer:.2%}")

if __name__ == "__main__":
    main()