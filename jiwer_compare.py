#!/usr/bin/env python3

import jiwer
import sys
from itertools import zip_longest

# ANSI colors
RED = "\033[92m"   # word in reference
GREEN = "\033[91m"  # word in hypothesis
RESET = "\033[0m"

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()  # Keep line structure

def clean_words(text):
    transforms = jiwer.Compose([
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
    ])
    return transforms(text).split()

def main():
    if len(sys.argv) != 3:
        print("Usage: ./jiwer_compare.py <reference.txt> <hypothesis.txt>")
        sys.exit(1)

    ref_lines = read_file(sys.argv[1])
    hyp_lines = read_file(sys.argv[2])

    # Pad shorter file with empty lines
    max_lines = max(len(ref_lines), len(hyp_lines))
    while len(ref_lines) < max_lines:
        ref_lines.append("\n")
    while len(hyp_lines) < max_lines:
        hyp_lines.append("\n")

    total_ref_words = []
    total_hyp_words = []

    print("Differences by line:\n")

    for lineno, (ref_line, hyp_line) in enumerate(zip(ref_lines, hyp_lines), 1):
        ref_words = clean_words(ref_line)
        hyp_words = clean_words(hyp_line)

        total_ref_words.extend(ref_words)
        total_hyp_words.extend(hyp_words)

        # Only show lines with differences
        if ref_words != hyp_words:
            colored_ref = []
            colored_hyp = []
            for r, h in zip_longest(ref_words, hyp_words, fillvalue="<missing>"):
                if r != h:
                    colored_ref.append(f"{RED}{r}{RESET}")
                    colored_hyp.append(f"{GREEN}{h}{RESET}")
                else:
                    colored_ref.append(r)
                    colored_hyp.append(h)

            print(f"Line {lineno}:")
            print("Reference: ", " ".join(colored_ref))
            print("Hypothesis:", " ".join(colored_hyp))
            print()

    # Compute overall WER
    overall_wer = jiwer.wer(total_ref_words, total_hyp_words)
    print(f"WER: {overall_wer:.2%}")

if __name__ == "__main__":
    main()
