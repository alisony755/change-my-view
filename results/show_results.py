"""
Shows each model's latest results, saved by results/run_log.py in results/logs/<model>.txt
(bilstm.txt, tfidf_lr.txt, roberta.txt), as they were printed in the terminal.

Run:
    python results/show_results.py          one table with the main results of every model
    python results/show_results.py bilstm   print the bilstm results in full
"""
import re
import sys

from run_log import LOGS


def results_by_model():
    """{model: its latest output} for every model with a file in results/logs/."""
    return {path.stem: path.read_text() for path in sorted(LOGS.glob("*.txt"))
            if not path.stem.endswith(".running")}  # .running: a run that has not finished yet


def block(text, title, lines=4):
    """The lines printed right under a title such as "Dev (changemyview)", or "" if missing."""
    match = re.search(rf"^{re.escape(title)}\s*\n((?:.*\n?){{0,{lines}}})", text, re.MULTILINE)
    return match.group(1) if match else ""


def number(pattern, text):
    """The first number after the pattern in the text, or "" if it is not there."""
    match = re.search(pattern + r"\s*([\d.]+)", text)
    return match.group(1) if match else ""


def main_results(text):
    """The main results of a run, read from the lines the shared metrics print. Results a run did
    not print (no test sets without --test, no ROC AUC for TF-IDF) stay empty."""
    dev = block(text, "Dev (changemyview)")
    return {
        "Dev acc": number(r"^Accuracy:", dev) if dev.startswith("Accuracy") else "",
        "Length-bal": number(r"Length-balanced accuracy:", dev),
        "Similar": number(r"Similar-length accuracy:", dev),
        "F1": number(r"F1", text),  # the first F1 is the dev one
        "ROC AUC": number(r"ROC AUC:?", text),
        "Test acc": number(r"Accuracy:", block(text, "Seen test (changemyview)")),
        "Persuasion": number(r"Accuracy:", block(text, "Persuasion test (Winning Arguments)")),
    }


def print_table(rows):
    """Prints rows (lists of strings) as columns, the first row as the header."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for index, row in enumerate(rows):
        print("  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)))
        if index == 0:
            print("  " + "  ".join("-" * width for width in widths))


def compare(results):
    """One table: each model's main results."""
    print("\nLatest results of each model\n============================")
    rows = [[model] + list(main_results(output).values()) for model, output in results.items()]
    print_table([["Model"] + list(main_results("").keys())] + rows)
    print("\nTo see a model's results in full: python results/show_results.py <model>")


def show(results, model):
    """Prints a model's latest results in full."""
    if model not in results:
        sys.exit(f"No results for {model}. Models with results: {', '.join(results) or 'none yet'}")
    print(f"{model}\n{'#' * len(model)}")
    print(results[model])


if __name__ == "__main__":
    results = results_by_model()
    if not results:
        sys.exit(f"No saved results yet. They are saved in {LOGS}")
    if len(sys.argv) == 1:
        compare(results)
    else:
        show(results, sys.argv[1])