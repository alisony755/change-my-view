"""
Generalization: does a model hold up on subreddits it never trained on, and on the persuasion
test? Shared by all models. generalization_report prints every test set as one table, with the
pooled summaries, and test_results_from_log reads that table back from a saved log, for the
charts in results/visualizations/.

Use it in a training script, after the test sets (evaluate returns the numbers it prints):
    test_results = {"Seen": {}, "Unseen": {}, "Persuasion": {}}
    test_results["Seen"][name] = evaluate(...)
    generalization_report(test_results)
"""
import re

GROUPS = ("Seen", "Unseen", "Persuasion")
PAIR_COUNT = {"accuracy": "pairs", "balanced": "balanced_pairs"}  # which count goes with which accuracy
ROW = re.compile(r"^(.+?)\s{2,}(Seen|Unseen|Persuasion)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$")


def pooled(results, key="accuracy"):
    """Accuracy over all the pairs of several test sets together, so a bigger set counts for more.
    results is {set name: the numbers evaluate returns}, key is "accuracy" or "balanced"."""
    pairs = sum(result[PAIR_COUNT[key]] for result in results.values())
    return sum(result[key] * result[PAIR_COUNT[key]] for result in results.values()) / pairs


def average(results, key="accuracy"):
    """Plain average over the test sets: every subreddit counts the same, however small."""
    return sum(result[key] for result in results.values()) / len(results)


def generalization_report(test_results):
    """Prints every test set in one table, then seen vs unseen accuracy. test_results is
    {"Seen": {...}, "Unseen": {...}, "Persuasion": {...}}, each {set name: what evaluate returned}."""
    print("\nGeneralization\n==============")
    print(f"{'Test set':<22}  {'Group':<10}  {'Pairs':>5}  {'Accuracy':>8}  {'Bal. pairs':>10}  {'Balanced':>8}")
    for group in GROUPS:
        for name, result in test_results.get(group, {}).items():
            print(f"{name:<22}  {group:<10}  {result['pairs']:>5}  {result['accuracy']:>8.4f}  "
                  f"{result['balanced_pairs']:>10}  {result['balanced']:>8.4f}")
    seen, unseen = test_results.get("Seen", {}), test_results.get("Unseen", {})
    if seen and unseen:
        print()
        for key, label in (("accuracy", "all pairs"), ("balanced", "length-balanced pairs")):
            print(f"Seen {pooled(seen, key):.4f} | unseen pooled {pooled(unseen, key):.4f}, "
                  f"unseen average {average(unseen, key):.4f} | gap (seen - unseen pooled) "
                  f"{pooled(seen, key) - pooled(unseen, key):+.4f}  [{label}]")


def test_results_from_log(text):
    """The reverse of generalization_report's table: {group: {set name: numbers}} read from the
    text of a saved log. Empty groups for a log made without --test."""
    results = {group: {} for group in GROUPS}
    for line in text.splitlines():
        row = ROW.match(line)
        if row:
            name, group, pairs, accuracy, balanced_pairs, balanced = row.groups()
            results[group][name] = {"accuracy": float(accuracy), "pairs": int(pairs),
                                    "balanced": float(balanced), "balanced_pairs": int(balanced_pairs)}
    return results