"""
Keeps the output of each model's latest run: results/logs/<model>.txt (for example
results/logs/bilstm.txt) holds everything that run printed, as the terminal showed it.
A new run replaces the file, but only when it finishes: if it stops early (an error, or Ctrl+C),
the previous file stays as it was. results/show_results.py shows the files.

Use it once, at the start of a training script's main part:
    from results.run_log import save_output
    save_output("tfidf_lr")  # -> results/logs/tfidf_lr.txt
"""
import atexit
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent  # the results/ folder
LOGS = RESULTS / "logs"                    # one text file per model


class Tee:
    """Sends everything printed both to the terminal and to a file, like the tee command."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal, self.file = sys.stdout, open(path, "w")

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def __getattr__(self, name):  # anything else, such as isatty(), behaves like the terminal
        return getattr(self.terminal, name)


def save_output(model):
    """From now on, everything printed also goes to results/logs/<model>.txt. It is written to a
    temporary file first, which replaces the model's file when the run finishes. Warnings and
    progress bars are not saved, because they are not printed to the same place."""
    path = LOGS / f"{model}.txt"
    temporary = path.with_name(f"{model}.running.txt")
    tee = Tee(temporary)
    sys.stdout = tee

    # A run that ends with an error, Ctrl+C or sys.exit("...") did not finish
    finished = [True]
    show_error, leave = sys.excepthook, sys.exit

    def on_error(*error):
        finished[0] = False
        show_error(*error)

    def on_leave(code=None):
        if code not in (None, 0):
            finished[0] = False
        leave(code)

    sys.excepthook, sys.exit = on_error, on_leave

    def at_end():
        sys.stdout = tee.terminal
        tee.file.close()
        shown = path.relative_to(RESULTS.parent)
        if finished[0]:
            temporary.replace(path)
            print(f"\nSaved this output to {shown}")
        else:
            temporary.unlink()
            print(f"\nThe run did not finish, so {shown} was left as it was")

    atexit.register(at_end)
    return path