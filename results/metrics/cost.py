"""
Cost: how long a model takes to train, how many parameters it has and how much memory the run
needs, printed the same way for every model so their costs can be compared.
"""
import sys


def peak_memory_mb():
    """The most memory this Python process has used so far, in MB, data loading included. None on
    Windows, where Python has no resource module."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # macOS counts bytes, Linux KB

def report_cost(seconds, parameters):
    """Prints the training time, the number of parameters and the run's peak memory. Time the
    training with time.perf_counter(): tracking memory while training (tracemalloc) makes it
    several times slower, so the memory comes from the operating system instead."""
    memory = peak_memory_mb()
    print(f"\nTraining time: {seconds:.1f} s | parameters: {parameters:,} | peak memory of the run: "
          + (f"{memory:,.0f} MB" if memory is not None else "not available on Windows"))
