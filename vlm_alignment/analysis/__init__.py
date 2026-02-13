"""Analysis tools for alignment research.

Imports are lazy to avoid requiring torch at package import time.
"""


def __getattr__(name):
    if name == "CKA":
        from .cka import CKA
        return CKA
    elif name == "AlignmentAnalyzer":
        from .alignment import AlignmentAnalyzer
        return AlignmentAnalyzer
    elif name == "InferenceSpeedBenchmark":
        from .speed_benchmark import InferenceSpeedBenchmark
        return InferenceSpeedBenchmark
    elif name == "CORALAnalyzer":
        from .coral import CORALAnalyzer
        return CORALAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
