"""
Mnemon evaluation framework — cognitive capability benchmarks.

Provides a suite of backend-agnostic benchmarks that characterise the
performance of each brain-inspired memory subsystem.  All benchmarks
operate against the ABC interfaces in mnemon.core.interfaces and can
run without a live Qdrant or FalkorDB instance.

Quick start
-----------
::

    from mnemon.evaluation import (
        BenchmarkSuite,
        BenchmarkResult,
        RetrievalBenchmark,
        ConsolidationBenchmark,
        ForgettingCurveBenchmark,
        MemoryInterferenceBenchmark,
        ConfidenceCalibrationBenchmark,
        WorkingMemoryBenchmark,
        ContinualLearningBenchmark,
        CycleLatencyBenchmark,
    )

    suite = BenchmarkSuite(benchmarks=[
        RetrievalBenchmark(),
        ForgettingCurveBenchmark(),
        MemoryInterferenceBenchmark(),
        ConfidenceCalibrationBenchmark(),
        WorkingMemoryBenchmark(),
        ContinualLearningBenchmark(),
        CycleLatencyBenchmark(),
    ])

    results = await suite.run_all(
        retrieval=(episodic, query_pairs, 10),
        ...
    )
    print(suite.summary(results))
"""

from __future__ import annotations

from mnemon.evaluation.benchmarks import (
    Benchmark,
    BenchmarkResult,
    ConfidenceCalibrationBenchmark,
    ConsolidationBenchmark,
    ContinualLearningBenchmark,
    CycleLatencyBenchmark,
    ForgettingCurveBenchmark,
    MemoryInterferenceBenchmark,
    RetrievalBenchmark,
    WorkingMemoryBenchmark,
)
from mnemon.evaluation.suite import BenchmarkSuite

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkSuite",
    "ConfidenceCalibrationBenchmark",
    "ConsolidationBenchmark",
    "ContinualLearningBenchmark",
    "CycleLatencyBenchmark",
    "ForgettingCurveBenchmark",
    "MemoryInterferenceBenchmark",
    "RetrievalBenchmark",
    "WorkingMemoryBenchmark",
]
