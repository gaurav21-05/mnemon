"""
BenchmarkSuite — aggregates and runs multiple Mnemon benchmarks.

Brain analog: A comprehensive neuropsychological test battery that assesses
multiple cognitive domains in a single session, producing a unified performance
profile across memory encoding, retrieval, consolidation, and metacognition.
"""

from __future__ import annotations

import logging
from typing import Any

from mnemon.evaluation.benchmarks import Benchmark, BenchmarkResult

logger = logging.getLogger(__name__)

# Column widths for the summary table
_COL_NAME = 34
_COL_METRIC = 24
_COL_VALUE = 10
_COL_DURATION = 12


class BenchmarkSuite:
    """
    A composable collection of cognitive capability benchmarks.

    Brain analog: A full neuropsychological assessment battery — individual
    benchmarks probe distinct cognitive subsystems, and the suite provides
    an integrated performance profile that reveals trade-offs across memory
    encoding, retrieval, consolidation, decay, interference resistance,
    calibration, and working-memory management.

    Usage
    -----
    ::

        suite = BenchmarkSuite(benchmarks=[retrieval_bench, consolidation_bench])
        results = await suite.run_all(
            retrieval=(episodic, queries),
            consolidation=(engine, episodes, gt_triples),
        )
        print(suite.summary(results))
    """

    def __init__(self, benchmarks: list[Benchmark]) -> None:
        """
        Initialise the suite with a list of Benchmark instances.

        Parameters
        ----------
        benchmarks:
            Ordered list of benchmarks to run.  Each benchmark's ``name``
            attribute is used to look up its kwargs in ``run_all``.
        """
        self._benchmarks = list(benchmarks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def benchmarks(self) -> list[Benchmark]:
        """Return the ordered list of benchmarks in this suite."""
        return list(self._benchmarks)

    def add(self, benchmark: Benchmark) -> None:
        """Append a benchmark to the suite."""
        self._benchmarks.append(benchmark)

    async def run_all(self, **kwargs: Any) -> list[BenchmarkResult]:
        """
        Run every benchmark in the suite sequentially.

        Keyword arguments are dispatched to each benchmark by name.  Pass
        per-benchmark arguments as a tuple or dict keyed by the benchmark's
        ``name`` attribute:

        ::

            await suite.run_all(
                retrieval=(episodic, query_pairs, 10),
                consolidation=(engine, episodes, gt_triples),
                forgetting_curve=(episodic, episode_ids, [60, 3600, 86400]),
                memory_interference=(episodic, old_qs, new_eps, old_expected),
                confidence_calibration=(predictions,),
                working_memory=(wm_instance, blocks, 8192),
            )

        If a benchmark's name is not present in kwargs, it is called with no
        arguments (useful for benchmarks that need no external dependencies).

        Parameters
        ----------
        **kwargs:
            Mapping of benchmark_name -> positional args tuple or dict of
            keyword args.  Tuples are unpacked as positional arguments;
            dicts are unpacked as keyword arguments.

        Returns
        -------
        list[BenchmarkResult]
            Results in the same order as the suite's benchmark list.
        """
        results: list[BenchmarkResult] = []

        for bench in self._benchmarks:
            bench_kwargs = kwargs.get(bench.name, ())

            logger.info("BenchmarkSuite: starting benchmark '%s'", bench.name)
            try:
                if isinstance(bench_kwargs, dict):
                    result = await bench.run(**bench_kwargs)
                elif isinstance(bench_kwargs, (list, tuple)):
                    result = await bench.run(*bench_kwargs)
                else:
                    result = await bench.run()
            except Exception as exc:
                logger.error(
                    "BenchmarkSuite: benchmark '%s' raised an exception: %s",
                    bench.name,
                    exc,
                    exc_info=True,
                )
                # Record a failure result so the suite always returns one
                # entry per benchmark, preserving index alignment.
                result = BenchmarkResult(
                    benchmark_name=bench.name,
                    metrics={},
                    details={"error": str(exc)},
                    duration_ms=0.0,
                )

            results.append(result)
            logger.info(
                "BenchmarkSuite: '%s' completed in %.1f ms",
                bench.name,
                result.duration_ms,
            )

        return results

    def summary(self, results: list[BenchmarkResult]) -> str:
        """
        Format a text summary table of all benchmark results.

        Each benchmark occupies one or more rows — one row per metric —
        followed by a duration row.  Failed benchmarks (empty metrics with
        an 'error' key in details) are clearly flagged.

        Parameters
        ----------
        results:
            List returned by ``run_all``.

        Returns
        -------
        str
            Multi-line table suitable for printing to a terminal or log file.
        """
        if not results:
            return "BenchmarkSuite: no results to display."

        separator = "-" * (_COL_NAME + _COL_METRIC + _COL_VALUE + _COL_DURATION + 5)
        header = (
            f"{'Benchmark':<{_COL_NAME}}  "
            f"{'Metric':<{_COL_METRIC}}  "
            f"{'Value':>{_COL_VALUE}}  "
            f"{'Duration':>{_COL_DURATION}}"
        )

        lines: list[str] = [
            separator,
            "  Mnemon Benchmark Suite Summary",
            separator,
            header,
            separator,
        ]

        for result in results:
            bench_label = result.benchmark_name

            if not result.metrics and "error" in result.details:
                lines.append(
                    f"{bench_label:<{_COL_NAME}}  "
                    f"{'ERROR':<{_COL_METRIC}}  "
                    f"{'---':>{_COL_VALUE}}  "
                    f"{'---':>{_COL_DURATION}}"
                )
                lines.append(
                    f"{'':^{_COL_NAME}}  "
                    f"{result.details['error'][:_COL_METRIC + _COL_VALUE + 4]}"
                )
                lines.append(separator)
                continue

            first_row = True
            for metric, value in result.metrics.items():
                duration_cell = f"{result.duration_ms:.1f} ms" if first_row else ""
                label_cell = bench_label if first_row else ""
                lines.append(
                    f"{label_cell:<{_COL_NAME}}  "
                    f"{metric:<{_COL_METRIC}}  "
                    f"{value:>{_COL_VALUE}.4f}  "
                    f"{duration_cell:>{_COL_DURATION}}"
                )
                first_row = False

            if not result.metrics:
                lines.append(
                    f"{bench_label:<{_COL_NAME}}  "
                    f"{'(no metrics)':<{_COL_METRIC}}  "
                    f"{'---':>{_COL_VALUE}}  "
                    f"{result.duration_ms:>{_COL_DURATION - 3}.1f} ms"
                )

            lines.append(separator)

        return "\n".join(lines)
