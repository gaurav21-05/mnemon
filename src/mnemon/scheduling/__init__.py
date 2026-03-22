"""
Mnemon scheduling subsystem — automatic consolidation triggering.

Exports :class:`ConsolidationScheduler` when APScheduler 4.x is installed.
If APScheduler is absent the module still imports cleanly; attempting to
call :meth:`ConsolidationScheduler.start` will raise ``RuntimeError`` with
a clear installation hint.
"""

from mnemon.scheduling.scheduler import ConsolidationScheduler

__all__ = ["ConsolidationScheduler"]
