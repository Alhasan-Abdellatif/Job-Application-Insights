"""Evaluation harness — golden sets, retrieval metrics, baseline runner.

This package answers the only question that matters when you change a
retrieval pipeline: *did it actually get better?*

Layout:

* :mod:`.metrics` — pure functions over (retrieved, relevant). No I/O.
* :mod:`.golden_set` *(coming next)* — typed Pydantic model for the
  ground-truth dataset and JSONL loaders.
* :mod:`.runner` *(coming next)* — glue that runs a retriever against the
  golden set and prints/returns a metric summary.
"""
