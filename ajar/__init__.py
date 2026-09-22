"""lp-bench: a least-privilege benchmark harness.

Takes a security benchmark's tasks + ground truth, generates probes (candidate
tool calls with oracle-assigned ALLOW/DENY/UNKNOWN labels and authored harm
tiers), runs access-control defenses against them, and measures how far each
defense is from least privilege (its permitted blast radius), decomposed by axis.

Two extension seams:
  * BenchmarkAdapter  (ajar.adapters.base)  — add a benchmark
  * Defense           (ajar.defenses.base)  — add a system under test
The LLM is an injected, optional dependency (ajar.llm.base): it PROPOSES
probes, the oracle DISPOSES labels, so correctness never depends on it.
"""

__version__ = "0.0.1"
