"""Rendering the third metric as a leaderboard.

Lower LPD = closer to least privilege. The per-axis columns are the interpretive
payload: they say WHERE a defense leaks, not just how much. A defense that fails
the sufficiency gate (deny-everything) shows LPD as `gate-fail`, never as a low
number, so it can't masquerade as tight.
"""
from __future__ import annotations

import math

from ajar.score.aggregate import AggregateScore

_AXES = ["TOOL", "PARAMETER", "SEQUENCE", "PROVENANCE"]


def _fmt(x: float) -> str:
    return "  —  " if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.3f}"


def leaderboard(aggs: list[AggregateScore]) -> str:
    # Tighter (lower LPD) first; gate-failing defenses (nan LPD) sort to the bottom.
    ordered = sorted(aggs, key=lambda a: (math.isnan(a.lpd), a.lpd if not math.isnan(a.lpd) else 0.0))

    head = (
        f"{'defense':16s} {'cover':>6s} {'gate%':>6s} {'LPD':>7s}  "
        + "".join(f"{ax[:5]:>7s}" for ax in _AXES)
        + f"  {'o-rst':>6s} {'attacks':>9s}"
    )
    lines = [head, "-" * len(head)]
    for a in ordered:
        lpd = "gate-fail" if math.isnan(a.lpd) else f"{a.lpd:.3f}"
        axis_cols = "".join(f"{_fmt(a.lpd_by_axis.get(ax, math.nan)):>7s}" for ax in _AXES)
        atk = f"{a.attacks_admitted}/{a.attacks_total}"
        lines.append(
            f"{a.defense:16s} {a.n_scored:>2d}/{a.n_total:<3d} "
            f"{a.gate_pass_rate*100:>5.0f}% {lpd:>7s}  {axis_cols}  "
            f"{a.over_restriction:>6.2f} {atk:>9s}"
        )
    lines.append("")
    lines.append("LPD = harm-weighted over-privilege, macro-averaged over gate-passing tasks (lower = tighter).")
    lines.append("Per-axis columns show which axis each defense leaves open. o-rst = permissible reads denied.")
    return "\n".join(lines)
