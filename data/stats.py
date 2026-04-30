"""
Dataset statistics: collection, reporting, and persistence.

Statistics are computed after preprocessing completes and saved as
{output_root}/{dataset_name}/stats.json  (per-dataset)
{output_root}/combined_stats.json        (all datasets together)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DatasetStats:
    dataset_name: str
    total_frames: int = 0
    skipped_frames: int = 0             # already existed on disk
    frames_per_group: Dict[str, int] = field(default_factory=dict)  # e.g. per-task
    processing_time_seconds: float = 0.0
    output_size_bytes: int = 0

    # Derived
    @property
    def written_frames(self) -> int:
        return self.total_frames - self.skipped_frames

    @property
    def num_groups(self) -> int:
        return len(self.frames_per_group)

    @property
    def output_size_gb(self) -> float:
        return self.output_size_bytes / 1024 ** 3


def _dir_size(path: Path) -> int:
    """Recursively sum file sizes under path (bytes)."""
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


def compute_output_size(output_root: Path, dataset_name: str) -> int:
    return _dir_size(output_root / dataset_name)


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m}m {s}s"
    elif m:
        return f"{m}m {s}s"
    return f"{s}s"


def print_dataset_stats(stats: DatasetStats) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Dataset : {stats.dataset_name}")
    print(sep)
    print(f"  Total frames    : {stats.total_frames:>12,}")
    print(f"  Written         : {stats.written_frames:>12,}")
    print(f"  Skipped (cached): {stats.skipped_frames:>12,}")
    print(f"  Groups          : {stats.num_groups:>12,}")
    print(f"  Processing time : {format_time(stats.processing_time_seconds):>12}")
    print(f"  Output size     : {stats.output_size_gb:>11.2f} GB")

    if stats.frames_per_group:
        print(f"\n  Per-group breakdown (top 20):")
        sorted_groups = sorted(stats.frames_per_group.items(), key=lambda x: -x[1])
        for name, count in sorted_groups[:20]:
            pct = 100 * count / stats.total_frames if stats.total_frames else 0
            print(f"    {name:<40s} {count:>8,}  ({pct:5.1f}%)")
        if len(sorted_groups) > 20:
            print(f"    ... and {len(sorted_groups) - 20} more groups")
    print(sep)


def print_combined_stats(all_stats: List[DatasetStats]) -> None:
    total_frames = sum(s.total_frames for s in all_stats)
    total_written = sum(s.written_frames for s in all_stats)
    total_gb = sum(s.output_size_gb for s in all_stats)

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  COMBINED STATISTICS  ({len(all_stats)} dataset(s))")
    print(sep)
    for s in all_stats:
        pct = 100 * s.total_frames / total_frames if total_frames else 0
        print(f"  {s.dataset_name:<35s}  {s.total_frames:>10,}  ({pct:5.1f}%)")
    print(f"  {'─' * 55}")
    print(f"  {'TOTAL':<35s}  {total_frames:>10,}  (100.0%)")
    print(f"\n  Total written this run : {total_written:,}")
    print(f"  Total output size      : {total_gb:.2f} GB")
    print(sep)


def save_stats(stats: DatasetStats, output_root: Path) -> None:
    """Save per-dataset stats to {output_root}/{dataset_name}/stats.json."""
    out_dir = output_root / stats.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stats.json", "w") as f:
        json.dump(asdict(stats), f, indent=2)


def save_combined_stats(all_stats: List[DatasetStats], output_root: Path) -> None:
    """Save combined stats to {output_root}/combined_stats.json."""
    payload = {
        "datasets": [asdict(s) for s in all_stats],
        "total_frames": sum(s.total_frames for s in all_stats),
        "total_written": sum(s.written_frames for s in all_stats),
        "total_output_size_gb": sum(s.output_size_gb for s in all_stats),
        "num_datasets": len(all_stats),
    }
    with open(output_root / "combined_stats.json", "w") as f:
        json.dump(payload, f, indent=2)
