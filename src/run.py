"""
CLI runner for the Lobito Corridor analysis pipeline.

Usage
-----
Run all steps for a single AOI:
    python src/run.py --aoi huambo

Run specific steps:
    python src/run.py --aoi huambo --steps 0,1,2,7

Run all steps for multiple AOIs:
    python src/run.py --aoi benguela,bie,huambo,moxico,moxicoleste

Run a range of steps:
    python src/run.py --aoi huambo --steps 0-7

Environment
-----------
Sets PROJECT_ROOT and AOI env vars, then imports and runs each step's ``main()``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path


# Step modules in execution order
STEP_MODULES = {
    0:  "step_00_align_and_rasterize",
    1:  "step_01_isochrones",
    2:  "step_02_kpis_population_cropland_electric",
    3:  "step_03_priority_surface",
    4:  "step_04_flood_bottlenecks_from_road_raster",
    5:  "step_05_site_audit_points",
    6:  "step_06_muni_ingest",
    7:  "step_07_priority_tunable",
    8:  "step_08_project_kpis",
    9:  "step_09_muni_targeting",
    10: "step_10_priority_scenarios",
    11: "step_11_priority_clusters",
    12: "step_12_traveltime_catchments",
    13: "step_13_synergies_overlay",
    14: "step_14_lite_od",
    15: "step_15_corridor_dashboard",
    16: "step_16_intervention_simulator",
}

ALL_STEPS = sorted(STEP_MODULES.keys())


def _parse_steps(spec: str | None) -> list[int]:
    """
    Parse a step specification string into a sorted list of step numbers.

    Accepts:
      - None or "all" → all steps
      - "0,1,2,7"     → specific steps
      - "0-7"         → range (inclusive)
      - "0-7,14"      → range + individual
    """
    if spec is None or spec.strip().lower() == "all":
        return ALL_STEPS

    steps: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                if i in STEP_MODULES:
                    steps.add(i)
        else:
            i = int(part)
            if i in STEP_MODULES:
                steps.add(i)
            else:
                print(f"Warning: step {i} not found, skipping.")
    return sorted(steps)


def _run_aoi(aoi: str, steps: list[int]) -> None:
    """Run the given steps for a single AOI."""
    os.environ["AOI"] = aoi.lower().replace(" ", "-")

    # Force reload of config so it picks up the new AOI
    importlib.invalidate_caches()
    if "config" in sys.modules:
        del sys.modules["config"]

    # Also clear cached step modules so they re-import config
    for mod_name in STEP_MODULES.values():
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    print(f"\n{'='*60}")
    print(f"  AOI: {os.environ['AOI']}  |  Steps: {steps}")
    print(f"{'='*60}")

    for step_num in steps:
        mod_name = STEP_MODULES[step_num]
        print(f"\n--- Step {step_num:02d}: {mod_name} ---")
        t0 = time.time()
        try:
            mod = importlib.import_module(mod_name)
            mod.main()
        except Exception as e:
            print(f"  ERROR in step {step_num:02d}: {e}")
            raise
        elapsed = time.time() - t0
        print(f"  Step {step_num:02d} done in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Lobito Corridor spatial analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/run.py --aoi huambo
  python src/run.py --aoi huambo --steps 0-7
  python src/run.py --aoi benguela,bie,huambo,moxico,moxicoleste --steps 0,7,11
        """,
    )
    parser.add_argument(
        "--aoi",
        required=True,
        help="Comma-separated AOI(s), e.g. 'huambo' or 'benguela,bie,huambo'",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Steps to run: 'all' (default), '0,1,2,7', '0-7', or '0-7,14'",
    )
    args = parser.parse_args()

    # Ensure src/ is on path
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJECT_ROOT", str(project_root))
    src_dir = str(project_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    aois = [a.strip() for a in args.aoi.split(",") if a.strip()]
    steps = _parse_steps(args.steps)

    print(f"Lobito Corridor Pipeline")
    print(f"AOIs:  {aois}")
    print(f"Steps: {steps}")

    t_total = time.time()
    for aoi in aois:
        _run_aoi(aoi, steps)
    elapsed_total = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"All done | {len(aois)} AOI(s) x {len(steps)} step(s) in {elapsed_total:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
