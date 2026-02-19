"""
Smoke tests for the Lobito Corridor analysis pipeline.

These tests verify that:
- Core modules import without errors
- Key utility functions work on synthetic data
- The CLI runner parses arguments correctly

They do NOT require real geospatial data on disk.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Set minimal env so config.py doesn't fail on missing files
os.environ.setdefault("PROJECT_ROOT", str(_ROOT))
os.environ.setdefault("AOI", "test")


# ---------------------------------------------------------------
# 1. Import smoke tests
# ---------------------------------------------------------------

class TestImports:
    """All step modules and utilities should import without error."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "config",
            "utils_geo",
            "run",
            "step_00_align_and_rasterize",
            "step_01_isochrones",
            "step_02_kpis_population_cropland_electric",
            "step_03_priority_surface",
            "step_04_flood_bottlenecks_from_road_raster",
            "step_05_site_audit_points",
            "step_06_muni_ingest",
            "step_07_priority_tunable",
            "step_08_project_kpis",
            "step_09_muni_targeting",
            "step_10_priority_scenarios",
            "step_11_priority_clusters",
            "step_12_traveltime_catchments",
            "step_13_synergies_overlay",
            "step_14_lite_od",
            "step_15_corridor_dashboard",
            "step_16_intervention_simulator",
        ],
    )
    def test_import(self, module_name):
        mod = importlib.import_module(module_name)
        assert mod is not None


# ---------------------------------------------------------------
# 2. utils_geo unit tests
# ---------------------------------------------------------------

class TestUtilsGeo:
    """Test key utility functions with synthetic data."""

    def test_ensure_aligned_passthrough(self):
        """If grids already match, ensure_aligned returns the original array."""
        import xarray as xr
        import rioxarray  # noqa: F401  (registers .rio accessor)

        # Create a small synthetic raster
        data = np.random.rand(10, 10).astype("float32")
        da = xr.DataArray(data, dims=("y", "x"),
                          coords={"y": np.arange(10), "x": np.arange(10)})
        da.rio.write_crs("EPSG:4326", inplace=True)
        da.rio.write_transform(inplace=True)

        from utils_geo import ensure_aligned
        result = ensure_aligned(da, da)
        # Should be the same object (no reproject needed)
        assert result is da

    def test_ensure_aligned_none_input(self):
        """ensure_aligned(None, ...) should return None."""
        import xarray as xr

        template = xr.DataArray(np.zeros((5, 5)), dims=("y", "x"))
        from utils_geo import ensure_aligned
        assert ensure_aligned(None, template) is None

    def test_cell_area_km2_latlon(self):
        """Cell area computation should return reasonable values for 1-km cells."""
        import xarray as xr
        import rioxarray  # noqa: F401

        from rasterio.transform import from_bounds

        # ~1 km cells near equator (0.00833 deg ≈ 1 km)
        ny, nx = 10, 10
        res = 0.00833
        transform = from_bounds(0, 0, nx * res, ny * res, nx, ny)

        da = xr.DataArray(
            np.ones((ny, nx), dtype="float32"),
            dims=("y", "x"),
            coords={
                "y": np.linspace(ny * res - res / 2, res / 2, ny),
                "x": np.linspace(res / 2, nx * res - res / 2, nx),
            },
        )
        da.rio.write_crs("EPSG:4326", inplace=True)
        da.rio.write_transform(transform, inplace=True)

        from utils_geo import cell_area_km2_latlon
        area = cell_area_km2_latlon(da)

        # Near equator, ~0.00833 deg ≈ ~0.86 km² per cell
        mean_area = float(area.values.mean())
        assert 0.5 < mean_area < 1.5, f"Expected ~0.86 km², got {mean_area:.3f}"

    def test_focal_mean(self):
        """Focal mean with radius=1 should smooth the array."""
        import xarray as xr

        data = np.zeros((11, 11), dtype="float32")
        data[5, 5] = 100.0  # single bright pixel
        da = xr.DataArray(data, dims=("y", "x"))

        from utils_geo import focal_mean
        smoothed = focal_mean(da, radius=1)

        # Center should be lower after smoothing
        assert float(smoothed.values[5, 5]) < 100.0
        # Neighbors should be > 0
        assert float(smoothed.values[5, 6]) > 0


# ---------------------------------------------------------------
# 3. CLI runner tests
# ---------------------------------------------------------------

class TestCLIRunner:
    """Test the run.py argument parsing."""

    def test_parse_steps_all(self):
        from run import _parse_steps, ALL_STEPS
        assert _parse_steps(None) == ALL_STEPS
        assert _parse_steps("all") == ALL_STEPS

    def test_parse_steps_specific(self):
        from run import _parse_steps
        assert _parse_steps("0,1,7") == [0, 1, 7]

    def test_parse_steps_range(self):
        from run import _parse_steps
        result = _parse_steps("0-3")
        assert result == [0, 1, 2, 3]

    def test_parse_steps_mixed(self):
        from run import _parse_steps
        result = _parse_steps("0-2,7,14")
        assert result == [0, 1, 2, 7, 14]
