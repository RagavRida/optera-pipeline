"""Tests for optera.preflight — the free gate."""
from __future__ import annotations

import pytest
from pathlib import Path
from PIL import Image

from optera.imaging import sniff_type, load, dhash, hamming
from optera.preflight import run, PreflightResult


class TestPreflightRun:
    """Integration-level tests for the preflight stage."""

    def test_rejects_html_as_jpg(self, tmp_html_as_jpg, tmp_path):
        results = run([tmp_html_as_jpg], dedupe=False)
        assert len(results) == 1
        assert results[0].ok is False
        assert "not_an_image" in results[0].reason

    def test_accepts_valid_jpeg(self, tmp_image):
        results = run([tmp_image], dedupe=False)
        assert len(results) == 1
        assert results[0].ok is True

    def test_exact_dedup(self, tmp_image, tmp_path):
        """Two copies of the same file should be deduped."""
        import shutil
        copy = tmp_path / "copy.jpg"
        shutil.copy(tmp_image, copy)
        results = run(sorted([tmp_image, copy]), dedupe=True)
        assert len(results) == 2
        passed = [r for r in results if r.ok]
        rejected = [r for r in results if not r.ok]
        assert len(passed) == 1
        assert len(rejected) == 1
        assert "duplicate:exact" in rejected[0].reason

    def test_dedup_disabled(self, tmp_image, tmp_path):
        """With dedup disabled, identical files should both pass."""
        import shutil
        copy = tmp_path / "copy.jpg"
        shutil.copy(tmp_image, copy)
        results = run(sorted([tmp_image, copy]), dedupe=False)
        passed = [r for r in results if r.ok]
        assert len(passed) == 2

    def test_quality_flags_dark_image(self, tmp_path):
        path = tmp_path / "dark.jpg"
        img = Image.new("RGB", (200, 200), color=(5, 5, 5))
        img.save(path, "JPEG")
        results = run([path], dedupe=False)
        assert results[0].ok is True
        assert "underexposed" in results[0].warnings

    def test_low_resolution_flag(self, tmp_path):
        path = tmp_path / "tiny.jpg"
        img = Image.new("RGB", (30, 30), color=(128, 128, 128))
        img.save(path, "JPEG")
        results = run([path], dedupe=False)
        assert results[0].ok is True
        assert "very_low_resolution" in results[0].warnings

    def test_sha256_populated(self, tmp_image):
        results = run([tmp_image], dedupe=False)
        assert len(results[0].sha256) == 64

    def test_dimensions_populated(self, tmp_image):
        results = run([tmp_image], dedupe=False)
        assert results[0].width == 100
        assert results[0].height == 100


class TestPreflightResult:
    """Unit tests for the PreflightResult dataclass."""

    def test_to_json(self, tmp_image):
        results = run([tmp_image], dedupe=False)
        j = results[0].to_json()
        assert j["ok"] is True
        assert j["dimensions"] == [100, 100]
        assert isinstance(j["sha256"], str)
        assert isinstance(j["warnings"], list)

    def test_to_json_rejected(self, tmp_html_as_jpg):
        results = run([tmp_html_as_jpg], dedupe=False)
        j = results[0].to_json()
        assert j["ok"] is False
        assert "not_an_image" in j["reason"]


class TestNearDedup:
    """Perceptual hash deduplication (WhatsApp re-encoding simulation)."""

    def test_slightly_different_quality(self, tmp_path):
        """Same image at different JPEG qualities should be near-duplicates."""
        img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        # Add some structure so dhash can work
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 80, 80], fill="white")
        draw.rectangle([120, 120, 180, 180], fill="black")

        p1 = tmp_path / "q95.jpg"
        p2 = tmp_path / "q50.jpg"
        img.save(p1, "JPEG", quality=95)
        img.save(p2, "JPEG", quality=50)

        results = run(sorted([p1, p2]), dedupe=True)
        # At least one should be deduped (exact or near)
        rejected = [r for r in results if not r.ok]
        assert len(rejected) >= 1

    def test_genuinely_different_images_not_deduped(self, tmp_path):
        from PIL import ImageDraw
        img1 = Image.new("RGB", (200, 200), "white")
        d = ImageDraw.Draw(img1)
        d.rectangle([0, 0, 100, 200], fill="black")

        img2 = Image.new("RGB", (200, 200), "black")
        d = ImageDraw.Draw(img2)
        d.rectangle([100, 0, 200, 100], fill="white")

        p1 = tmp_path / "img1.jpg"
        p2 = tmp_path / "img2.jpg"
        img1.save(p1, "JPEG")
        img2.save(p2, "JPEG")

        results = run(sorted([p1, p2]), dedupe=True)
        passed = [r for r in results if r.ok]
        assert len(passed) == 2
