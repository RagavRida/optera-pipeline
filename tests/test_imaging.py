"""Tests for optera.imaging — file type sniffing, hashing, quality metrics."""
from __future__ import annotations

import io
import struct
import pytest
from pathlib import Path
from PIL import Image

from optera.imaging import (
    sniff_type, load, encode, dhash, hamming, blur_score, brightness,
)


class TestSniffType:
    """Content-based file type detection."""

    def test_jpeg(self, tmp_image):
        assert sniff_type(tmp_image) == "image/jpeg"

    def test_png(self, tmp_png):
        assert sniff_type(tmp_png) == "image/png"

    def test_html_as_jpg(self, tmp_html_as_jpg):
        """The classic WhatsApp failure: an HTML error page saved as .jpg."""
        assert sniff_type(tmp_html_as_jpg) is None

    def test_gif(self, tmp_path):
        path = tmp_path / "test.gif"
        img = Image.new("RGB", (10, 10), color="green")
        img.save(path, "GIF")
        assert sniff_type(path) == "image/gif"

    def test_bmp(self, tmp_path):
        path = tmp_path / "test.bmp"
        img = Image.new("RGB", (10, 10), color="yellow")
        img.save(path, "BMP")
        assert sniff_type(path) == "image/bmp"

    def test_random_bytes(self, tmp_path):
        path = tmp_path / "random.bin"
        path.write_bytes(b"\x00\x01\x02\x03" * 8)
        assert sniff_type(path) is None

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.dat"
        path.write_bytes(b"")
        assert sniff_type(path) is None

    def test_webp(self, tmp_path):
        path = tmp_path / "test.webp"
        img = Image.new("RGB", (10, 10), color="purple")
        img.save(path, "WEBP")
        assert sniff_type(path) == "image/webp"


class TestLoad:
    """Image loading with defect repair."""

    def test_basic_load(self, tmp_image):
        li = load(tmp_image)
        assert li.width == 100
        assert li.height == 100
        assert li.bytes_on_disk > 0
        assert len(li.sha256) == 64
        assert li.truncated is False

    def test_sha256_deterministic(self, tmp_image):
        a = load(tmp_image)
        b = load(tmp_image)
        assert a.sha256 == b.sha256

    def test_megapixels(self, tmp_image):
        li = load(tmp_image)
        assert li.megapixels == pytest.approx(0.01, abs=0.001)


class TestEncode:
    """Image encoding to base64 for API calls."""

    def test_downscale(self, tmp_image):
        li = load(tmp_image)
        media, b64, nbytes = encode(li.image, max_dim=50)
        assert media == "image/jpeg"
        assert len(b64) > 0
        assert nbytes > 0
        # Decode back and check dimensions
        decoded = Image.open(io.BytesIO(__import__("base64").b64decode(b64)))
        assert max(decoded.size) <= 50

    def test_no_upscale(self, tmp_image):
        """Images smaller than max_dim should not be upscaled."""
        li = load(tmp_image)
        media, b64, nbytes = encode(li.image, max_dim=500)
        decoded = Image.open(io.BytesIO(__import__("base64").b64decode(b64)))
        assert max(decoded.size) <= 100  # original is 100x100


class TestDhash:
    """Perceptual hashing for deduplication."""

    def test_identical_images_zero_hamming(self):
        img = Image.new("RGB", (100, 100), color="red")
        h1 = dhash(img)
        h2 = dhash(img)
        assert hamming(h1, h2) == 0

    def test_very_different_images_high_hamming(self):
        white = Image.new("RGB", (100, 100), color="white")
        black = Image.new("RGB", (100, 100), color="black")
        h1 = dhash(white)
        h2 = dhash(black)
        # Solid images have no edges, so their hashes may be similar.
        # Use structured images instead.
        from PIL import ImageDraw
        a = Image.new("RGB", (100, 100), "white")
        d = ImageDraw.Draw(a)
        d.rectangle([0, 0, 50, 100], fill="black")

        b = Image.new("RGB", (100, 100), "white")
        d = ImageDraw.Draw(b)
        d.rectangle([50, 0, 100, 50], fill="black")

        ha = dhash(a)
        hb = dhash(b)
        assert hamming(ha, hb) > 5

    def test_hamming_symmetry(self):
        img1 = Image.new("RGB", (100, 100), "red")
        img2 = Image.new("RGB", (100, 100), "blue")
        h1, h2 = dhash(img1), dhash(img2)
        assert hamming(h1, h2) == hamming(h2, h1)

    def test_hamming_identity(self):
        h = dhash(Image.new("RGB", (50, 50), "green"))
        assert hamming(h, h) == 0


class TestQualityMetrics:
    """Blur and brightness scoring."""

    def test_blur_score_positive(self, tmp_image):
        li = load(tmp_image)
        score = blur_score(li.image)
        assert score >= 0

    def test_brightness_range(self, tmp_image):
        li = load(tmp_image)
        b = brightness(li.image)
        assert 0 <= b <= 255

    def test_dark_image_low_brightness(self, tmp_path):
        path = tmp_path / "dark.jpg"
        img = Image.new("RGB", (100, 100), color=(10, 10, 10))
        img.save(path, "JPEG")
        li = load(path)
        b = brightness(li.image)
        assert b < 50

    def test_bright_image_high_brightness(self, tmp_path):
        path = tmp_path / "bright.jpg"
        img = Image.new("RGB", (100, 100), color=(240, 240, 240))
        img.save(path, "JPEG")
        li = load(path)
        b = brightness(li.image)
        assert b > 200
