"""Image loading, defect repair, encoding and quality metrics.

Deliberately tolerant on input: this is a WhatsApp inbox, so files arrive
truncated, mislabelled, rotated by EXIF, and occasionally are not images at all.
Every repair that happens here is a repair we never pay a model to work around.
"""
from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFile, ImageFilter, ImageOps

# Real inboxes contain half-transferred files. Decode what is there rather than
# discarding an otherwise readable invoice over four missing bytes.
ImageFile.LOAD_TRUNCATED_IMAGES = True

_MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"BM": "image/bmp",
}


def sniff_type(path: Path) -> str | None:
    """Identify by content, never by extension.

    The dataset ships an HTML error page named .jpg; trusting the extension
    would send a Cloudflare 404 to a vision model at full price.
    """
    head = path.open("rb").read(32)
    for magic, mime in _MAGIC.items():
        if head.startswith(magic):
            return mime
    if head[4:8] == b"ftyp":
        brand = head[8:12].decode("ascii", "replace")
        return "image/heic" if brand.startswith(("heic", "heix", "mif1", "hevc")) else None
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


@dataclass
class LoadedImage:
    path: Path
    image: Image.Image
    width: int
    height: int
    bytes_on_disk: int
    truncated: bool = False
    sha256: str = ""

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000


def load(path: Path) -> LoadedImage:
    raw = path.read_bytes()
    truncated = False
    im = Image.open(io.BytesIO(raw))
    try:
        im.load()
    except OSError:
        truncated = True  # LOAD_TRUNCATED_IMAGES already salvaged the partial frame
    im = ImageOps.exif_transpose(im)      # phone photos are frequently EXIF-rotated
    im = im.convert("RGB")
    return LoadedImage(
        path=path, image=im, width=im.width, height=im.height,
        bytes_on_disk=len(raw), truncated=truncated,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def encode(im: Image.Image, max_dim: int, quality: int = 80) -> tuple[str, str, int]:
    """Downscale to a long edge of max_dim and JPEG-encode to base64.

    Returns (media_type, b64, encoded_bytes). Vision models bill by pixel area,
    so the long edge is the single most direct cost control available.
    """
    work = im.copy()
    if max(work.size) > max_dim:
        work.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    work.save(buf, "JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    return "image/jpeg", base64.b64encode(data).decode(), len(data)


def estimate_image_tokens(width: int, height: int, max_dim: int) -> int:
    """Anthropic's documented approximation: tokens ~= (w*h)/750 after scaling.

    Used only for *planning* a resolution choice. Reported costs always come
    from API-returned counts, never from this function.
    """
    scale = min(1.0, max_dim / max(width, height))
    return int((width * scale) * (height * scale) / 750)


# ------------------------------------------------------------ quality gates --
def dhash(im: Image.Image, size: int = 8) -> int:
    """Difference hash - robust to the re-encoding WhatsApp applies on forward.

    Exact byte hashing misses genuine duplicates because every forward produces
    new bytes; dhash compares structure instead.
    """
    small = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(small.getdata())
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | int(px[base + col] < px[base + col + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def blur_score(im: Image.Image) -> float:
    """Variance of the Laplacian. Low variance means few sharp edges: out of focus."""
    g = im.convert("L")
    if max(g.size) > 640:
        g.thumbnail((640, 640), Image.LANCZOS)
    lap = g.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1))
    px = list(lap.getdata())
    n = len(px)
    mean = sum(px) / n
    return sum((p - mean) ** 2 for p in px) / n


def brightness(im: Image.Image) -> float:
    g = im.convert("L")
    if max(g.size) > 320:
        g.thumbnail((320, 320), Image.LANCZOS)
    px = list(g.getdata())
    return sum(px) / len(px)
