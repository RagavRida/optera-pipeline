"""Stage 0 - the free gate.

Everything here costs zero tokens. Its whole job is to reduce the number of
images that reach a paid model, and to attach cheap signals that make the paid
calls smaller. On the starter set this alone removes one file entirely (an HTML
error page wearing a .jpg extension) and collapses re-sent duplicates.

Order matters: reject the impossible, repair the damaged, deduplicate the
redundant, then measure quality on what survives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

from . import imaging
from .config import BLUR_VAR_FLOOR, DARK_MEAN_FLOOR, PHASH_THRESHOLD


@dataclass
class PreflightResult:
    doc_id: str
    path: Path
    ok: bool                       # may proceed to a paid stage
    reason: str = ""               # populated when ok is False
    sha256: str = ""
    dhash: int | None = None
    duplicate_of: str | None = None
    width: int = 0
    height: int = 0
    bytes_on_disk: int = 0
    truncated: bool = False
    blur: float | None = None
    brightness: float | None = None
    warnings: list[str] = field(default_factory=list)
    image: object = field(default=None, repr=False)

    def to_json(self) -> dict:
        return {
            "doc_id": self.doc_id, "ok": self.ok, "reason": self.reason,
            "sha256": self.sha256[:16], "dhash": (f"{self.dhash:016x}" if self.dhash else None),
            "duplicate_of": self.duplicate_of,
            "dimensions": [self.width, self.height], "bytes": self.bytes_on_disk,
            "truncated": self.truncated,
            "blur": round(self.blur, 1) if self.blur is not None else None,
            "brightness": round(self.brightness, 1) if self.brightness is not None else None,
            "warnings": self.warnings,
        }


def run(paths: list[Path], dedupe: bool = True) -> list[PreflightResult]:
    results: list[PreflightResult] = []
    seen_sha: dict[str, str] = {}
    seen_dhash: list[tuple[int, str]] = []

    for path in sorted(paths):
        doc_id = path.stem
        mime = imaging.sniff_type(path)

        # 1. Not an image at all -> reject for free.
        if mime is None:
            with path.open("rb") as _fh:
                head = _fh.read(64)
            hint = "html_error_page" if b"<!doctype" in head.lower() or b"<html" in head.lower() else "unrecognised_format"
            results.append(PreflightResult(
                doc_id=doc_id, path=path, ok=False,
                reason=f"not_an_image:{hint}", bytes_on_disk=path.stat().st_size))
            logger.info("preflight reject %s: not_an_image:%s", doc_id, hint)
            continue

        if mime == "image/heic":
            results.append(PreflightResult(
                doc_id=doc_id, path=path, ok=False,
                reason="unsupported_format:heic_needs_transcode",
                bytes_on_disk=path.stat().st_size))
            continue

        # 2. Decode, repairing truncation and EXIF rotation.
        try:
            li = imaging.load(path)
        except Exception as exc:
            results.append(PreflightResult(
                doc_id=doc_id, path=path, ok=False,
                reason=f"undecodable:{type(exc).__name__}",
                bytes_on_disk=path.stat().st_size))
            continue

        res = PreflightResult(
            doc_id=doc_id, path=path, ok=True, sha256=li.sha256,
            width=li.width, height=li.height, bytes_on_disk=li.bytes_on_disk,
            truncated=li.truncated, image=li.image)
        if li.truncated:
            res.warnings.append("truncated_file_recovered")

        # 3. Deduplicate. Byte-identical first, then perceptual for re-encodes.
        if dedupe:
            if li.sha256 in seen_sha:
                res.ok = False
                res.reason = "duplicate:exact"
                res.duplicate_of = seen_sha[li.sha256]
                results.append(res)
                logger.info("preflight dedup %s: exact match of %s", doc_id, res.duplicate_of)
                continue
            h = imaging.dhash(li.image)
            res.dhash = h
            near = next((d for hv, d in seen_dhash if imaging.hamming(hv, h) <= PHASH_THRESHOLD), None)
            if near is not None:
                res.ok = False
                res.reason = "duplicate:near"
                res.duplicate_of = near
                results.append(res)
                logger.info("preflight dedup %s: near match of %s", doc_id, near)
                continue
            seen_sha[li.sha256] = doc_id
            seen_dhash.append((h, doc_id))

        # 4. Cheap quality signals. These do not reject - a blurry odometer is
        #    still a real reading - but they justify spending more resolution.
        res.blur = imaging.blur_score(li.image)
        res.brightness = imaging.brightness(li.image)
        if res.blur < BLUR_VAR_FLOOR:
            res.warnings.append("low_sharpness")
        if res.brightness < DARK_MEAN_FLOOR:
            res.warnings.append("underexposed")
        if li.megapixels < 0.15:
            res.warnings.append("very_low_resolution")

        results.append(res)

    return results
