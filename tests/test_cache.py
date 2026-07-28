from __future__ import annotations

from pathlib import Path

from PIL import Image

from optera.cache import ResultCache
from optera.preflight import run
from optera.schemas import empty_envelope, refusal


def _preflight(tmp_path: Path):
    image = tmp_path / "doc.jpg"
    Image.new("RGB", (48, 48), "white").save(image)
    return run([image], dedupe=True)[0]


def test_cache_reuses_exact_validated_result_and_resets_provenance(tmp_path):
    pf = _preflight(tmp_path)
    cache = ResultCache(tmp_path / "cache")
    env = empty_envelope(pf.doc_id)
    env.update({"doc_class": "vendor_bill", "status": "extracted", "confidence": 0.9,
                "data": {"total_amount": 100}, "validation": {"passed": True, "issues": []}})
    assert cache.put(pf, env)

    cached = cache.get(pf)
    assert cached is not None
    assert cached["data"] == {"total_amount": 100}
    assert cached["provenance"]["cost_usd"] == 0.0
    assert cached["provenance"]["stages"] == ["cache:exact_validated_hit"]


def test_cache_rejects_unvalidated_low_confidence_result(tmp_path):
    pf = _preflight(tmp_path)
    cache = ResultCache(tmp_path / "cache")
    env = empty_envelope(pf.doc_id)
    env.update({"doc_class": "vendor_bill", "status": "extracted", "confidence": 0.2,
                "data": {}, "validation": {"passed": False, "issues": ["bad"]}})
    assert not cache.put(pf, env)
    assert cache.get(pf) is None


def test_cache_allows_high_confidence_refusal(tmp_path):
    pf = _preflight(tmp_path)
    cache = ResultCache(tmp_path / "cache")
    env = refusal(pf.doc_id, "not_a_document", confidence=0.99)
    assert cache.put(pf, env)
    assert cache.get(pf)["status"] == "refused"
