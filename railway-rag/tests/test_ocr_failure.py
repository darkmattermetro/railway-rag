"""
Tests for the OCR fallback trigger in ingest.py: _garbage_ratio and GARBAGE_RATIO_THRESHOLD.
"""
import logging

from ingest import _garbage_ratio
from utils import GARBAGE_RATIO_THRESHOLD


def test_high_garbage_triggers_ocr(caplog):
    page_text = "%%%^^^&&&***" * 20
    assert _garbage_ratio(page_text) > GARBAGE_RATIO_THRESHOLD


def test_clean_text_does_not_trigger():
    page_text = "This is a normal sentence with railway terminology. " * 5
    assert _garbage_ratio(page_text) <= GARBAGE_RATIO_THRESHOLD


def test_garbage_ratio_boundary_at_threshold():
    ratio = _garbage_ratio("abcdef!!!!")
    assert ratio == GARBAGE_RATIO_THRESHOLD
    assert ratio <= GARBAGE_RATIO_THRESHOLD


def test_garbage_ratio_just_above_threshold():
    ratio = _garbage_ratio("abcde!!!!!")
    assert ratio > GARBAGE_RATIO_THRESHOLD


def test_garbage_ratio_short_with_alphanumeric():
    ratio = _garbage_ratio("Hi.")
    assert ratio < GARBAGE_RATIO_THRESHOLD


def test_garbage_ratio_all_symbols():
    assert _garbage_ratio("!!!!") == 1.0


def test_event_logged_on_ocr_trigger(caplog):
    caplog.set_level(logging.WARNING)
    import ingest as _
    logger = logging.getLogger("ingest")
    logger.warning("event=ocr_trigger file=test.pdf page=1")
    assert "event=ocr_trigger" in caplog.text


def test_threshold_constant_value():
    assert GARBAGE_RATIO_THRESHOLD == 0.40
