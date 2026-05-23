"""
Tests for the OCR silent failure detection logic from local_builder.py (§2.9)
"""
from ingest import (
    _should_trigger_ocr_fallback,
    OCR_MIN_TEXT_LEN,
    OCR_GARBAGE_RATIO_THRESHOLD,
)


def test_short_text_triggers_fallback():
    """Test that short text triggers OCR fallback."""
    page_text = "!!!"  # Short (3 chars < 20), no alphanumeric
    assert _should_trigger_ocr_fallback(page_text) is True


def test_long_clean_text_does_not_trigger():
    """Test that long clean text does not trigger OCR fallback."""
    page_text = "A" * 500  # 500 alphanumeric chars
    assert _should_trigger_ocr_fallback(page_text) is False


def test_garbage_ratio_calculation_high():
    """Test garbage ratio calculation for high garbage content."""
    page_text = "%%%^^^&&&***" * 20  # entirely non-alphanumeric
    assert _should_trigger_ocr_fallback(page_text) is True


def test_garbage_ratio_calculation_low():
    """Test garbage ratio calculation for low garbage content."""
    page_text = "This is a normal sentence with railway terminology. " * 5
    assert _should_trigger_ocr_fallback(page_text) is False


def test_garbage_ratio_boundary_at_threshold():
    """Test boundary condition where garbage ratio equals threshold."""
    # garbage_ratio = 0.4 exactly (6 alnum, 4 non-alnum)
    page_text = "abcdef!!!!"  # 6 alnum, 4 non-alnum = 0.4 garbage ratio
    # Since condition is ratio > threshold (not >=), this should NOT trigger
    assert _should_trigger_ocr_fallback(page_text) is False

    # Test with slightly more garbage to trigger
    page_text_trigger = "abcde!!!!!"  # 5 alnum, 5 non-alnum = 0.5 garbage ratio
    assert _should_trigger_ocr_fallback(page_text_trigger) is True


def test_both_conditions_independent():
    """Test that each condition can independently trigger fallback."""
    # Short text with no alphanumeric should trigger (condition 1)
    short_text = "!!!"  # Less than 20 chars, no alphanumeric
    assert _should_trigger_ocr_fallback(short_text) is True

    # Long text with high garbage ratio should trigger (condition 2 alone)
    long_garbage_text = "!!!!" * 30  # 120 chars, 0% alphanumeric
    assert _should_trigger_ocr_fallback(long_garbage_text) is True


def test_short_text_with_alphanumeric_does_not_trigger():
    """Test that short text with alphanumeric does NOT trigger via condition 1.

    Condition 1 requires BOTH short length AND no alphanumeric characters.
    A short string that contains letters should fall through to garbage check.
    """
    page_text = "Hi."  # 3 chars < 20, but contains alphanumeric
    # garbage: clean = "Hi", ratio = 1 - 2/3 = 0.33..., not > 0.4
    assert _should_trigger_ocr_fallback(page_text) is False


def test_imported_constants_match():
    """Verify that OCR_MIN_TEXT_LEN is 20 as used by local_builder."""
    assert OCR_MIN_TEXT_LEN == 20
    assert OCR_GARBAGE_RATIO_THRESHOLD == 0.4
