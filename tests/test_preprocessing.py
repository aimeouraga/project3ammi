import pytest
from data.preprocessing import clean_text

def test_preprocessing():
    sample_text = "This is a <b>great</b> movie!"
    cleaned_text = clean_text(sample_text).split()
    assert "great" in cleaned_text, "Stop words should be removed"
    assert "<b>" not in cleaned_text, "HTML tags should be removed"