"""Shared helper utilities used across the fake-review and sentiment pipelines."""
import re
import string


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip punctuation/extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def load_pickle(path: str):
    import joblib
    return joblib.load(path)
