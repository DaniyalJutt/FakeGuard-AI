"""
Fake Review Detection Pipeline
Web App Integration Module
"""

from .scraper import scrape_and_preprocess
from .scorer import compute_scores
from .detector import FakeReviewDetector
from .reporter import generate_report

__all__ = [
    'scrape_and_preprocess',
    'compute_scores',
    'FakeReviewDetector',
    'generate_report'
]

