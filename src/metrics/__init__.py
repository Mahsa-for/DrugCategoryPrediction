"""
Metrics package for drug prediction evidence assessment.

Provides tools to evaluate and explain drug category predictions
based on molecular evidence from gene expression data.
"""

from .brain_evidence import BrainEvidenceMetrics

__all__ = ['BrainEvidenceMetrics']
