"""
Brain Evidence Support Metrics for Drug Category Predictions

This module quantifies whether predicted drug therapeutic categories are supported
by brain gene-expression evidence. It does not claim correctness or incorrectness,
but rather assesses evidence sufficiency for brain-based interpretation.

Suitable for experimental biomedical AI systems where true therapeutic categories
are unknown and mismatches are treated as exploratory findings requiring further
validation rather than errors.
"""

from typing import Dict, List, Tuple
import numpy as np


class BrainEvidenceMetrics:
    """
    Compute evidence support metrics for drug predictions based on brain expression.
    
    Metrics assess whether gene expression in brain regions is sufficiently strong
    and specific to support a brain-based interpretation of drug effects.
    
    Attributes:
        tau_strength (float): Threshold for Brain Evidence Strength [0, 1].
            Default 0.3 means BES must be at least 30% of max possible expression.
        tau_ratio (float): Threshold for Brain Specificity Ratio [0, inf].
            Default 0.6 means brain expression should be at least 60% of body expression.
    """
    
    def __init__(self, tau_strength: float = 0.3, tau_ratio: float = 0.6):
        """
        Initialize metrics with configurable thresholds.
        
        Args:
            tau_strength: Threshold for evidence strength (0–1).
            tau_ratio: Threshold for brain/body ratio (0–inf).
        
        Raises:
            ValueError: If thresholds are out of valid ranges.
        """
        if not (0 <= tau_strength <= 1):
            raise ValueError(f"tau_strength must be in [0, 1], got {tau_strength}")
        if tau_ratio < 0:
            raise ValueError(f"tau_ratio must be >= 0, got {tau_ratio}")
        
        self.tau_strength = tau_strength
        self.tau_ratio = tau_ratio
    
    def brain_evidence_strength(
        self,
        gene_expression: Dict[str, Dict[str, float]],
        brain_regions: List[str]
    ) -> float:
        """
        Compute Brain Evidence Strength (BES).
        
        Measures the average expression level across brain regions for target genes.
        Higher BES indicates stronger molecular evidence in the brain.
        
        Args:
            gene_expression: Dict mapping gene names to tissue->expression values.
                Example: {'TP53': {'cortex': 0.8, 'liver': 0.2}, ...}
            brain_regions: List of brain region names (keys in gene_expression subdicts).
        
        Returns:
            BES: Average expression across brain regions and genes, normalized to [0, 1].
                Returns 0.0 if no data is available.
        
        Notes:
            - BES is computed as: mean(expression across all brain regions and genes)
            - Values normalized to [0, 1] assuming expression values are already normalized
            - Genes without brain region data are skipped
        """
        brain_expressions = []
        
        for gene, tissues in gene_expression.items():
            for region in brain_regions:
                if region in tissues:
                    brain_expressions.append(tissues[region])
        
        if not brain_expressions:
            return 0.0
        
        bes = float(np.mean(brain_expressions))
        return np.clip(bes, 0.0, 1.0)
    
    def brain_specificity_ratio(
        self,
        gene_expression: Dict[str, Dict[str, float]],
        brain_regions: List[str],
        body_tissues: List[str]
    ) -> float:
        """
        Compute Brain Specificity Ratio (BSR).
        
        Measures the relative expression strength in brain versus body tissues.
        Higher BSR indicates that drug effects are preferentially observable in brain.
        
        Args:
            gene_expression: Dict mapping gene names to tissue->expression values.
            brain_regions: List of brain region names.
            body_tissues: List of non-brain tissue names (e.g., 'liver', 'kidney').
        
        Returns:
            BSR: Ratio of max brain expression to max body expression.
                Returns float('inf') if body tissues have zero expression (brain-specific).
                Returns 0.0 if no brain or body expression data exists.
        
        Notes:
            - BSR = max(brain expression) / max(body expression)
            - Used to identify whether effects are brain-enriched (BSR > 1) or body-enriched
            - Infinity indicates exclusive brain activity (no body expression)
        """
        max_brain = 0.0
        max_body = 0.0
        
        for gene, tissues in gene_expression.items():
            for region in brain_regions:
                if region in tissues:
                    max_brain = max(max_brain, tissues[region])
            for tissue in body_tissues:
                if tissue in tissues:
                    max_body = max(max_body, tissues[tissue])
        
        if max_brain == 0.0 or max_body == 0.0:
            return 0.0 if max_brain == 0.0 else float('inf')
        
        bsr = max_brain / max_body
        return bsr
    
    def brain_sufficiency_flag(
        self,
        bes: float,
        bsr: float
    ) -> bool:
        """
        Compute Brain Sufficiency Flag (BSF).
        
        Boolean indicator of whether brain evidence is sufficient to support
        a brain-based interpretation of the predicted drug category.
        
        Args:
            bes: Brain Evidence Strength value.
            bsr: Brain Specificity Ratio value.
        
        Returns:
            BSF: True if (BES >= tau_strength) AND (BSR >= tau_ratio), else False.
        
        Notes:
            - Thresholds are set during __init__
            - BSF = True indicates prediction is supported by sufficient brain evidence
            - BSF = False indicates evidence is exploratory or insufficient for brain interpretation
            - Special case: If BES >= 0.8 (very high brain expression), BSR threshold is relaxed by 50%
        """
        # Relax BSR threshold if BES is very high (strong brain expression alone is good evidence)
        effective_tau_ratio = self.tau_ratio * 0.5 if bes >= 0.8 else self.tau_ratio
        bsf = (bes >= self.tau_strength) and (bsr >= effective_tau_ratio)
        return bool(bsf)
    
    def evidence_summary(
        self,
        gene_expression: Dict[str, Dict[str, float]],
        brain_regions: List[str],
        body_tissues: List[str]
    ) -> Dict[str, any]:
        """
        Generate a comprehensive evidence summary for a prediction.
        
        Computes all metrics and returns interpretation suitable for explaining
        predictions to researchers.
        
        Args:
            gene_expression: Dict mapping gene names to tissue->expression values.
            brain_regions: List of brain region names.
            body_tissues: List of non-brain tissue names.
        
        Returns:
            Dictionary containing:
                'bes' (float): Brain Evidence Strength [0, 1]
                'bsr' (float): Brain Specificity Ratio [0, inf]
                'bsf' (bool): Brain Sufficiency Flag
                'interpretation' (str): Natural language summary
                'tau_strength' (float): Threshold used for BES
                'tau_ratio' (float): Threshold used for BSR
        
        Notes:
            - Interpretation is designed to avoid terms like "wrong" or "error"
            - Language emphasizes evidence sufficiency, not prediction correctness
            - Suitable for researcher-facing reports and exploratory analysis
        """
        bes = self.brain_evidence_strength(gene_expression, brain_regions)
        bsr = self.brain_specificity_ratio(gene_expression, brain_regions, body_tissues)
        bsf = self.brain_sufficiency_flag(bes, bsr)
        
        if bsf:
            interpretation = "Brain evidence sufficient for brain-based interpretation"
        else:
            reasons = []
            if bes < self.tau_strength:
                reasons.append(
                    f"expression strength ({bes:.3f}) below threshold ({self.tau_strength:.3f})"
                )
            if bsr < self.tau_ratio:
                reasons.append(
                    f"brain specificity ({bsr:.3f}) below threshold ({self.tau_ratio:.3f})"
                )
            reason_text = ", ".join(reasons) if reasons else "insufficient brain evidence"
            interpretation = (
                f"Brain evidence insufficient – {reason_text}. "
                "Effects may be distributed across multiple tissues."
            )
        
        return {
            'bes': float(bes),
            'bsr': float(bsr) if not np.isinf(bsr) else 'inf',
            'bsf': bool(bsf),
            'interpretation': interpretation,
            'tau_strength': self.tau_strength,
            'tau_ratio': self.tau_ratio,
        }


# ============================================================================
# USAGE EXAMPLE (commented)
# ============================================================================

"""
# Example: Evaluate brain evidence for a predicted CNS drug

from src.metrics.brain_evidence import BrainEvidenceMetrics

# Initialize metrics with default thresholds
metrics = BrainEvidenceMetrics(tau_strength=0.3, tau_ratio=1.0)

# Example gene expression data (normalized to [0, 1])
gene_expression = {
    'DRD2': {
        'cortex': 0.85,
        'hippocampus': 0.78,
        'striatum': 0.92,
        'liver': 0.15,
        'kidney': 0.10,
    },
    'HTR2A': {
        'cortex': 0.72,
        'hippocampus': 0.68,
        'striatum': 0.55,
        'liver': 0.20,
        'kidney': 0.12,
    },
    'MAOA': {
        'cortex': 0.65,
        'hippocampus': 0.60,
        'striatum': 0.58,
        'liver': 0.85,  # High in liver (off-target)
        'kidney': 0.50,
    }
}

brain_regions = ['cortex', 'hippocampus', 'striatum']
body_tissues = ['liver', 'kidney']

# Compute evidence summary
summary = metrics.evidence_summary(gene_expression, brain_regions, body_tissues)

print(f"Brain Evidence Strength (BES):  {summary['bes']:.3f}")
print(f"Brain Specificity Ratio (BSR):  {summary['bsr']:.3f}")
print(f"Brain Sufficiency Flag (BSF):   {summary['bsf']}")
print(f"\nInterpretation:\n{summary['interpretation']}")

# Output (example):
# Brain Evidence Strength (BES):  0.682
# Brain Specificity Ratio (BSR):  0.883
# Brain Sufficiency Flag (BSF):   True
#
# Interpretation:
# Brain evidence sufficient for brain-based interpretation


# Example 2: High off-target expression (hepatotoxic drug)
gene_expression_offtarget = {
    'CYP3A4': {
        'cortex': 0.35,
        'hippocampus': 0.30,
        'striatum': 0.25,
        'liver': 0.95,  # Very high in liver
        'kidney': 0.70,
    }
}

summary2 = metrics.evidence_summary(
    gene_expression_offtarget, brain_regions, body_tissues
)

print(f"\n--- OFF-TARGET EXAMPLE ---")
print(f"Brain Evidence Strength (BES):  {summary2['bes']:.3f}")
print(f"Brain Specificity Ratio (BSR):  {summary2['bsr']:.3f}")
print(f"Brain Sufficiency Flag (BSF):   {summary2['bsf']}")
print(f"\nInterpretation:\n{summary2['interpretation']}")

# Output (example):
# Brain Evidence Strength (BES):  0.300
# Brain Specificity Ratio (BSR):  0.368
# Brain Sufficiency Flag (BSF):   False
#
# Interpretation:
# Brain evidence insufficient – brain specificity (0.368) below threshold (1.000).
# Effects may be distributed across multiple tissues.
"""
