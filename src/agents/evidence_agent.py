
from typing import Dict
from .sklearn_agent import SklearnAgent, EvidenceInputs


class EvidenceAgent:
    """
    Structured evidence-based reasoning agent for drug predictions (scikit-learn version).
    Supports both CNS and ATC (drug category) models.
    """

    def __init__(self):
        self.models = {
            'cns': SklearnAgent("results/task5a_cns_classifier.pkl"),
            'atc': SklearnAgent("results/task5_classifier_model.pkl")
        }

    def run(self, inputs: EvidenceInputs, task: str = 'atc') -> Dict:
        """
        Run the appropriate agent for the given task.
        task: 'cns' for CNS classifier, 'atc' for drug category classifier (default)
        """
        if task not in self.models:
            raise ValueError(f"Unknown task '{task}'. Valid options: {list(self.models.keys())}")
        return self.models[task].explain(inputs)
