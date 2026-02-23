from typing import Dict, List
from dataclasses import dataclass
import joblib

@dataclass
class EvidenceInputs:
    drug_name: str
    predicted_category: str
    target_genes: List[str]
    model_probs: Dict[str, float]
    cns_score: float = None
    bes: float = None
    bsr: float = None
    gene_coverage: float = None
    brain_top_clusters: any = None
    features: List[float] = None  # Full feature vector for model compatibility

class SklearnAgent:
    """
    Loads and explains predictions from a scikit-learn model trained in the pipeline (Task 5/5a).
    Supports both plain sklearn models and model wrappers (dicts with scaler/model).
    """
    def __init__(self, model_path: str):
        loaded = joblib.load(model_path)
        # Support both plain model and dict wrapper (as in task5a_cns_classifier)
        if isinstance(loaded, dict) and 'best_model' in loaded:
            self.model = loaded['best_model']
            self.scaler = loaded.get('scaler', None)
            self.model_name = loaded.get('model_name', None)
        else:
            self.model = loaded
            self.scaler = None
            self.model_name = None
        # Try to infer feature names from model or default
        self.feature_names = getattr(self.model, 'feature_names_in_', None)

    def explain(self, inputs: EvidenceInputs) -> Dict:
        # Use full feature vector if provided
        if inputs.features is not None:
            X = [inputs.features]
            # Check length matches model expectation
            if self.feature_names is not None and len(inputs.features) != len(self.feature_names):
                raise ValueError(f"Feature vector length ({len(inputs.features)}) does not match model ({len(self.feature_names)}). Provide all required features.")
        else:
            # Fallback to minimal features if model expects only a few
            input_dict = {
                'bes': inputs.bes,
                'bsr': inputs.bsr,
                'cns_score': inputs.cns_score
            }
            if self.feature_names:
                try:
                    X = [[input_dict[name] for name in self.feature_names]]
                except KeyError as e:
                    raise ValueError(f"Missing required feature: {e.args[0]}. Available: {list(input_dict.keys())}, expected: {self.feature_names}")
            else:
                X = [[inputs.bes, inputs.bsr, inputs.cns_score]]
        # Apply scaler if present
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Ensure pred is assigned before use
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        feature_importances = getattr(self.model, "feature_importances_", None)

        # Always report the top category from model_probs (if available), regardless of confidence
        atc_names = getattr(self, 'atc_descriptions', None)
        # Fallback to ATC level 1 names if available
        ATC_MAIN_GROUPS = {
            'A': 'Alimentary tract and metabolism',
            'B': 'Blood and blood forming organs',
            'C': 'Cardiovascular system',
            'D': 'Dermatologicals',
            'G': 'Genito-urinary system and sex hormones',
            'H': 'Systemic hormonal preparations',
            'J': 'Antiinfectives for systemic use',
            'L': 'Antineoplastic and immunomodulating agents',
            'M': 'Musculo-skeletal system',
            'N': 'Nervous system',
            'P': 'Antiparasitic products',
            'R': 'Respiratory system',
            'S': 'Sensory organs',
            'V': 'Various'
        }
        top_categories = getattr(inputs, 'model_probs', {})
        if top_categories:
            # Get the top category by probability
            sorted_cats = sorted(top_categories.items(), key=lambda x: -x[1])
            top_code, top_prob = sorted_cats[0]
            # Try atc_names, then fallback to ATC_MAIN_GROUPS for level 1
            if atc_names and top_code in atc_names:
                top_name = atc_names[top_code]
            elif len(top_code) == 1 and top_code in ATC_MAIN_GROUPS:
                top_name = ATC_MAIN_GROUPS[top_code]
            else:
                top_name = "Unknown Category"
            summary = f"Predicted category: {top_code} ({top_name}) with confidence {top_prob:.2f}."
        else:
            # Fallback to model prediction if model_probs not available
            if hasattr(self.model, 'classes_'):
                if isinstance(pred, int) and pred < len(self.model.classes_):
                    pred_code = self.model.classes_[pred]
                else:
                    pred_code = str(pred)
            else:
                pred_code = str(pred)
            pred_name = atc_names[pred_code] if atc_names and pred_code in atc_names else "Unknown Category"
            summary = f"Predicted category: {pred_code} ({pred_name}) with confidence {max(proba):.2f}."
        summary += f"\nImported genes: {', '.join(inputs.target_genes)}."
        summary += f"\nBES (Brain Expression Strength): {inputs.bes:.2f} — average expression in brain regions."
        summary += f"\nBSR (Brain Specificity Ratio): {inputs.bsr:.2f} — ratio of brain to baseline expression."
        summary += f"\nCNS Score: {inputs.cns_score:.2f} — likelihood of CNS activity."

        # Sufficiency check
        sufficiency = "Sufficient" if (inputs.bes and inputs.bes > 0.3) and (inputs.bsr and inputs.bsr > 1.0) else "Insufficient"
        summary += f"\nEvidence sufficiency: {sufficiency}."

        # Top 3 category explanations
        top_categories = getattr(inputs, 'model_probs', {})
        if top_categories:
            sorted_cats = sorted(top_categories.items(), key=lambda x: -x[1])[:3]
            summary += "\nTop 3 predicted categories: "
            for code, prob in sorted_cats:
                # Try atc_names, then fallback to ATC_MAIN_GROUPS for level 1
                if atc_names and code in atc_names:
                    cat_name = atc_names[code]
                elif len(code) == 1 and code in ATC_MAIN_GROUPS:
                    cat_name = ATC_MAIN_GROUPS[code]
                else:
                    cat_name = "Unknown Category"
                summary += f"\n- {code} ({cat_name}): {prob:.2f} confidence."

        # Gene expression and drug targeting
        gene_expr_quality = "Good" if inputs.gene_coverage and inputs.gene_coverage > 0.5 else "Limited"
        summary += f"\nGene expression quality: {gene_expr_quality} ({inputs.gene_coverage:.2f} coverage)."

        # Closest drugs (by gene overlap)
        closest_drugs = self._find_closest_drugs(inputs.target_genes)
        summary += "\nMost similar drugs (by target gene overlap): "
        for drug in closest_drugs:
            summary += f"\n- {drug['name']} (ID: {drug['drug_id']}, overlap: {drug['overlap']})"

        # Compose reasoning steps
        reasoning_steps = []
        if self.feature_names and feature_importances is not None:
            sorted_features = sorted(zip(self.feature_names, feature_importances), key=lambda x: -x[1])
            for idx, (name, imp) in enumerate(sorted_features[:5]):
                reasoning_steps.append({
                    "step": idx+1,
                    "name": f"Feature: {name}",
                    "question": f"How does {name} contribute?",
                    "decision": {"status": f"Importance: {imp:.2f}"}
                })
        else:
            reasoning_steps.append({
                "step": 1,
                "name": "Basic Evidence",
                "question": "Are BES, BSR, CNS Score sufficient?",
                "decision": {"status": "Evaluated"}
            })

        # Compose agent full answer
        txagent_explanation = {
            "confidence": float(max(proba)),
            "evidence_label": "Strong" if max(proba) > 0.7 else "Moderate" if max(proba) > 0.5 else "Weak",
            "details": self._reasoning(inputs, feature_importances, X),
            "sufficiency": sufficiency,
            "gene_expression_quality": gene_expr_quality,
            "closest_drugs": closest_drugs
        }

        explanation = {
            "prediction": pred,
            "confidence": float(max(proba)),
            "feature_importance": dict(zip(self.feature_names, feature_importances)) if feature_importances is not None and self.feature_names is not None else "N/A",
            "reasoning": self._reasoning(inputs, feature_importances, X),
            "summary": summary,
            "reasoning_steps": reasoning_steps,
            "txagent_explanation": txagent_explanation
        }
        return explanation

    def _find_closest_drugs(self, user_genes, top_n=4):
        import pandas as pd
        import ast
        drugs_path = 'results/drugs_data.csv'
        if not hasattr(self, '_drugs_df'):
            try:
                self._drugs_df = pd.read_csv(drugs_path)
            except Exception:
                return []
        df = self._drugs_df
        user_genes_set = set([g.upper() for g in user_genes])
        closest = []
        for _, row in df.iterrows():
            try:
                drug_genes = ast.literal_eval(row['target_genes'])
                drug_genes_set = set([g.upper() for g in drug_genes])
                overlap = len(user_genes_set & drug_genes_set)
                if overlap > 0:
                    closest.append({
                        'drug_id': row['drug_id'],
                        'name': row['name'],
                        'overlap': overlap,
                        'atc_codes': row['atc_codes'],
                        'target_genes': drug_genes
                    })
            except Exception:
                continue
        closest_sorted = sorted(closest, key=lambda x: -x['overlap'])[:top_n]
        return closest_sorted

    def _reasoning(self, inputs, feature_importances, X=None):
        lines = []
        if feature_importances is not None and self.feature_names is not None:
            sorted_features = sorted(zip(self.feature_names, feature_importances), key=lambda x: -x[1])
            lines.append(f"Top contributing features: {', '.join(f'{name} ({imp:.2f})' for name, imp in sorted_features)}.")
            # Show top feature values
            if X is not None and len(X[0]) == len(self.feature_names):
                top_feats = sorted_features[:5]
                lines.append("Top input values: " + ", ".join(f"{name}={X[0][self.feature_names.index(name)]:.2f}" for name, _ in top_feats))
        lines.append(f"BES: {inputs.bes}, BSR: {inputs.bsr}, CNS Score: {inputs.cns_score}")
        return " ".join(lines)
