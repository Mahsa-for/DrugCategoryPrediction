"""
Interactive Drug Category Prediction Interface
User-friendly interface for predicting drug categories from target genes
"""

import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path


class InteractiveDrugPredictor:
    """Interactive interface for drug category prediction"""
    
    def __init__(self, models_dir='results'):
        """Initialize with trained models from NeSy system (always use 'results' directory)"""
        self.models_dir = Path('results')
        self.predictor = None
        self.gene_expression = None
        self.classifier_model = None
        self.cns_classifier = None
        self.label_encoder = None
        self.feature_scaler = None
        
        # ATC descriptions
        self.atc_descriptions = {
            'A': 'Alimentary Tract and Metabolism',
            'B': 'Blood and Blood Forming Organs',
            'C': 'Cardiovascular System',
            'D': 'Dermatologicals',
            'G': 'Genito-Urinary System and Sex Hormones',
            'H': 'Systemic Hormonal Preparations',
            'J': 'Anti-infectives for Systemic Use',
            'L': 'Antineoplastic and Immunomodulating Agents',
            'M': 'Musculo-Skeletal System',
            'N': 'Nervous System',
            'P': 'Antiparasitic Products',
            'R': 'Respiratory System',
            'S': 'Sensory Organs',
            'V': 'Various'
        }
        
        self.load_models()
    
    def load_models(self):
        """Load trained models and data from NeSy results (always from 'results' directory)"""
        print("\n" + "="*70)
        print("LOADING MODELS FROM 'results' DIRECTORY")
        print("="*70)
        try:
            # Always use 'results' directory for all model/data files
            expr_file = Path('results') / 'task1_brain_expression.csv'
            if expr_file.exists():
                self.gene_expression = pd.read_csv(expr_file, index_col=0)
                print(f"[OK] Gene expression data loaded: {self.gene_expression.shape[0]} genes x {self.gene_expression.shape[1]} regions")
            else:
                print(f"[ERROR] Gene expression file not found: {expr_file}")
                return False

            model_file = Path('results') / 'task5_classifier_model.pkl'
            if model_file.exists():
                self.classifier_model = joblib.load(model_file)
                print(f"[OK] Drug category classifier model loaded")
            else:
                print(f"[ERROR] Classifier model not found: {model_file}")
                return False

            cns_file = Path('results') / 'task5a_cns_classifier.pkl'
            if cns_file.exists():
                self.cns_classifier = joblib.load(cns_file)
                print(f"[OK] CNS classifier (Stage 1) loaded")
            else:
                print(f"[WARN] CNS classifier not found: {cns_file}")

            encoder_file = Path('results') / 'task5_label_encoder.pkl'
            if encoder_file.exists():
                self.label_encoder = joblib.load(encoder_file)
                print(f"[OK] Label encoder loaded: {len(self.label_encoder.classes_)} categories")
            else:
                print(f"[ERROR] Label encoder not found: {encoder_file}")
                return False

            scaler_file = Path('results') / 'task4_feature_scaler.pkl'
            if scaler_file.exists():
                self.feature_scaler = joblib.load(scaler_file)
                print(f"[OK] Feature scaler loaded")
            else:
                print(f"[ERROR] Feature scaler not found: {scaler_file}")
                return False

            print("\n[OK] All models loaded successfully!")
            return True
        except Exception as e:
            print(f"\n[ERROR] Error loading models: {e}")
            return False
    
    @staticmethod
    def extract_features_static(drug_genes, gene_expression, bem):
        """
        Extract features for a drug based on its target genes and gene expression matrix.
        Matches the logic in Task 4 (task4_integrate_gene_signatures.py), including BES and BSR.
        Returns (feature_vector, input_genes, available_genes)
        """
        available_genes = [g for g in drug_genes if g in gene_expression.index]
        if not available_genes:
            return None, drug_genes, []
        gene_profiles = gene_expression.loc[available_genes]
        features = []
        # 1. Mean expression per brain cell cluster type
        features.extend(gene_profiles.mean(axis=0).values)
        # 2. Max expression per brain cell cluster type
        features.extend(gene_profiles.max(axis=0).values)
        # 3. Standard deviation per brain cell cluster type
        features.extend(gene_profiles.std(axis=0).values)
        # 4. Overall statistics
        features.append(gene_profiles.values.mean())  # Overall mean
        features.append(gene_profiles.values.std())   # Overall std
        features.append(gene_profiles.values.max())   # Overall max
        features.append(len(available_genes))          # Number of matching genes
        features.append(len(available_genes) / len(drug_genes) if drug_genes else 0)  # Gene coverage ratio
        # 5. Compute BES and BSR using BrainEvidenceMetrics
        gene_expr_dict = {g: {ct: float(gene_profiles.loc[g, ct]) for ct in gene_profiles.columns} for g in available_genes}
        brain_regions = list(gene_profiles.columns)
        body_tissues = []  # Not present in this matrix, so use empty list
        bes = bem.brain_evidence_strength(gene_expr_dict, brain_regions)
        bsr = bem.brain_specificity_ratio(gene_expr_dict, brain_regions, body_tissues)
        features.append(bes)
        features.append(bsr)
        # Enforce feature length (should match model/scaler)
        if len(features) != 109:
            raise ValueError(f"Feature vector length mismatch: expected 109, got {len(features)}")
        import numpy as np
        feature_vector = np.array(features).reshape(1, -1)
        # Clean inf/NaN values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        return feature_vector, drug_genes, available_genes

    def extract_features(self, gene_list):
        """
        Wrapper for static feature extraction using self's gene_expression and BrainEvidenceMetrics.
        """
        from src.metrics.brain_evidence import BrainEvidenceMetrics
        bem = BrainEvidenceMetrics()
        # gene_list may be upper/lower case, ensure matching index
        gene_list_upper = [g.upper() for g in gene_list]
        gene_expression = self.gene_expression
        # Map gene_expression index to upper for matching
        gene_expression.index = gene_expression.index.str.upper()
        return self.extract_features_static(gene_list_upper, gene_expression, bem)
    
    def predict(self, gene_list, top_k=5):
        """
        Predict drug category from target genes
        
        Args:
            gene_list: List of gene names
            top_k: Number of top predictions
            
        Returns:
            Dictionary with predictions and full explanation
        """
        if not self.classifier_model or not self.label_encoder:
            return {'error': 'Models not loaded. Run NeSy system first.'}
        
        # Extract features
        feature_vector, input_genes, available_genes = self.extract_features(gene_list)
        
        if feature_vector is None:
            return {
                'error': f'None of the {len(gene_list)} provided genes found in brain expression database',
                'input_genes': input_genes,
                'database_genes_sample': list(self.gene_expression.index[:10])
            }
        
        # Scale features
        feature_vector_scaled = self.feature_scaler.transform(feature_vector)
        
        # Get predictions
        y_pred = self.classifier_model.predict(feature_vector_scaled)[0]
        
        # Get probabilities if available
        if hasattr(self.classifier_model, 'predict_proba'):
            probas = self.classifier_model.predict_proba(feature_vector_scaled)[0]
        else:
            # Fallback for models without predict_proba
            probas = np.zeros(len(self.label_encoder.classes_))
            probas[y_pred] = 1.0
        
        # Get top-k predictions
        top_k = min(top_k, len(self.label_encoder.classes_))
        top_indices = np.argsort(probas)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            category_code = self.label_encoder.classes_[idx]
            probability = probas[idx]
            predictions.append({
                'category_code': category_code,
                'category_name': self.atc_descriptions.get(category_code, 'Unknown Category'),
                'probability': float(probability)
            })
        
        # Compute brain evidence metrics
        brain_evidence, region_info = self._compute_brain_evidence(available_genes)

        # CNS relevance (prefer brain evidence metrics - they're more reliable)
        cns_classification = self._compute_cns_relevance(brain_evidence, region_info)
        
        # Compute top brain clusters for available genes
        brain_top_clusters = self._compute_top_brain_clusters(available_genes)
        
        # --- Use EvidenceAgent for full explanation ---
        from src.agents.evidence_agent import EvidenceAgent, EvidenceInputs
        model_probs = {self.label_encoder.classes_[i]: float(probas[i]) for i in range(len(self.label_encoder.classes_))}
        inputs = EvidenceInputs(
            drug_name="",
            predicted_category=predictions[0]['category_code'] if predictions else '',
            target_genes=available_genes,
            model_probs=model_probs,
            cns_score=cns_classification.get('cns_probability') if cns_classification else None,
            bes=brain_evidence.get('bes') if brain_evidence else None,
            bsr=brain_evidence.get('bsr') if brain_evidence else None,
            gene_coverage=len(available_genes) / len(input_genes) if input_genes else 0.0,
            brain_top_clusters=brain_top_clusters,
            features=feature_vector.tolist()[0]
        )
        agent = EvidenceAgent()
        explanation = agent.run(inputs, task='atc')
        # Return both raw predictions and full explanation
        return {
            'success': True,
            'input_genes': input_genes,
            'genes_found': len(available_genes),
            'genes_used': available_genes,
            'primary_prediction': predictions[0] if predictions else None,
            'top_predictions': predictions,
            'brain_evidence': brain_evidence,
            'brain_region_info': region_info,
            'cns_classification': cns_classification,
            'brain_top_clusters': brain_top_clusters,
            'gene_coverage': len(available_genes) / len(input_genes) if input_genes else 0.0,
            'model_probs': model_probs,
            'explanation': explanation
        }
    
    def _compute_brain_evidence(self, available_genes):
        """Compute brain evidence metrics for given genes"""
        try:
            from src.metrics.brain_evidence import BrainEvidenceMetrics
            
            # Get expression data for available genes
            if not available_genes:
                return None
            
            gene_expr_data = self.gene_expression.loc[available_genes]
            
            # All columns are brain cell clusters - treat them as brain regions
            all_regions = self.gene_expression.columns.tolist()
            brain_regions = all_regions  # All are brain-derived cell types
            
            # Create synthetic "body tissue" baseline for specificity comparison
            # For datasets with only brain data, use mean expression as baseline
            body_tissues = ['mean_baseline']  # Synthetic for comparison
            
            region_info = {
                'brain_regions': brain_regions,
                'body_tissues': body_tissues,
                'brain_region_count': len(brain_regions),
                'body_tissue_count': len(body_tissues),
                'mapping_method': 'all_brain_clusters',
                'mapping_valid': True,
                'data_type': 'brain_cell_clusters'
            }

            # Build gene expression dictionary in the format expected by BrainEvidenceMetrics
            gene_expression_dict = {}
            for gene in available_genes:
                gene_expression_dict[gene] = {}
                # Add brain cluster expression
                for region in brain_regions:
                    gene_expression_dict[gene][region] = float(gene_expr_data.loc[gene, region])
                # Add synthetic body tissue baseline (mean of all brain expression)
                gene_expression_dict[gene]['mean_baseline'] = float(gene_expr_data.loc[gene].mean())
            
            # Compute metrics
            # tau_strength=0.3 means BES >= 30% of max for sufficient brain evidence
            # tau_ratio=0.6 means brain expression >= 60% of body baseline for specificity
            metrics = BrainEvidenceMetrics(tau_strength=0.3, tau_ratio=1.2)
            summary = metrics.evidence_summary(gene_expression_dict, brain_regions, body_tissues)
            
            return summary, region_info
            
        except Exception as e:
            print(f"Warning: Could not compute brain evidence metrics: {e}")
            return None, {
                'brain_regions': [],
                'body_tissues': [],
                'brain_region_count': 0,
                'body_tissue_count': 0,
                'mapping_method': 'keyword-heuristic',
                'mapping_valid': False,
                'error': str(e)
            }

    def _compute_cns_relevance(self, brain_evidence, region_info, cns_threshold=0.3):
        """Compute CNS relevance from brain evidence metrics. CNS threshold is now configurable."""
        if not brain_evidence or not region_info.get('mapping_valid'):
            return {
                'is_cns_active': False,
                'cns_probability': 0.0,
                'method': 'brain_evidence',
                'reason': region_info.get('error', 'Brain evidence unavailable')
            }

        # Compute CNS probability from BES and BSR metrics
        bes = float(brain_evidence.get('bes', 0.0))
        bsr = float(brain_evidence.get('bsr', 0.0))
        # Handle infinity/invalid BSR values
        if np.isinf(bsr) or np.isnan(bsr):
            bsr = 0.0
        # CNS score formula:
        bsr_normalized = min(bsr / 3.0, 1.0)
        cns_probability = (bes * 0.7 + bsr_normalized * 0.3)
        is_cns_active = bool(cns_probability > cns_threshold)
        return {
            'is_cns_active': is_cns_active,
            'cns_probability': float(cns_probability),
            'method': 'brain_evidence',
            'reason': f'Computed from BES ({bes:.2f}) and BSR ({bsr:.2f}) metrics, threshold={cns_threshold}'
        }

    def analyze_cns_threshold(self, y_true, cns_probabilities):
        """Analyze CNS threshold using ROC and suggest optimal value."""
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        import matplotlib.pyplot as plt
        fpr, tpr, thresholds = roc_curve(y_true, cns_probabilities)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('CNS Score ROC Curve')
        plt.legend(loc='lower right')
        plt.show()
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        print(f'Optimal CNS threshold (Youden J): {best_threshold:.2f}')
        return best_threshold

    def _compute_cns_from_model(self, feature_vector):
        """Compute CNS probability using the trained CNS classifier (Stage 1) if available."""
        if not self.cns_classifier:
            return None
        try:
            scaler = self.cns_classifier.get('scaler')
            model = self.cns_classifier.get('best_model')
            model_name = self.cns_classifier.get('model_name', 'Unknown')
            
            if scaler is None or model is None:
                return None

            feature_vector_cns = scaler.transform(feature_vector)
            cns_pred = model.predict(feature_vector_cns)[0]
            cns_proba = model.predict_proba(feature_vector_cns)[0]

            return {
                'is_cns_active': bool(cns_pred == 1),
                'cns_probability': float(cns_proba[1]),
                'method': 'cns_classifier_stage1',
                'reason': f'Computed from CNS Stage 1 classifier ({model_name})'
            }
        except Exception as e:
            print(f"Warning: CNS classifier failed: {e}")
            return None

    def get_cns_relevance_for_genes(self, gene_list):
        """Compute CNS relevance directly from a gene list."""
        available_genes = [g.upper() for g in gene_list if g.upper() in self.gene_expression.index]
        if not available_genes:
            return {
                'is_cns_active': False,
                'cns_probability': 0.0,
                'method': 'brain_evidence',
                'reason': 'No genes found in expression database'
            }

        # Compute brain evidence and use it directly (more reliable than classifier)
        brain_evidence, region_info = self._compute_brain_evidence(available_genes)
        return self._compute_cns_relevance(brain_evidence, region_info)
    
    def _compute_top_brain_clusters(self, available_genes, top_k=5):
        """
        Identify top brain regions/clusters with highest average expression.
        
        Args:
            available_genes: List of genes found in expression database
            top_k: Number of top clusters to return
            
        Returns:
            List of tuples: [(region_name, avg_expression), ...]
        """
        try:
            if not available_genes:
                return []
            
            # Get expression data
            gene_expr = self.gene_expression.loc[available_genes]
            
            # Calculate mean expression per region
            region_means = gene_expr.mean(axis=0)
            
            # Sort and get top-k
            top_regions = region_means.nlargest(top_k)
            
            # Return as list of tuples with normalized values
            result = []
            for region, value in top_regions.items():
                # Normalize to 0-1 range
                normalized = min(float(value) / max(region_means.max(), 1.0), 1.0)
                result.append((str(region), normalized))
            
            return result
        except Exception as e:
            print(f"Warning: Could not compute top brain clusters: {e}")
            return []
    

    
    def display_prediction(self, result):
        """Display prediction result in user-friendly format"""
        if 'error' in result:
            print(f"\n[ERROR] {result['error']}")
            if 'database_genes_sample' in result:
                print(f"\nExample genes in database:")
                for gene in result['database_genes_sample']:
                    print(f"  - {gene}")
            return
        
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        
        # Input summary
        print(f"\n📥 INPUT:")
        print(f"  Genes provided: {result['genes_found']}/{len(result['input_genes'])}")
        print(f"  Genes used: {', '.join(result['genes_used'])}")
        
        # Primary prediction
        primary = result['primary_prediction']
        print(f"\n🎯 PRIMARY PREDICTION:")
        print(f"  Category: {primary['category_name']}")
        print(f"  Code: {primary['category_code']}")
        print(f"  Confidence: {primary['probability']:.2%}")
        
        # Top predictions
        print(f"\n📊 TOP {len(result['top_predictions'])} PREDICTIONS:")
        for i, pred in enumerate(result['top_predictions'], 1):
            confidence_bar = "█" * int(pred['probability'] * 20) + "░" * (20 - int(pred['probability'] * 20))
            print(f"  {i}. [{confidence_bar}] {pred['category_name']} ({pred['category_code']})")
            print(f"     Confidence: {pred['probability']:.2%}")

        # Always show full explanation from EvidenceAgent
        explanation = result.get('explanation')
        if explanation:
            print("\n" + "="*70)
            print("FULL EXPLANATION")
            print("="*70)
            # Summary
            if 'summary' in explanation:
                print(f"\n📋 SUMMARY:\n{explanation['summary']}")
            # Reasoning steps
            if 'reasoning_steps' in explanation:
                print(f"\n🧠 REASONING STEPS:")
                for step in explanation['reasoning_steps']:
                    print(f"  Step {step['step']}: {step['name']}")
                    print(f"    Q: {step['question']}")
                    print(f"    Decision: {step['decision']['status']}")
            # Top 3 categories (already in summary, but can be shown again if needed)
            # Additional details (txagent_explanation, etc.)
            if 'txagent_explanation' in explanation:
                tx = explanation['txagent_explanation']
                print(f"\n🔬 TXAGENT EXPLANATION:")
                print(f"  Confidence: {tx.get('confidence')}")
                print(f"  Evidence label: {tx.get('evidence_label')}")
                print(f"  Sufficiency: {tx.get('sufficiency')}")
                print(f"  Gene expression quality: {tx.get('gene_expression_quality')}")
                if tx.get('closest_drugs'):
                    print(f"  Closest drugs:")
                    for drug in tx['closest_drugs']:
                        print(f"    - {drug['name']} (ID: {drug['drug_id']}, overlap: {drug['overlap']})")


def main():
    """Main interactive interface"""
    
    # Initialize
    predictor = InteractiveDrugPredictor()
    
    # Check if models are loaded
    if not predictor.classifier_model:
        print("\n[ERROR] Models not loaded. Please run 'python nesy_system.py' first.")
        return  # Now properly indented inside the function
    
    print("\n" + "="*70)
    print("INTERACTIVE DRUG CATEGORY PREDICTION")
    print("="*70)
    print("\nPredicts therapeutic drug categories based on target genes")
    print("\nCommands:")
    print("  'predict'  - Predict category for genes")
    print("  'explain'  - Get detailed explanation for last prediction")
    print("  'batch'    - Predict for multiple drugs")
    print("  'examples' - Show example predictions")
    print("  'help'     - Show this help message")
    print("  'exit'     - Exit program")
    
    last_result = None  # Store last prediction for explanation
    
    while True:
        print("\n" + "-"*70)
        command = input("Enter command (predict/explain/batch/examples/help/exit): ").strip().lower()
        
        if command == 'exit':
            print("\n[OK] Goodbye!")
            break
        
        elif command == 'help':
            print("\n" + "="*70)
            print("COMMAND HELP")
            print("="*70)
            print("\n'predict' - Make a single drug prediction")
            print("  Usage: Enter gene names separated by commas")
            print("  Example: TP53, EGFR, BRCA1")
            print("\n'explain' - Get detailed explanation for last prediction")
            print("  Shows: Gene evidence, Brain regions, CNS classification, Confidence")
            print("\n'batch' - Predict for multiple drugs at once")
            print("  Format: drug_name:gene1,gene2,gene3")
            print("\n'examples' - View pre-built example predictions")
            print("\n'exit' - Close the program")
            continue
        
        elif command == 'predict':
            print("\nEnter target gene names (comma-separated)")
            print("Example: TP53, EGFR, BRCA1")
            genes_input = input("Genes: ").strip()
            
            if not genes_input:
                print("[ERROR] No genes provided")
                continue
            
            # Parse genes
            genes = [g.strip().upper() for g in genes_input.split(',')]
            genes = [g for g in genes if g]  # Remove empty strings
            
            if not genes:
                print("[ERROR] No valid genes provided")
                continue
            
            # Get top_k
            try:
                top_k_input = input("Top-k predictions (default 5): ").strip()
                top_k = int(top_k_input) if top_k_input else 5
            except ValueError:
                top_k = 5
            
            # Predict
            result = predictor.predict(genes, top_k=top_k)
            predictor.display_prediction(result)
            last_result = result
            
            # Ask if user wants explanation
            if result.get('success'):
                ask_explain = input("\nWould you like a detailed explanation? (y/n): ").strip().lower()
                if ask_explain == 'y':
                    explanation = predictor.generate_explanation(result)
                    if explanation:
                        _display_explanation(explanation)
        
        elif command == 'explain':
            if last_result is None:
                print("\n[ERROR] No prediction to explain. Make a prediction first!")
                continue
            
            explanation = predictor.generate_explanation(last_result)
            if explanation:
                _display_explanation(explanation)
            else:
                print("\n[ERROR] Could not generate explanation")
        
        elif command == 'batch':
            print("\nEnter drugs in format: drug_name:gene1,gene2,gene3")
            print("Example:")
            print("  Aspirin:PTGS1,PTGS2")
            print("  Metformin:AMP,GATA4")
            print("\nEnter 'done' when finished")
            
            drugs = {}
            while True:
                drug_input = input("Drug: ").strip()
                
                if drug_input.lower() == 'done':
                    break
                
                if ':' not in drug_input:
                    print("[ERROR] Invalid format. Use: drug_name:gene1,gene2")
                    continue
                
                drug_name, genes_str = drug_input.split(':', 1)
                genes = [g.strip().upper() for g in genes_str.split(',')]
                genes = [g for g in genes if g]
                
                if genes:
                    drugs[drug_name.strip()] = genes
            
            if drugs:
                print(f"\n[OK] Predicting for {len(drugs)} drugs...")
                for drug_name, genes in drugs.items():
                    print(f"\n{'='*70}")
                    print(f"Drug: {drug_name}")
                    result = predictor.predict(genes, top_k=3)
                    predictor.display_prediction(result)
        
        elif command == 'examples':
            print("\n" + "="*70)
            print("EXAMPLE PREDICTIONS")
            print("="*70)
            
            examples = {
                'Antipsychotic (Dopamine antagonist)': ['DRD2', 'DRD3', 'HTR2A'],
                'Beta-blocker (Cardiovascular)': ['ADRB1', 'ADRB2'],
                'Antibiotic (Bacterial DNA)': ['GYRA', 'GYRB', 'TOPOISOMERASE'],
                'Cancer drug (EGFR inhibitor)': ['EGFR', 'ERBB2'],
                'Opioid (Pain management)': ['OPRM1', 'OPRD1']
            }
            
            for description, genes in examples.items():
                print(f"\n{'─'*70}")
                print(f"Example: {description}")
                result = predictor.predict(genes, top_k=3)
                predictor.display_prediction(result)
        
        else:
            print("[ERROR] Unknown command. Try 'predict', 'batch', 'examples', 'help', or 'exit'")


def _display_explanation(explanation):
    """Display a formatted explanation"""
    print("\n" + "="*70)
    print("DETAILED EXPLANATION (TxAgent Analysis)")
    print("="*70)
    
    # Summary
    if 'summary' in explanation:
        print(f"\n📋 SUMMARY:\n{explanation['summary']}")
    
    # Gene Evidence
    if 'gene_evidence' in explanation:
        ge = explanation['gene_evidence']
        print(f"\n🧬 GENE EVIDENCE:")
        print(f"  Genes found: {ge.get('found_genes', 'N/A')}")
        print(f"  Coverage: {ge.get('total_genes')}")
        print(f"  Interpretation: {ge.get('interpretation', 'N/A')}")
        if 'warning' in ge:
            print(f"  ⚠️  {ge['warning']}")
        if 'quality' in ge:
            print(f"  ✓ {ge['quality']}")
    
    # Brain Evidence
    if 'brain_evidence' in explanation:
        be = explanation['brain_evidence']
        print(f"\n🧠 BRAIN REGIONS:")
        for region_info in be.get('top_regions', []):
            print(f"  • {region_info['region']} (expr: {region_info['expression']})")
            print(f"    → {region_info['significance']}")
        if be.get('interpretation'):
            print(f"  Interpretation: {be['interpretation']}")
    
    # CNS Reasoning
    if 'cns_reasoning' in explanation:
        cr = explanation['cns_reasoning']
        print(f"\n🔬 CNS CLASSIFICATION:")
        print(f"  Classification: {cr['classification']} ({cr['probability']})")
        print(f"  Reasoning: {cr['reasoning']}")
    
    # Category Reasoning
    if 'category_reasoning' in explanation:
        cat = explanation['category_reasoning']
        pc = cat['primary_category']
        print(f"\n💊 DRUG CATEGORY PREDICTION:")
        print(f"  Primary: {pc['name']} ({pc['code']}) - {pc['confidence']} confidence")
        print(f"  Description: {pc['description']}")
        if cat.get('alternative_categories'):
            print(f"  Alternatives:")
            for alt in cat['alternative_categories']:
                print(f"    • {alt['name']} ({alt['code']}) - {alt['probability']}")
        print(f"  Reasoning: {cat['reasoning']}")
        if 'reliability' in cat:
            print(f"  {cat['reliability']}")
    
    # Confidence Assessment
    if 'confidence_assessment' in explanation:
        ca = explanation['confidence_assessment']
        print(f"\n📊 CONFIDENCE ASSESSMENT:")
        print(f"  Overall Score: {ca['overall_score']}")
        print(f"  Reliability: {ca['reliability']}")
        if 'caveats' in ca:
            print(f"  Caveats:")
            for caveat in ca['caveats']:
                print(f"    ⚠️  {caveat}")
    
    # Evidence Chain
    if 'evidence_chain' in explanation:
        print(f"\n🔗 EVIDENCE REASONING CHAIN:")
        for step in explanation['evidence_chain']:
            print(f"  {step}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
