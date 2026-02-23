"""
Drug Category Prediction based on Target Genes
This tool predicts drug therapeutic categories using target gene information
and brain gene expression profiles.
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_score, 
                             recall_score, multilabel_confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
from tqdm import tqdm


class DatasetValidator:
    """Validate datasets for drug category prediction based on gene expression"""
    
    def __init__(self, drugbank_xml_path, cluster_type_tsv_path):
        self.drugbank_xml_path = drugbank_xml_path
        self.cluster_type_tsv_path = cluster_type_tsv_path
        self.validation_results = {
            'drugbank_exists': False,
            'gene_expression_exists': False,
            'drugbank_valid': False,
            'gene_expression_valid': False,
            'sufficient_data': False,
            'errors': [],
            'warnings': []
        }
    
    def validate_all(self):
        """Run all validation checks"""
        print("="*60)
        print("DATASET VALIDATION")
        print("="*60)
        
        # Check file existence
        self._check_file_existence()
        
        # Validate DrugBank XML
        if self.validation_results['drugbank_exists']:
            self._validate_drugbank_xml()
        
        # Validate gene expression data
        if self.validation_results['gene_expression_exists']:
            self._validate_gene_expression()
        
        # Check data sufficiency
        if (self.validation_results['drugbank_valid'] and 
            self.validation_results['gene_expression_valid']):
            self._check_data_sufficiency()
        
        # Print results
        self._print_validation_results()
        
        # Determine if we can proceed
        can_proceed = (self.validation_results['drugbank_valid'] and 
                      self.validation_results['gene_expression_valid'] and
                      self.validation_results['sufficient_data'])
        
        return can_proceed, self.validation_results
    
    def _check_file_existence(self):
        """Check if required files exist"""
        print("\n[1/4] Checking file existence...")
        
        # Check DrugBank XML
        if os.path.exists(self.drugbank_xml_path):
            self.validation_results['drugbank_exists'] = True
            file_size = os.path.getsize(self.drugbank_xml_path) / (1024**2)  # MB
            print(f"  ✓ DrugBank XML found: {self.drugbank_xml_path}")
            print(f"    File size: {file_size:.2f} MB")
        else:
            self.validation_results['errors'].append(
                f"DrugBank XML file not found: {self.drugbank_xml_path}"
            )
            print(f"  ✗ DrugBank XML NOT FOUND: {self.drugbank_xml_path}")
        
        # Check gene expression TSV
        if os.path.exists(self.cluster_type_tsv_path):
            self.validation_results['gene_expression_exists'] = True
            file_size = os.path.getsize(self.cluster_type_tsv_path) / (1024**2)  # MB
            print(f"  ✓ Gene expression TSV found: {self.cluster_type_tsv_path}")
            print(f"    File size: {file_size:.2f} MB")
        else:
            self.validation_results['errors'].append(
                f"Gene expression file not found: {self.cluster_type_tsv_path}"
            )
            print(f"  ✗ Gene expression TSV NOT FOUND: {self.cluster_type_tsv_path}")
    
    def _validate_drugbank_xml(self):
        """Validate DrugBank XML structure and content"""
        print("\n[2/4] Validating DrugBank XML...")
        
        try:
            # Try to parse the XML
            context = ET.iterparse(self.drugbank_xml_path, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)
            
            namespace = {'db': 'http://www.drugbank.ca'}
            drug_count = 0
            drugs_with_genes = 0
            drugs_with_atc = 0
            
            # Sample first 100 drugs to validate structure
            for event, elem in context:
                if event == 'end' and elem.tag == '{http://www.drugbank.ca}drug':
                    drug_count += 1
                    
                    # Check for genes
                    targets = elem.findall('db:targets/db:target', namespace)
                    if targets:
                        drugs_with_genes += 1
                    
                    # Check for ATC codes
                    atc_codes = elem.findall('db:atc-codes/db:atc-code', namespace)
                    if atc_codes:
                        drugs_with_atc += 1
                    
                    elem.clear()
                    root.clear()
                    
                    if drug_count >= 100:  # Sample first 100
                        break
            
            if drug_count > 0:
                self.validation_results['drugbank_valid'] = True
                print(f"  ✓ DrugBank XML is valid")
                print(f"    Sampled {drug_count} drugs")
                print(f"    Drugs with target genes: {drugs_with_genes} ({drugs_with_genes/drug_count*100:.1f}%)")
                print(f"    Drugs with ATC codes: {drugs_with_atc} ({drugs_with_atc/drug_count*100:.1f}%)")
                
                if drugs_with_genes < drug_count * 0.3:
                    self.validation_results['warnings'].append(
                        f"Warning: Only {drugs_with_genes/drug_count*100:.1f}% of drugs have target genes"
                    )
                
                if drugs_with_atc < drug_count * 0.3:
                    self.validation_results['warnings'].append(
                        f"Warning: Only {drugs_with_atc/drug_count*100:.1f}% of drugs have ATC codes"
                    )
            else:
                self.validation_results['errors'].append(
                    "DrugBank XML contains no valid drug entries"
                )
                print(f"  ✗ DrugBank XML contains no valid drug entries")
                
        except ET.ParseError as e:
            self.validation_results['errors'].append(
                f"DrugBank XML parsing error: {str(e)}"
            )
            print(f"  ✗ DrugBank XML parsing failed: {str(e)}")
        except Exception as e:
            self.validation_results['errors'].append(
                f"Error validating DrugBank XML: {str(e)}"
            )
            print(f"  ✗ Error validating DrugBank XML: {str(e)}")
    
    def _validate_gene_expression(self):
        """Validate gene expression TSV structure and content"""
        print("\n[3/4] Validating gene expression data...")
        
        try:
            # Try to load the TSV
            df = pd.read_csv(self.cluster_type_tsv_path, sep='\t', nrows=1000)  # Sample first 1000 rows
            
            # Check required columns
            required_columns = ['Gene name', 'Cluster type', 'nCPM']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.validation_results['errors'].append(
                    f"Gene expression TSV missing required columns: {missing_columns}"
                )
                print(f"  ✗ Missing required columns: {missing_columns}")
                print(f"    Found columns: {list(df.columns)}")
                return
            
            # Check data types and values
            if not pd.api.types.is_numeric_dtype(df['nCPM']):
                self.validation_results['errors'].append(
                    "nCPM column should contain numeric values"
                )
                print(f"  ✗ nCPM column is not numeric")
                return
            
            # Count unique values
            unique_genes = df['Gene name'].nunique()
            unique_clusters = df['Cluster type'].nunique()
            
            if unique_genes < 100:
                self.validation_results['warnings'].append(
                    f"Warning: Only {unique_genes} unique genes found (minimum 1000 recommended)"
                )
            
            if unique_clusters < 5:
                self.validation_results['warnings'].append(
                    f"Warning: Only {unique_clusters} unique cluster types found (minimum 10 recommended)"
                )
            
            self.validation_results['gene_expression_valid'] = True
            print(f"  ✓ Gene expression TSV is valid")
            print(f"    Unique genes (sampled): {unique_genes}")
            print(f"    Unique cluster types: {unique_clusters}")
            print(f"    Expression values range: {df['nCPM'].min():.2f} - {df['nCPM'].max():.2f}")
            
        except pd.errors.ParserError as e:
            self.validation_results['errors'].append(
                f"Gene expression TSV parsing error: {str(e)}"
            )
            print(f"  ✗ TSV parsing failed: {str(e)}")
        except Exception as e:
            self.validation_results['errors'].append(
                f"Error validating gene expression data: {str(e)}"
            )
            print(f"  ✗ Error validating gene expression: {str(e)}")
    
    def _check_data_sufficiency(self):
        """Check if datasets have sufficient data for prediction"""
        print("\n[4/4] Checking data sufficiency...")
        
        try:
            # Load full gene expression data
            df = pd.read_csv(self.cluster_type_tsv_path, sep='\t')
            unique_genes = df['Gene name'].nunique()
            unique_clusters = df['Cluster type'].nunique()
            
            # Minimum requirements
            min_genes = 1000
            min_clusters = 10
            
            if unique_genes < min_genes:
                self.validation_results['warnings'].append(
                    f"Gene count ({unique_genes}) below recommended minimum ({min_genes})"
                )
                print(f"  ⚠ Gene count ({unique_genes}) below recommended ({min_genes})")
            else:
                print(f"  ✓ Sufficient genes: {unique_genes}")
            
            if unique_clusters < min_clusters:
                self.validation_results['warnings'].append(
                    f"Cluster count ({unique_clusters}) below recommended minimum ({min_clusters})"
                )
                print(f"  ⚠ Cluster count ({unique_clusters}) below recommended ({min_clusters})")
            else:
                print(f"  ✓ Sufficient clusters: {unique_clusters}")
            
            # Consider it sufficient if we have at least some reasonable data
            if unique_genes >= 500 and unique_clusters >= 5:
                self.validation_results['sufficient_data'] = True
                print(f"  ✓ Datasets contain sufficient data for prediction")
            else:
                self.validation_results['errors'].append(
                    f"Insufficient data: genes={unique_genes}, clusters={unique_clusters}"
                )
                print(f"  ✗ Insufficient data for reliable prediction")
                
        except Exception as e:
            self.validation_results['errors'].append(
                f"Error checking data sufficiency: {str(e)}"
            )
            print(f"  ✗ Error checking data sufficiency: {str(e)}")
    
    def _print_validation_results(self):
        """Print summary of validation results"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        if self.validation_results['errors']:
            print("\n❌ ERRORS:")
            for error in self.validation_results['errors']:
                print(f"  - {error}")
        
        if self.validation_results['warnings']:
            print("\n⚠ WARNINGS:")
            for warning in self.validation_results['warnings']:
                print(f"  - {warning}")
        
        if not self.validation_results['errors']:
            print("\n✅ All validation checks passed!")
            print("   Datasets are ready for drug category prediction.")
        else:
            print("\n❌ Validation failed!")
            print("   Please fix the errors before proceeding.")
        
        print("="*60 + "\n")


class DrugBankParser:
    """Parse DrugBank XML database to extract drug information"""
    
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.namespace = {'db': 'http://www.drugbank.ca'}
        self.drugs_data = []
        
    def parse(self):
        """Parse DrugBank XML and extract drug information"""
        print("Parsing DrugBank XML (this may take a few minutes)...")
        
        # Use iterparse for large files
        context = ET.iterparse(self.xml_path, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        drug_count = 0
        for event, elem in tqdm(context, desc="Parsing drugs"):
            if event == 'end' and elem.tag == '{http://www.drugbank.ca}drug':
                drug_data = self._extract_drug_info(elem)
                if drug_data:
                    self.drugs_data.append(drug_data)
                    drug_count += 1
                
                # Clear element to save memory
                elem.clear()
                root.clear()
                
        print(f"Parsed {drug_count} drugs from DrugBank")
        return pd.DataFrame(self.drugs_data)
    
    def _extract_drug_info(self, drug_elem):
        """Extract relevant information from a drug element"""
        drug_id = self._get_text(drug_elem, 'db:drugbank-id[@primary="true"]')
        name = self._get_text(drug_elem, 'db:name')
        
        # Extract ATC codes
        atc_codes = []
        atc_levels = []
        for atc in drug_elem.findall('db:atc-codes/db:atc-code', self.namespace):
            atc_code = atc.get('code', '')
            if atc_code:
                atc_codes.append(atc_code)
                # Extract therapeutic category (first level description)
                levels = atc.findall('db:level', self.namespace)
                if levels:
                    atc_levels.append(levels[0].text)
        
        # Extract target genes
        target_genes = []
        for target in drug_elem.findall('db:targets/db:target', self.namespace):
            for polypeptide in target.findall('db:polypeptide', self.namespace):
                gene = self._get_text(polypeptide, 'db:gene-name')
                if gene:
                    target_genes.append(gene)
        
        # Extract enzyme genes
        enzyme_genes = []
        for enzyme in drug_elem.findall('db:enzymes/db:enzyme', self.namespace):
            for polypeptide in enzyme.findall('db:polypeptide', self.namespace):
                gene = self._get_text(polypeptide, 'db:gene-name')
                if gene:
                    enzyme_genes.append(gene)
        
        # Extract transporter genes
        transporter_genes = []
        for transporter in drug_elem.findall('db:transporters/db:transporter', self.namespace):
            for polypeptide in transporter.findall('db:polypeptide', self.namespace):
                gene = self._get_text(polypeptide, 'db:gene-name')
                if gene:
                    transporter_genes.append(gene)
        
        # Combine all genes
        all_genes = list(set(target_genes + enzyme_genes + transporter_genes))
        
        if not all_genes or not atc_codes:
            return None
        
        return {
            'drug_id': drug_id,
            'name': name,
            'atc_codes': atc_codes,
            'atc_categories': list(set(atc_levels)),
            'target_genes': target_genes,
            'enzyme_genes': enzyme_genes,
            'transporter_genes': transporter_genes,
            'all_genes': all_genes,
            'num_genes': len(all_genes)
        }
    
    def _get_text(self, elem, path):
        """Safely extract text from XML element"""
        found = elem.find(path, self.namespace)
        return found.text if found is not None and found.text else None


class BrainGeneExpressionLoader:
    """Load and process brain gene expression data"""
    
    def __init__(self, cluster_type_path, brain_cluster_path=None):
        self.cluster_type_path = cluster_type_path
        self.brain_cluster_path = brain_cluster_path
        self.gene_expression = None
        
    def load(self):
        """Load brain gene expression data"""
        print("Loading brain gene expression data...")
        
        # Load cluster type data (gene expression by cluster type)
        df = pd.read_csv(self.cluster_type_path, sep='\t')
        print(f"Loaded {len(df)} gene-cluster pairs")
        print(f"Unique genes: {df['Gene name'].nunique()}")
        print(f"Unique cluster types: {df['Cluster type'].nunique()}")
        
        # Create a gene expression matrix: genes x cluster types
        self.gene_expression = df.pivot_table(
            index='Gene name',
            columns='Cluster type',
            values='nCPM',
            aggfunc='mean'
        ).fillna(0)
        
        print(f"Gene expression matrix shape: {self.gene_expression.shape}")
        return self.gene_expression


class DrugFeatureExtractor:
    """Extract features for drugs based on target genes and brain expression"""
    
    def __init__(self, gene_expression_df):
        self.gene_expression = gene_expression_df
        self.available_genes = set(gene_expression_df.index)
        
    def extract_features(self, drugs_df):
        """Extract features for each drug based on their target genes"""
        print("Extracting features for drugs...")
        
        features_list = []
        valid_indices = []
        
        for idx, row in tqdm(drugs_df.iterrows(), total=len(drugs_df)):
            drug_genes = [g for g in row['all_genes'] if g in self.available_genes]
            
            if not drug_genes:
                continue
            
            # Get expression values for drug's target genes
            gene_expr = self.gene_expression.loc[drug_genes]
            
            # Aggregate features
            features = {
                'mean_expression': gene_expr.mean().values,
                'max_expression': gene_expr.max().values,
                'std_expression': gene_expr.std().fillna(0).values,
                'sum_expression': gene_expr.sum().values,
            }
            
            # Flatten all features into a single vector
            feature_vector = np.concatenate([
                features['mean_expression'],
                features['max_expression'],
                features['std_expression'],
                features['sum_expression']
            ])
            
            # Replace any remaining NaN with 0
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)
            
            features_list.append(feature_vector)
            valid_indices.append(idx)
        
        feature_matrix = np.array(features_list)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Valid drugs with features: {len(valid_indices)}")
        
        return feature_matrix, valid_indices


class CNSClassifier:
    """First stage: Classify drugs as CNS-active or Non-CNS"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    def is_cns_drug(self, atc_codes):
        """Determine if a drug is CNS-active based on ATC codes"""
        cns_prefixes = ['N']  # N = Nervous system
        return any(code.startswith(tuple(cns_prefixes)) for code in atc_codes)
    
    def prepare_data(self, X, drugs_df, valid_indices):
        """Prepare data for CNS classification"""
        y = np.array([
            1 if self.is_cns_drug(drugs_df.loc[idx, 'atc_codes']) else 0
            for idx in valid_indices
        ])
        return X, y
    
    def train_models(self, X_train, y_train):
        """Train multiple classifiers"""
        print("\nTraining CNS vs Non-CNS classifiers...")
        
        # Define models
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Select best model
        best_name = max(results, key=lambda k: results[k]['cv_mean'])
        self.best_model = results[best_name]['model']
        print(f"\nBest model: {best_name}")
        
        return results
    
    def evaluate(self, X_test, y_test):
        """Evaluate the best model"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.best_model.predict(X_test_scaled)
        
        print("\n=== CNS vs Non-CNS Classification Results ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-CNS', 'CNS']))
        
        return y_pred, {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }


class DrugCategoryClassifier:
    """Second stage: Classify drugs into therapeutic categories"""
    
    def __init__(self, min_samples_per_class=5):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.min_samples = min_samples_per_class
        self.label_encoder = None
        self.class_names = None
        
    def prepare_data(self, X, drugs_df, valid_indices):
        """Prepare data for multi-class classification"""
        # Extract primary ATC category (first letter)
        y = [drugs_df.loc[idx, 'atc_codes'][0][0] if drugs_df.loc[idx, 'atc_codes'] 
             else None for idx in valid_indices]
        
        # Filter out None values
        valid_mask = [i for i, label in enumerate(y) if label is not None]
        X_filtered = X[valid_mask]
        y_filtered = [y[i] for i in valid_mask]
        
        # Count class frequencies
        from collections import Counter
        class_counts = Counter(y_filtered)
        print(f"\nClass distribution: {dict(class_counts)}")
        
        # Keep only classes with sufficient samples
        valid_classes = {cls for cls, count in class_counts.items() 
                        if count >= self.min_samples}
        
        final_mask = [i for i, label in enumerate(y_filtered) 
                     if label in valid_classes]
        X_final = X_filtered[final_mask]
        y_final = [y_filtered[i] for i in final_mask]
        
        print(f"\nFiltered to {len(valid_classes)} classes with >= {self.min_samples} samples")
        print(f"Total samples: {len(y_final)}")
        
        return X_final, np.array(y_final)
    
    def train_models(self, X_train, y_train):
        """Train multiple classifiers for drug category prediction"""
        print("\nTraining Drug Category classifiers...")
        
        # Store class names
        self.class_names = sorted(set(y_train))
        
        # Define models
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Select best model
        best_name = max(results, key=lambda k: results[k]['cv_mean'])
        self.best_model = results[best_name]['model']
        print(f"\nBest model: {best_name}")
        
        return results
    
    def evaluate(self, X_test, y_test):
        """Evaluate the best model"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.best_model.predict(X_test_scaled)
        
        print("\n=== Drug Category Classification Results ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Macro F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
        print(f"Weighted F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return y_pred, {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'class_names': self.class_names
        }


class ModelEvaluator:
    """Evaluate and visualize model performance"""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_confusion_matrix(self, cm, class_names, title, filename):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix to {filename}")
    
    def plot_model_comparison(self, results_dict, metric_name, filename):
        """Plot comparison of different models"""
        models = list(results_dict.keys())
        scores = [results_dict[m]['cv_mean'] for m in models]
        stds = [results_dict[m]['cv_std'] for m in models]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, scores, yerr=stds, capsize=5, alpha=0.7)
        plt.ylabel(metric_name)
        plt.title(f'Model Comparison - {metric_name}')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved model comparison to {filename}")
    
    def save_results_summary(self, cns_metrics, category_metrics, filename='results_summary.txt'):
        """Save comprehensive results summary"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DRUG CATEGORY PREDICTION - RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Stage 1: CNS vs Non-CNS Classification\n")
            f.write("-" * 60 + "\n")
            for metric, value in cns_metrics.items():
                if metric != 'confusion_matrix':
                    f.write(f"{metric.capitalize()}: {value:.4f}\n")
            f.write("\n")
            
            f.write("Stage 2: Drug Category Classification\n")
            f.write("-" * 60 + "\n")
            for metric, value in category_metrics.items():
                if metric not in ['confusion_matrix', 'class_names']:
                    f.write(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}\n")
            f.write("\n")
            
        print(f"Saved results summary to {filename}")


def main():
    """Main pipeline for drug category prediction"""
    
    print("=" * 60)
    print("DRUG CATEGORY PREDICTION PIPELINE")
    print("=" * 60)
    
    # Paths (project-relative for portability)
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    drugbank_xml = base_dir / "drugbank_all_full_database.xml" / "full_database.xml"
    cluster_type_tsv = base_dir / "dataset" / "rna_single_nuclei_cluster_type.tsv" / "rna_single_nuclei_cluster_type.tsv"
    
    # Step 0: Validate Datasets
    print("\n### STEP 0: Validate Datasets ###")
    validator = DatasetValidator(drugbank_xml, cluster_type_tsv)
    can_proceed, validation_results = validator.validate_all()
    
    if not can_proceed:
        print("\n❌ PIPELINE ABORTED: Dataset validation failed!")
        print("Please ensure all required datasets are available and properly formatted.")
        return
    
    print("\n✅ Dataset validation successful! Proceeding with pipeline...\n")
    
    # Step 1: Parse DrugBank
    print("\n### STEP 1: Parse DrugBank Database ###")
    parser = DrugBankParser(drugbank_xml)
    drugs_df = parser.parse()
    drugs_df.to_csv(results_dir / "drugs_data.csv", index=False)
    print(f"Saved drugs data to drugs_data.csv")
    
    # Step 2: Load brain gene expression
    print("\n### STEP 2: Load Brain Gene Expression Data ###")
    brain_loader = BrainGeneExpressionLoader(cluster_type_tsv)
    gene_expression = brain_loader.load()
    gene_expression.to_csv('results/gene_expression.csv')
    print(f"Saved gene expression data")
    
    # Step 3: Extract features
    print("\n### STEP 3: Extract Drug Features ###")
    feature_extractor = DrugFeatureExtractor(gene_expression)
    X, valid_indices = feature_extractor.extract_features(drugs_df)
    
    # Step 4: CNS vs Non-CNS Classification
    print("\n### STEP 4: CNS vs Non-CNS Classification ###")
    cns_classifier = CNSClassifier()
    X_cns, y_cns = cns_classifier.prepare_data(X, drugs_df, valid_indices)
    
    print(f"\nCNS drug distribution:")
    print(f"  Non-CNS: {np.sum(y_cns == 0)} ({np.sum(y_cns == 0) / len(y_cns) * 100:.1f}%)")
    print(f"  CNS: {np.sum(y_cns == 1)} ({np.sum(y_cns == 1) / len(y_cns) * 100:.1f}%)")
    
    # Split data
    X_train_cns, X_test_cns, y_train_cns, y_test_cns = train_test_split(
        X_cns, y_cns, test_size=0.2, random_state=42, stratify=y_cns
    )
    
    # Train models
    cns_results = cns_classifier.train_models(X_train_cns, y_train_cns)
    
    # Evaluate
    y_pred_cns, cns_metrics = cns_classifier.evaluate(X_test_cns, y_test_cns)
    
    # Step 5: Drug Category Classification
    print("\n### STEP 5: Drug Category Classification ###")
    category_classifier = DrugCategoryClassifier(min_samples_per_class=10)
    X_cat, y_cat = category_classifier.prepare_data(X, drugs_df, valid_indices)
    
    # Split data
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X_cat, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )
    
    # Train models
    category_results = category_classifier.train_models(X_train_cat, y_train_cat)
    
    # Evaluate
    y_pred_cat, category_metrics = category_classifier.evaluate(X_test_cat, y_test_cat)
    
    # Step 6: Generate evaluation reports
    print("\n### STEP 6: Generate Evaluation Reports ###")
    evaluator = ModelEvaluator(output_dir='results')
    
    # Plot confusion matrices
    evaluator.plot_confusion_matrix(
        cns_metrics['confusion_matrix'],
        ['Non-CNS', 'CNS'],
        'CNS vs Non-CNS - Confusion Matrix',
        'confusion_matrix_cns.png'
    )
    
    evaluator.plot_confusion_matrix(
        category_metrics['confusion_matrix'],
        category_metrics['class_names'],
        'Drug Category - Confusion Matrix',
        'confusion_matrix_category.png'
    )
    
    # Plot model comparisons
    evaluator.plot_model_comparison(
        cns_results,
        'Cross-Validation Accuracy',
        'model_comparison_cns.png'
    )
    
    evaluator.plot_model_comparison(
        category_results,
        'Cross-Validation Accuracy',
        'model_comparison_category.png'
    )
    
    # Save results summary
    evaluator.save_results_summary(cns_metrics, category_metrics)
    
    # Save models
    print("\n### Saving Models ###")
    joblib.dump(cns_classifier, 'results/cns_classifier.pkl')
    joblib.dump(category_classifier, 'results/category_classifier.pkl')
    print("Models saved successfully")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nResults saved in 'results/' directory:")
    print("  - confusion_matrix_cns.png")
    print("  - confusion_matrix_category.png")
    print("  - model_comparison_cns.png")
    print("  - model_comparison_category.png")
    print("  - results_summary.txt")
    print("  - drugs_data.csv")
    print("  - gene_expression.csv")
    print("  - cns_classifier.pkl")
    print("  - category_classifier.pkl")


if __name__ == "__main__":
    main()