"""
Task 5a: Train CNS vs Non-CNS Classification Model
Input: Integrated features + ATC hierarchy (to identify CNS drugs)
Output: Trained CNS classifier
Process: Infer (learn CNS-relevant patterns from brain gene expression)

This is Stage 1 of the two-stage classification system following the workflow design:
Stage 1: CNS-active vs Non-CNS classification
Stage 2: Drug category prediction within each class
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib


def identify_cns_drugs(atc_hierarchy: pd.DataFrame, drug_targets: pd.DataFrame, valid_indices: List[int]) -> np.ndarray:
    """
    Identify CNS drugs based on ATC categories
    
    CNS drugs have ATC codes starting with 'N' (Nervous System)
    
    Args:
        atc_hierarchy: ATC hierarchical categories
        drug_targets: Drug target information
        valid_indices: Indices of valid drugs
    
    Returns:
        Binary array indicating CNS (1) or Non-CNS (0) drugs
    """
    # Map drug_ids to ATC categories
    drug_to_atc = atc_hierarchy.groupby('drug_id')['primary_category'].first()
    
    # Create binary labels: 1 if CNS (N*), 0 otherwise
    labels = []
    for idx in valid_indices:
        drug_id = drug_targets.loc[idx, 'drug_id']
        if drug_id in drug_to_atc.index:
            atc_code = drug_to_atc[drug_id]
            # CNS drugs have primary category 'N' (Nervous System)
            is_cns = 1 if str(atc_code).startswith('N') else 0
            labels.append(is_cns)
        else:
            labels.append(0)  # Default to non-CNS if unknown
    
    return np.array(labels)


def train_multiple_models(X_train: np.ndarray, y_train: np.ndarray, scaler: StandardScaler = None) -> Tuple[Dict[str, Any], StandardScaler]:
    """
    Train and evaluate multiple CNS classification models
    
    Args:
        X_train: Training features
        y_train: Binary labels (CNS: 1, Non-CNS: 0)
        scaler: Optional pre-fitted scaler
    
    Returns:
        Tuple of (results_dict, fitted_scaler)
    """
    print("\n  Training multiple CNS classification models...")
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n    Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        
        # Fit on full training data
        model.fit(X_train_scaled, y_train)
        
        # Training metrics
        train_pred = model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred)
        train_prec = precision_score(y_train, train_pred)
        train_rec = recall_score(y_train, train_pred)
        
        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'train_precision': train_prec,
            'train_recall': train_rec
        }
        
        print(f"      CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"      Train Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"      Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
    
    # Select best model based on CV score
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    print(f"\n  ✓ Best model: {best_model_name} (CV: {results[best_model_name]['cv_mean']:.4f})")
    
    return results, scaler


def execute(X: np.ndarray,
            drug_targets: pd.DataFrame,
            atc_hierarchy: pd.DataFrame,
            valid_indices: List[int],
            output_dir: Path,
            feature_names: List[str]) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
    """
    Execute Task 5a: Train CNS vs Non-CNS classification model
    
    Args:
        X: Feature matrix from Task 4
        drug_targets: Drug target information
        atc_hierarchy: ATC hierarchical categories
        valid_indices: Valid drug indices
        output_dir: Directory to save outputs
    
    Returns:
        Tuple of (best_model, scaler, metrics_dict)
    """
    print("\n[Task 5a] Training CNS vs Non-CNS classification model...")
    
    # Prepare CNS binary labels
    print("\n  Identifying CNS vs Non-CNS drugs from ATC categories...")
    y = identify_cns_drugs(atc_hierarchy, drug_targets, valid_indices)
    
    print(f"    Total samples: {len(y)}")
    
    # Class distribution
    print(f"\n  Class distribution:")
    non_cns_count = np.sum(y == 0)
    cns_count = np.sum(y == 1)
    print(f"    Non-CNS: {non_cns_count} ({non_cns_count/len(y)*100:.1f}%)")
    print(f"    CNS: {cns_count} ({cns_count/len(y)*100:.1f}%)")
    
    # Check for extreme class imbalance
    if cns_count == 0 or non_cns_count == 0:
        print(f"    ⚠ Warning: No CNS or Non-CNS drugs found!")
    else:
        imbalance_ratio = max(non_cns_count, cns_count) / min(non_cns_count, cns_count)
        print(f"\n  Class imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 10:
            print(f"    ⚠ Warning: Significant class imbalance detected")
    
    # Train models
    results, scaler = train_multiple_models(X, y)

    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    best_model = results[best_model_name]['model']


    # Save model comparison
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'model': name,
            'cv_accuracy': result['cv_mean'],
            'cv_std': result['cv_std'],
            'train_accuracy': result['train_accuracy'],
            'train_f1': result['train_f1'],
            'train_precision': result['train_precision'],
            'train_recall': result['train_recall']
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'task5a_model_comparison.csv', index=False)
    print(f"\n  Model comparison saved: task5a_model_comparison.csv")

    # --- Visualizations ---
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Bar plot: Model comparison (CV accuracy)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='model', y='cv_accuracy', data=comparison_df, palette='Blues_d')
    plt.title('CNS Classifier Model Comparison (CV Accuracy)')
    plt.ylabel('CV Accuracy')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'task5a_model_comparison_bar.png')
    plt.close()

    # 2. Pie chart: Class distribution
    plt.figure(figsize=(5, 5))
    plt.pie([cns_count, non_cns_count], labels=['CNS', 'Non-CNS'], autopct='%1.1f%%', colors=['#4f8cff', '#ffb347'])
    plt.title('CNS vs Non-CNS Class Distribution')
    plt.tight_layout()
    plt.savefig(output_dir / 'task5a_class_distribution_pie.png')
    plt.close()

    # 3. Confusion matrix for best model (on train set)
    from sklearn.metrics import confusion_matrix
    X_scaled = scaler.transform(X)
    y_pred = best_model.predict(X_scaled)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-CNS', 'CNS'], yticklabels=['Non-CNS', 'CNS'])
    plt.title('Confusion Matrix (Train Set, Best Model)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_dir / 'task5a_confusion_matrix.png')
    plt.close()

    # 4. ROC curve for best model (on train set)
    from sklearn.metrics import roc_curve, auc
    if hasattr(best_model, 'predict_proba'):
        y_score = best_model.predict_proba(X_scaled)[:, 1]
    else:
        y_score = best_model.decision_function(X_scaled)
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_dir / 'task5a_roc_curve.png')
    plt.close()

    # Prepare metrics dictionary
    metrics = {
        'best_model': best_model_name,
        'cv_accuracy': results[best_model_name]['cv_mean'],
        'cv_std': results[best_model_name]['cv_std'],
        'train_accuracy': results[best_model_name]['train_accuracy'],
        'train_f1': results[best_model_name]['train_f1'],
        'train_precision': results[best_model_name]['train_precision'],
        'train_recall': results[best_model_name]['train_recall'],
        'num_cns_samples': int(cns_count),
        'num_non_cns_samples': int(non_cns_count),
        'class_distribution': {
            'CNS': int(cns_count),
            'Non-CNS': int(non_cns_count)
        }
    }

    # --- CNS Score Calibration: Logistic Regression on BES/BSR ---
    print("\n  Calibrating CNS score using logistic regression on BES and BSR...")
    from sklearn.linear_model import LogisticRegression
    # Explicitly extract BES and BSR columns by name
    try:
        bes_idx = feature_names.index('BES')
        bsr_idx = feature_names.index('BSR')
    except ValueError as e:
        raise ValueError("Feature names must include 'BES' and 'BSR'. Got: {}".format(feature_names))
    bes_bsr_features = X[:, [bes_idx, bsr_idx]]
    cns_lr = LogisticRegression(max_iter=1000, random_state=42)
    cns_lr.fit(bes_bsr_features, y)
    joblib.dump(cns_lr, output_dir / 'task5a_cns_score_calibrator.pkl')
    print(f"  CNS score calibrator saved: task5a_cns_score_calibrator.pkl")

    # Save model and scaler (pipeline output)
    model_wrapper = {
        'best_model': best_model,
        'scaler': scaler,
        'model_name': best_model_name
    }
    joblib.dump(model_wrapper, output_dir / 'task5a_cns_classifier.pkl')
    print(f"  CNS classifier model saved: task5a_cns_classifier.pkl")

    # Also save a copy for the agent in models/cns_classifier.joblib
    import os
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    agent_model_path = models_dir / 'cns_classifier.joblib'
    joblib.dump(model_wrapper, agent_model_path)
    print(f"  [Agent] CNS classifier model saved: {agent_model_path}")

    return best_model, scaler, metrics


if __name__ == "__main__":
    print("Task 5a requires outputs from Tasks 1-4")
    print("Run the full pipeline to test this task")
