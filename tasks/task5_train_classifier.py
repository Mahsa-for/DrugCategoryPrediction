"""
Task 5: Train Drug Category Classification Model
Input: Integrated features + ATC categories
Output: Trained classification model
Process: Infer (learn drug category patterns)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score


def prepare_labels(drug_targets: pd.DataFrame, atc_hierarchy: pd.DataFrame, valid_indices: List[int]) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare classification labels from ATC hierarchy
    
    Args:
        drug_targets: Drug target information
        atc_hierarchy: ATC hierarchical categories
        valid_indices: Indices of valid drugs
    
    Returns:
        Tuple of (labels, class_names)
    """
    # Map drug_ids to primary ATC categories
    drug_to_atc = atc_hierarchy.groupby('drug_id')['primary_category'].first()
    
    # Get labels for valid drugs
    labels = []
    for idx in valid_indices:
        drug_id = drug_targets.loc[idx, 'drug_id']
        if drug_id in drug_to_atc.index:
            labels.append(drug_to_atc[drug_id])
        else:
            labels.append('UNKNOWN')
    
    labels = np.array(labels)
    
    # Filter out UNKNOWN
    valid_mask = labels != 'UNKNOWN'
    labels = labels[valid_mask]
    
    class_names = sorted(list(set(labels)))
    
    return labels, class_names, valid_mask


def train_multiple_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Train and evaluate multiple classification models
    
    Returns:
        Dictionary with model performances
    """
    print("\n  Training multiple classification models...")
    
    import warnings
    warnings.filterwarnings('ignore')
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import GridSearchCV
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        LGBMClassifier = None

    model_defs = {
        'Random Forest': (RandomForestClassifier(), {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced']
        }),
        'Gradient Boosting': (GradientBoostingClassifier(), {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }),
        'SVM (RBF)': (SVC(probability=True), {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'class_weight': ['balanced']
        }),
        'Logistic Regression': (LogisticRegression(max_iter=1000, multi_class='multinomial'), {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'class_weight': ['balanced']
        })
    }
    if XGBClassifier:
        model_defs['XGBoost'] = (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.05, 0.1]
        })
    if LGBMClassifier:
        model_defs['LightGBM'] = (LGBMClassifier(), {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.05, 0.1]
        })

    results = {}
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, (model, param_grid) in model_defs.items():
        print(f"\n    Nested CV for {name}...")
        outer_scores = []
        for train_idx, test_idx in outer_cv.split(X_train, y_train):
            X_tr, X_te = X_train[train_idx], X_train[test_idx]
            y_tr, y_te = y_train[train_idx], y_train[test_idx]
            smote = SMOTE(random_state=42)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid = GridSearchCV(model, param_grid, cv=inner_cv, scoring='f1_macro', n_jobs=-1)
            grid.fit(X_tr_res, y_tr_res)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_te)
            score = f1_score(y_te, y_pred, average='macro')
            outer_scores.append(score)
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        print(f"      Nested CV Macro-F1: {mean_score:.4f} (+/- {std_score:.4f})")
        # Fit best model on all data
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        grid = GridSearchCV(model, param_grid, cv=inner_cv, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_resampled, y_resampled)
        best_model = grid.best_estimator_
        # --- Probability Calibration ---
        from sklearn.calibration import CalibratedClassifierCV
        calibrator = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
        calibrator.fit(X_resampled, y_resampled)
        train_pred = calibrator.predict(X_resampled)
        train_acc = accuracy_score(y_resampled, train_pred)
        train_f1 = f1_score(y_resampled, train_pred, average='macro')
        results[name] = {
            'model': calibrator,
            'cv_scores': np.array(outer_scores),
            'cv_mean': mean_score,
            'cv_std': std_score,
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'best_params': grid.best_params_
        }
        print(f"      Train Accuracy: {train_acc:.4f}")
        print(f"      Train Macro-F1: {train_f1:.4f}")
        print(f"      Best Params: {grid.best_params_}")
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    print(f"\n  ✓ Best model: {best_model_name} (CV: {results[best_model_name]['cv_mean']:.4f})")
    return results, best_model_name


def execute(X: np.ndarray,
            drug_targets: pd.DataFrame,
            atc_hierarchy: pd.DataFrame,
            valid_indices: List[int],
            output_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Execute Task 5: Train classification model
    
    Args:
        X: Feature matrix
        drug_targets: Drug target information
        atc_hierarchy: ATC hierarchical categories
        valid_indices: Valid drug indices
        output_dir: Directory to save outputs
    
    Returns:
        Tuple of (best_model, metrics_dict)
    """
    print("\n[Task 5] Training drug category classification model...")
    
    # Prepare labels
    print("\n  Preparing classification labels...")
    y, class_names, valid_mask = prepare_labels(drug_targets, atc_hierarchy, valid_indices)
    
    # Filter X to match y
    X_filtered = X[valid_mask]
    
    print(f"    Total samples: {len(y)}")
    print(f"    Number of classes: {len(class_names)}")
    print(f"    Class names: {', '.join(class_names)}")
    
    # Class distribution
    print(f"\n  Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"    {cls}: {count} ({count/len(y)*100:.1f}%)")
    
    # Check for class imbalance
    min_samples = counts.min()
    max_samples = counts.max()
    imbalance_ratio = max_samples / min_samples
    print(f"\n  Class imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 5:
        print(f"    ⚠ Warning: Significant class imbalance detected")
    
    # Train models
    results, best_model_name = train_multiple_models(X_filtered, y)

    best_model = results[best_model_name]['model']

    # Save model with class names for robust mapping and set model.classes_ to class_names
    import joblib
    # Patch model.classes_ to be class_names (ATC codes)
    if hasattr(best_model, 'classes_'):
        best_model.classes_ = np.array(class_names)
    model_save = {
        'best_model': best_model,
        'class_names': class_names,
        'model_name': best_model_name
    }
    joblib.dump(model_save, output_dir / 'task5_classifier_model.pkl')

    # Extract feature importance if available
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature_idx': range(len(importances)),
            'importance': importances
        }).sort_values('importance', ascending=False)

        importance_df.to_csv(output_dir / 'task5_feature_importances.csv', index=False)

        print(f"\n  Top 10 most important features:")
        for _, row in importance_df.head(10).iterrows():
            print(f"    Feature {int(row['feature_idx'])}: {row['importance']:.4f}")

    # Save model comparison
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'model': name,
            'cv_accuracy': result['cv_mean'],
            'cv_std': result['cv_std'],
            'train_accuracy': result['train_accuracy'],
            'train_f1': result['train_f1']
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'task5_model_comparison.csv', index=False)

    # Prepare metrics dictionary
    metrics = {
        'best_model': best_model_name,
        'cv_accuracy': results[best_model_name]['cv_mean'],
        'cv_std': results[best_model_name]['cv_std'],
        'train_accuracy': results[best_model_name]['train_accuracy'],
        'train_f1': results[best_model_name]['train_f1'],
        'num_classes': len(class_names),
        'num_samples': len(y),
        'class_names': class_names
    }

    return best_model, metrics


if __name__ == "__main__":
    print("Task 5 requires outputs from Tasks 1-4")
    print("Run the full pipeline to test this task")
