import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GroupKFold, ParameterGrid
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    precision_recall_curve,
    precision_score, recall_score
)
from tqdm import tqdm
import joblib
from pathlib import Path
import os

from src.utils import (
    load_merged_data,
    create_lag_features, 
    save_model,
    RANDOM_SEED
)

np.random.seed(RANDOM_SEED)

print("="*60)
print("ENGAGEMENT DETECTION MODEL TRAINING")
print("="*60)

# ==============================================================================
# 1. Loading data
# ==============================================================================

print("\nLoading data...")
df = load_merged_data()
print(f"  Loaded {len(df)} observations with {df.shape[1]} features")

# ==============================================================================
# 2. Feature engineering
# ==============================================================================

print("\nEngineering features...")

# Sort by student and time first!
df = df.sort_values(['student_id', 'bromp_time']).reset_index(drop=True)

# Define features to create lags for
lag_features = [
    'events_symmetric_90s', 
    'events_back_60s', 
    'pauses_back_60s',
    'pauses_symmetric_60s', 
    'time_since_last_event', 
    'assessment_back_60s'
]

# Create lag features (using our helper to avoid code duplication)
df = create_lag_features(df, lag_features)

# Add observation number
df['obs_number'] = df.groupby('student_id').cumcount() + 1

# Making deviation features
df['events_deviation_90s'] = df['events_symmetric_90s'] - df['student_avg_events_symmetric_90s']
df['events_deviation_60s'] = df['events_back_60s'] - df['student_avg_events_back_60s']
df['time_since_deviation'] = df['time_since_last_event'] - df['student_avg_time_since_last_event']
df['pauses_deviation'] = df['pauses_back_60s'] - df['student_avg_pauses_back_60s']
df['pause_rate_60s'] = np.where(
    df['events_back_60s'] > 0,
    df['pauses_back_60s'] / df['events_back_60s'],
    0
)

# Ratio to baseline
df['events_ratio_90s'] = df['events_symmetric_90s'] / (df['student_avg_events_symmetric_90s'] + 1)

# Deviation from RECENT baseline (last 5)
df['events_deviation_recent'] = df['events_symmetric_90s'] - df['student_recent_events_symmetric_90s']

# ==============================================================================
# 3. Creating target variable
# ==============================================================================

print("\nCreating target variable...")

df['engaged'] = df['behavior'].apply(
    lambda x: 'ENGAGED' if x in ['ON TASK', 'ON TASK CONV'] else 'DISENGAGED'
)

print("  Target variable distribution:")
print(df['engaged'].value_counts())

engagement_map = {'DISENGAGED': 0, 'ENGAGED': 1}
df['engaged'] = df['engaged'].map(engagement_map).astype('category')

# ==============================================================================
# 4. Feature prepping
# ==============================================================================

print("\nPreparing features...")

deviation_ft = [
    'events_deviation_90s',
    'events_deviation_60s', 
    'time_since_deviation',
    'pauses_deviation',
    'events_ratio_90s',
    'events_deviation_recent',
    'pause_rate_60s'
]

backward_ft = [
    'events_back_30s',
    'events_back_60s',
    'events_back_90s',
    'pauses_back_60s',
    'assessment_back_60s',
    'time_since_last_event'
]

symmetric_ft = [
    'events_symmetric_60s',
    'events_symmetric_90s',
    'pauses_symmetric_60s'
]

baseline_ft = [
    'student_avg_events_symmetric_90s',
    'student_std_events_symmetric_90s',
    'student_recent_events_symmetric_90s',
    'student_avg_time_since_last_event',
    'student_avg_pauses_back_60s',
    'student_avg_assessment_back_60s',
    'obs_number'
]

features = deviation_ft + backward_ft + symmetric_ft + baseline_ft

print(f"  Total features: {len(features)}")
print(f"    Deviation: {len(deviation_ft)}")
print(f"    Backward: {len(backward_ft)}")
print(f"    Symmetric: {len(symmetric_ft)}")
print(f"    Baseline: {len(baseline_ft)}")

X = df[features].copy()
y = df['engaged'].astype(int)
groups = df['student_id'].copy()

# POTENTIAL FLAG FOR DELETION
# THIS CHUNK IS SO THAT THE MODEL WOULD TRAIN ON OBSERVATIONS WITH ACTIVITY ONLY
print("  Filtering to observations with activity (events_symmetric_90s > 0)...")
activity_mask = df['events_symmetric_90s'] > 0
X = X[activity_mask]
y = y[activity_mask]
groups = groups[activity_mask]
print(f"  Kept {len(X)} / {len(df)} observations")

# ==============================================================================
# 5. Student-level train/test/val split
# ==============================================================================

print("\nCreating train/test/val splits (student-level)...")

# We want to split by students, not by observations, in order to prevent data leakage
unique_students = groups.unique()
train_students, test_students = train_test_split(
    unique_students, 
    test_size=0.2, 
    random_state=RANDOM_SEED
)

train_mask = groups.isin(train_students)
test_mask = groups.isin(test_students)

X_temp = X[train_mask]
y_temp = y[train_mask]
groups_temp = groups[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]
groups_test = groups[test_mask]

# Need to do this ^ again to split _temp into train and validation
temp_unique_students = groups_temp.unique()
train_students_final, val_students = train_test_split(
    temp_unique_students,
    test_size=0.2,
    random_state=RANDOM_SEED
)

train_mask = groups_temp.isin(train_students_final)
val_mask = groups_temp.isin(val_students)

X_train = X_temp[train_mask]
y_train = y_temp[train_mask]
groups_train = groups_temp[train_mask]

X_val = X_temp[val_mask]
y_val = y_temp[val_mask]
groups_val = groups_temp[val_mask]

print(f"  Train: {len(X_train)} obs, {len(train_students_final)} students")
print(f"  Val:   {len(X_val)} obs, {len(val_students)} students")
print(f"  Test:  {len(X_test)} obs, {len(test_students)} students")

# ==============================================================================
# 6. Grid Search with GroupKFold CV
# ==============================================================================

print("\nTraining XGBoost model...")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  Class imbalance ratio: {scale_pos_weight:.2f}")

# Parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'colsample_bylevel': [0.6, 0.7, 0.8],
    'scale_pos_weight': [0, scale_pos_weight]
}

cv_setting = GroupKFold(n_splits=5)
param_combos = list(ParameterGrid(param_grid))
total_folds = len(param_combos) * cv_setting.get_n_splits(X_train, y_train, groups_train)

print(f"  Grid search: {len(param_combos)} parameter combinations")
print(f"  Total fits: {total_folds}")

# 
cache_dir = Path('outputs/cache')
cache_dir.mkdir(parents=True, exist_ok=True)
cache_file = cache_dir / 'grid_search_results.pkl'

if cache_file.exists():
    print(f"\n  WARNING: Found cached grid search results at {cache_file}")
    print("  Loading cached results instead of re-running grid search...")
    print("  Delete this file to re-run grid search from scratch.")
    
    cached_results = joblib.load(cache_file)
    results = cached_results['results']
    best_params = cached_results['best_params']
    best_score = cached_results['best_score']
    
    print(f"  Loaded cached results")
    print(f"  Best CV AUC: {best_score:.4f}")
    print(f"  Best params: {best_params}")

else:
    print("\n  No cache found. Running full grid search...")
    
    best_score = -np.inf
    best_params = None
    results = []

    with tqdm(total=total_folds, desc="Grid Search Progress", ncols=100) as pbar:
        for i, params in enumerate(param_combos, start=1):
            model_params = params.copy()
            fold_scores = []

            for train_idx, val_idx in cv_setting.split(X_train, y_train, groups_train):
                X_tr, X_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = XGBClassifier(**model_params, random_state=RANDOM_SEED,
                                      eval_metric='aucpr', enable_categorical=True)
                model.fit(X_tr.values, y_tr.values)

                preds = model.predict_proba(X_cv.values)[:, 1]
                fold_auc = roc_auc_score(y_cv.values, preds)
                fold_scores.append(fold_auc)

                pbar.update(1)
                pbar.set_postfix({
                    'combo': f"{i}/{len(param_combos)}",
                    'mean_auc': f"{np.mean(fold_scores):.4f}"
                })

            mean_auc = np.mean(fold_scores)
            std_auc = np.std(fold_scores)
            
            # Store results
            results.append({
                **params,
                'mean_cv_auc': mean_auc,
                'std_cv_auc': std_auc
            })
            
            if mean_auc > best_score:
                best_score = mean_auc
                best_params = params

    print(f"\n  Grid search complete!")
    print(f"  Best CV AUC: {best_score:.4f}")
    print(f"  Best params: {best_params}")
    
    # Cache the results
    print(f"\n  Caching results to {cache_file}...")
    joblib.dump({
        'results': results,
        'best_params': best_params,
        'best_score': best_score,
        'timestamp': pd.Timestamp.now()
    }, cache_file)
    print("Results cached successfully.")

# ==============================================================================
# 7. Train Final Model
# ==============================================================================

print("\nTraining final model with best parameters...")

final_model = XGBClassifier(**best_params, random_state=RANDOM_SEED, 
                            eval_metric='aucpr', enable_categorical=True)
final_model.fit(X_train.values, y_train.values)

# ==============================================================================
# 8. Validation Set Evaluation
# ==============================================================================

print("\nEvaluating on validation set...")

y_pred_val = final_model.predict(X_val.values)
y_proba_val = final_model.predict_proba(X_val.values)[:, 1]

val_auc = roc_auc_score(y_val, y_proba_val)
val_f1 = f1_score(y_val, y_pred_val)

print(f"\n  Validation Results:")
print(f"    AUC: {val_auc:.4f}")
print(f"    F1:  {val_f1:.4f}")

# Threshold tuning for F1
precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)
f1s = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]

print(f"\n  Best F1 threshold: {best_threshold:.3f}")
print(f"  Best F1 score: {f1s[best_idx]:.3f}")

# Retrain on train+val combined
print("\n  Retraining on train+val combined...")
X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])
final_model.fit(X_train_final.values, y_train_final.values)

# ==============================================================================
# 9. Test Set Evaluation
# ==============================================================================

print("\nEvaluating on test set...")

y_pred_test = final_model.predict(X_test.values)
y_proba_test = final_model.predict_proba(X_test.values)[:, 1]

test_auc = roc_auc_score(y_test, y_proba_test)
test_f1 = f1_score(y_test, y_pred_test)

print(f"\n  Test Results:")
print(f"    AUC: {test_auc:.4f}")
print(f"    F1:  {test_f1:.4f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=['DISENGAGED', 'ENGAGED']))

# ==============================================================================
# 10. Save Outputs
# ==============================================================================

print("\nSaving outputs...")

# Save model with metadata (using our helper)
metadata = {
    'cv_auc': best_score,
    'val_auc': val_auc,
    'test_auc': test_auc,
    'val_f1': val_f1,
    'test_f1': test_f1,
    'best_params': best_params,
    'best_threshold': best_threshold,
    'n_features': len(features),
    'n_train': len(X_train),
    'n_val': len(X_val),
    'n_test': len(X_test)
}

save_model(final_model, 'outputs/models/engagement_detector.pkl', metadata)

# Save feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance.to_csv('outputs/models/feature_importance.csv', index=False)
print(f"    Feature importance saved to outputs/models/feature_importance.csv")

# Save grid search results
results_df = pd.DataFrame(results)
results_df.to_csv('outputs/models/grid_search_results.csv', index=False)
print(f"    Grid search results saved to outputs/models/grid_search_results.csv")

# Save final metrics
final_metrics = pd.DataFrame([metadata])
final_metrics.to_csv('outputs/models/model_metrics.csv', index=False)
print(f"    Model metrics saved to outputs/models/model_metrics.csv")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Best CV AUC:  {best_score:.4f}")
print(f"Val AUC:      {val_auc:.4f}")
print(f"Test AUC:     {test_auc:.4f}")
print("="*60)

# ==============================================================================
# 11. Export Data for Visualization
# ==============================================================================

print("\n  Exporting data for visualizations...")

# Create plots/data directory
plots_dir = Path('data/figures')
plots_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 11.1. Feature Importance with Categories
# ---------------------------------------------------------------------------

def categorize_feature(feature_name):
    if 'student_avg' in feature_name or 'student_std' in feature_name or 'student_recent' in feature_name:
        return 'Student Baseline (Lag Features)'
    elif 'deviation' in feature_name or 'ratio' in feature_name:
        return 'Deviation from Baseline'
    elif 'back' in feature_name or 'symmetric' in feature_name or 'forward' in feature_name:
        return 'Raw Activity Windows'
    elif 'pause' in feature_name:
        return 'Pause Behavior'
    elif 'assessment' in feature_name:
        return 'Assessment Activity'
    elif 'time_since' in feature_name:
        return 'Time Since Last Event'
    elif 'obs_number' in feature_name:
        return 'Observation Sequence'
    else:
        return 'Other'

# Reload feature importance (already saved earlier)
feature_importance = pd.read_csv('outputs/models/feature_importance.csv')

# Add category and enriched metrics
feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)
feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()
feature_importance['importance_pct'] = (
    feature_importance['importance'] / feature_importance['importance'].sum() * 100
).round(2)

# Save enriched version
feature_importance.to_csv('outputs/figures/data/feature_importance.csv', index=False)
print(f"    Feature importance exported to outputs/figures/data/feature_importance.csv")

# ---------------------------------------------------------------------------
# 11.2. Prediction Confidence
# ---------------------------------------------------------------------------

prediction_confidence = pd.DataFrame({
    'predicted_probability': y_proba_test,
    'actual_engaged': y_test.reset_index(drop=True).values,
    'predicted_engaged': y_pred_test,
    'correct': (y_pred_test == y_test.reset_index(drop=True).values).astype(int)
})

# Add confidence bins
prediction_confidence['confidence_bin'] = pd.cut(
    prediction_confidence['predicted_probability'], 
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
)

# Add prediction type (TP, FP, TN, FN)
def get_pred_type(row):
    if row['actual_engaged'] == 1 and row['predicted_engaged'] == 1:
        return 'True Positive'
    elif row['actual_engaged'] == 0 and row['predicted_engaged'] == 0:
        return 'True Negative'
    elif row['actual_engaged'] == 1 and row['predicted_engaged'] == 0:
        return 'False Negative'
    else:
        return 'False Positive'

prediction_confidence['prediction_type'] = prediction_confidence.apply(get_pred_type, axis=1)

# Save
prediction_confidence.to_csv('outputs/figures/data/prediction_confidence.csv', index=False)
print(f"    Prediction confidence exported to outputs/figures/data/prediction_confidence.csv")

# ---------------------------------------------------------------------------
# 11.3. Performance Summary
# ---------------------------------------------------------------------------

baseline_accuracy = y_test.value_counts().max() / len(y_test)

performance_metrics = pd.DataFrame({
    'metric': [
        'Test AUC',
        'Test Accuracy', 
        'Validation AUC',
        'Validation Accuracy',
        'CV AUC (Train)',
        'Baseline Accuracy',
        'Test F1 Score',
        'Test Precision (ENGAGED)',
        'Test Recall (ENGAGED)',
        'Test Precision (DISENGAGED)',
        'Test Recall (DISENGAGED)'
    ],
    'value': [
        roc_auc_score(y_test, y_proba_test),
        accuracy_score(y_test, y_pred_test),
        val_auc,
        accuracy_score(y_val, y_pred_val),
        best_score,  # CV AUC from grid search
        baseline_accuracy,
        f1_score(y_test, y_pred_test),
        precision_score(y_test, y_pred_test, pos_label=1),
        recall_score(y_test, y_pred_test, pos_label=1),
        precision_score(y_test, y_pred_test, pos_label=0),
        recall_score(y_test, y_pred_test, pos_label=0)
    ]
})

performance_metrics['value'] = performance_metrics['value'].round(3)
performance_metrics.to_csv('outputs/figures/data/performance_metrics.csv', index=False)
print(f"    Performance metrics exported to outputs/figures/data/performance_metrics.csv")

# ---------------------------------------------------------------------------
# 11.4. Category-Level Feature Importance
# ---------------------------------------------------------------------------

category_importance = feature_importance.groupby('category').agg({
    'importance': 'sum',
    'feature': 'count'
}).reset_index()

category_importance.columns = ['category', 'total_importance', 'num_features']
category_importance = category_importance.sort_values('total_importance', ascending=False)
category_importance['importance_pct'] = (
    category_importance['total_importance'] / 
    category_importance['total_importance'].sum() * 100
).round(2)

category_importance.to_csv('outputs/figures/data/category_importance.csv', index=False)
print(f"    Category importance exported to outputs/figures/data/category_importance.csv")

print("\nAll visualization data exported to outputs/figures/data/")
print("  Ready for Tableau dashboard!")