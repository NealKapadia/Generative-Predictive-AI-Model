import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor, StackingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV

import xgboost   as xgb
import lightgbm  as lgb
from catboost    import CatBoostRegressor
import optuna
# To save Optuna plots as static images, you may need to install kaleido:
# pip install kaleido
import optuna.visualization as vis

# -----------------------------------------------------------------------------
#  Utility for debug prints and plot saving
# -----------------------------------------------------------------------------
start_time = time.time()

# --- VISUALIZATION: Create a directory to save plots ---
if not os.path.exists("visualizations"):
    os.makedirs("visualizations")

def print_debug(msg):
    """Prints a message with the elapsed time."""
    elapsed = time.time() - start_time
    print(f"[{elapsed:6.1f}s] {msg}")

# -----------------------------------------------------------------------------
#  1) Load & split
# -----------------------------------------------------------------------------
print_debug("Loading data…")
# --- NOTE: Ensure you have an 'output.csv' file in the same directory ---
try:
    df = pd.read_csv("cleaned_final_features.csv").dropna(axis=1, how="all")
except FileNotFoundError:
    print_debug("ERROR: 'cleaned_final_features.csv' not found. Please place it in the correct directory.")
    # Create a dummy dataframe for demonstration purposes if the file doesn't exist
    data = np.random.rand(200, 15)
    cols = [f'feature_{i}' for i in range(12)] + ["CE_1 (%)", "CE_2 (%)", "CE_3 (%)"]
    df = pd.DataFrame(data, columns=cols)
    df['#'] = range(len(df))
    df['CE_aver. (%)'] = np.random.rand(len(df)) * 100
    print_debug("Using dummy data for demonstration.")


TARGET = "CE_aver. (%)"
drop_cols = [ "CE_1 (%)", "CE_2 (%)", "CE_3 (%)", TARGET]

# --- Get feature names before converting to numpy array for later plots ---
feature_names = df.drop(columns=drop_cols).columns.tolist()

X = (df
     .drop(columns=drop_cols)
     .apply(pd.to_numeric, errors="coerce")
     .fillna(0)
     .values)
y = df[TARGET].values

# --- NEW: Function to find outliers in the target variable ---
def find_target_outliers(y_data, target_name, threshold_std=3.0):
    """Identifies and prints outliers in the target variable based on standard deviation."""
    mean_y = np.mean(y_data)
    std_y = np.std(y_data)
    
    outlier_threshold_upper = mean_y + threshold_std * std_y
    outlier_threshold_lower = mean_y - threshold_std * std_y
    
    # Find indices of outliers
    outlier_indices = np.where((y_data > outlier_threshold_upper) | (y_data < outlier_threshold_lower))[0]
    
    print_debug(f"↓ Outlier Report for Target Variable '{target_name}' (Threshold: {threshold_std} StDev from mean):")
    if len(outlier_indices) > 0:
        for idx in outlier_indices:
            # This 'idx' corresponds to the original dataframe row index
            print(f"  - Outlier found at original index {df.index[idx]}: Value = {y_data[idx]:.2f} (Thresholds: <{outlier_threshold_lower:.2f} or >{outlier_threshold_upper:.2f})")
    else:
        print("  - No outliers found in the target variable distribution.")

# --- NEW: Find and report outliers in the initial target variable ---
find_target_outliers(y, TARGET)


# --- VISUALIZATION: Plot distribution of the target variable ---
print_debug("Generating target distribution plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(y, kde=True, ax=ax, bins=30)
ax.set_title(f'Distribution of Target Variable: {TARGET}', fontsize=16)
ax.set_xlabel(TARGET, fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig("visualizations/target_distribution.png", dpi=300)
plt.close(fig)
print_debug("Saved 'target_distribution.png'")


# 70% train+val, 30% test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
# from train+val carve 80/20 for actual tuning
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, random_state=42
)
print_debug(f"Shapes — train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

# -----------------------------------------------------------------------------
#  2) Helper to run an Optuna study
# -----------------------------------------------------------------------------
def run_optuna(objective, name, timeout_s):
    """Runs an Optuna study and saves visualization plots."""
    print_debug(f"→ Starting {name} tuning for {timeout_s}s …")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=timeout_s)
    
    best_value = study.best_trial.value
    print_debug(f"← {name} best R² = {best_value:.4f}")

    # --- VISUALIZATION: Generate and save Optuna plots ---
    try:
        # Plot 1: Optimization History
        fig_history = vis.plot_optimization_history(study)
        fig_history.update_layout(title=f'{name}: Optimization History', title_x=0.5)
        fig_history.write_image(f"visualizations/optuna_history_{name}.png", scale=2)
        print_debug(f"  Saved 'optuna_history_{name}.png'")

        # Plot 2: Parameter Importances
        if len(study.trials) > 1: # Importance can only be calculated with multiple trials
            fig_importance = vis.plot_param_importances(study)
            fig_importance.update_layout(title=f'{name}: Hyperparameter Importance', title_x=0.5)
            fig_importance.write_image(f"visualizations/optuna_importance_{name}.png", scale=2)
            print_debug(f"  Saved 'optuna_importance_{name}.png'")
        else:
            print_debug(f"  Skipping importance plot for {name} (only one trial).")

    except Exception as e:
        print_debug(f"  Could not generate Optuna plots for {name}. Error: {e}")

    return study.best_trial.params

# -----------------------------------------------------------------------------
#  3) XGBoost tuning
# -----------------------------------------------------------------------------
def objective_xgb(trial):
    params = {
        "tree_method": "auto",
        "device": "cuda",
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 150, 2000),
        "max_depth":    trial.suggest_int("max_depth", 4, 10),
        "learning_rate":trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":    trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    m = xgb.XGBRegressor(**params)
    m.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return r2_score(y_val, m.predict(X_val))


best_xgb = run_optuna(objective_xgb, "XGBoost", timeout_s=60)

# ---------------------------------------------------------------------
#  4) LightGBM tuning
# ---------------------------------------------------------------------
def objective_lgb(trial):
    params = {
        "device":             "gpu",
        "gpu_platform_id":    0,  # Adjust if you have multiple GPUs
        "gpu_device_id":      0,  # Adjust if you have multiple GPUs
        "objective":          "regression",
        "max_bin":            trial.suggest_int("max_bin", 31, 127),
        "n_estimators":       trial.suggest_int("n_estimators", 100, 1500),
        "max_depth":          trial.suggest_int("max_depth", 3, 12),
        "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":         trial.suggest_float("reg_lambda", 0.0, 1.0),
        "min_child_samples":  trial.suggest_int("min_child_samples", 5, 20),
        "random_state":       42,
        "verbosity":          -1,
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    return r2_score(y_val, m.predict(X_val))

best_lgb = run_optuna(objective_lgb, "LightGBM", timeout_s=60)

# -----------------------------------------------------------------------------
#  5) CatBoost tuning
# -----------------------------------------------------------------------------
#def objective_cat(trial):
#    params = {
#        "task_type":      "CPU", # Changed to CPU for wider compatibility
#        "iterations":     trial.suggest_int("iterations", 100, 1500),
#        "depth":          trial.suggest_int("depth", 3, 12),
#        "learning_rate":  trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
#        "l2_leaf_reg":    trial.suggest_float("l2_leaf_reg", 0.0, 5.0),
#        "random_seed":    42,
#        "verbose":        False,
#    }
#    m = CatBoostRegressor(**params)
#    m.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
#    return r2_score(y_val, m.predict(X_val))

#best_cat = run_optuna(objective_cat, "CatBoost", timeout_s=10)

# -----------------------------------------------------------------------------
#  6) RandomForest tuning
# -----------------------------------------------------------------------------
def objective_rf(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":        trial.suggest_int("max_depth", 3, 30),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state":     42,
        "n_jobs":           -1,
    }
    m = RandomForestRegressor(**params)
    m.fit(X_train, y_train)
    return r2_score(y_val, m.predict(X_val))

best_rf = run_optuna(objective_rf, "RandomForest", timeout_s=60)

# -----------------------------------------------------------------------------
#  7) ExtraTrees tuning
# -----------------------------------------------------------------------------
def objective_et(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":        trial.suggest_int("max_depth", 3, 30),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state":     42,
        "n_jobs":           -1,
    }
    m = ExtraTreesRegressor(**params)
    m.fit(X_train, y_train)
    return r2_score(y_val, m.predict(X_val))

best_et = run_optuna(objective_et, "ExtraTrees", timeout_s=60)

# -----------------------------------------------------------------------------
#  8) HistGradientBoosting tuning
# -----------------------------------------------------------------------------
def objective_hgb(trial):
    params = {
        "max_iter":       trial.suggest_int("max_iter", 100, 1000),
        "max_depth":      trial.suggest_int("max_depth", 3, 30),
        "learning_rate":  trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "random_state":   42,
    }
    m = HistGradientBoostingRegressor(**params)
    m.fit(X_train, y_train)
    return r2_score(y_val, m.predict(X_val))

best_hgb = run_optuna(objective_hgb, "HistGB", timeout_s=60)

# -----------------------------------------------------------------------------
#  9) MLP tuning
# -----------------------------------------------------------------------------
# def objective_mlp(trial):
#     params = {
#         "hidden_layer_sizes": tuple([trial.suggest_int(f"hl{i}", 10, 200)
#                                        for i in range(trial.suggest_int("n_layers", 1, 3))]),
#         "alpha":              trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
#         "learning_rate_init": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
#         "max_iter":           500,
#         "random_state":       42,
#     }
#     m = MLPRegressor(**params)
#     m.fit(X_train, y_train)
#     return r2_score(y_val, m.predict(X_val))
# 
# best_mlp = run_optuna(objective_mlp, "MLP", timeout_s=10)

# --- NEW: Function to find outliers in prediction plots ---
#def find_and_report_outliers(y_true, y_pred, model_name, threshold_std=3.0):
#    """Identifies and prints prediction outliers based on residuals."""
#   residuals = y_true - y_pred
#    mean_residual = np.mean(residuals)
#    std_residual = np.std(residuals)
#    
#    # Define outlier thresholds
#    outlier_threshold_upper = mean_residual + threshold_std * std_residual
#    outlier_threshold_lower = mean_residual - threshold_std * std_residual
#    
#    # Find indices of outliers within the test set
#    outlier_indices = np.where((residuals > outlier_threshold_upper) | (residuals < outlier_threshold_lower))[0]
#    
#    print_debug(f"  ↓ Outlier Report for {model_name} (Threshold: {threshold_std} StDev from mean residual):")
#    if len(outlier_indices) > 0:
#        for idx in outlier_indices:
#            # Note: The 'idx' here is the index within the 'y_test' array.
#            print(f"    - Outlier found at test set index {idx}:")
#            print(f"      Actual Value: {y_true[idx]:.2f}, Predicted Value: {y_pred[idx]:.2f}, Residual: {residuals[idx]:.2f}")
#    else:
#        print("    - No significant prediction outliers found.")

# -----------------------------------------------------------------------------
# 10) Individual Model Evaluation and Visualization
# -----------------------------------------------------------------------------
print_debug("Evaluating individual models and plotting predicted vs. actual...")

def plot_predicted_vs_actual(y_true, y_pred, model_name, r2):
    """Generates and saves a predicted vs. actual plot for a given model."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    
    # Add a y=x line for reference
    # Fix for ValueError: setting an array element with a sequence.
    flat_list = np.concatenate([y_true, y_pred, np.array(ax.get_xlim()), np.array(ax.get_ylim())])
    lims = [np.min(flat_list), np.max(flat_list)]

    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ax.set_title(f'{model_name}: Actual vs. Predicted (Test Set)', fontsize=16)
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    
    # Add R^2 score to the plot
    ax.text(0.05, 0.95, f'$R^2 = {r2:.4f}$', transform=ax.transAxes,
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            
    plt.tight_layout()
    plt.savefig(f"visualizations/predicted_vs_actual_{model_name}.png", dpi=300)
    plt.close(fig)
    print_debug(f"  Saved 'predicted_vs_actual_{model_name}.png'")

# Instantiate each model with its best params
models_to_evaluate = {
    "XGBoost": xgb.XGBRegressor(**best_xgb, device="cpu", random_state=42),
    "LightGBM": lgb.LGBMRegressor(**best_lgb, device="cpu", random_state=42),
    "RandomForest": RandomForestRegressor(**best_rf, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(**best_et, random_state=42),
    "HistGB": HistGradientBoostingRegressor(**best_hgb, random_state=42),
    # Uncomment the lines below if you run the tuning for CatBoost and MLP
    # "CatBoost": CatBoostRegressor(**best_cat, task_type="CPU", verbose=False, random_seed=42),
    # "MLP": MLPRegressor(**best_mlp, random_state=42)
}

# Loop through models to fit, evaluate, and plot
for name, model in models_to_evaluate.items():
    print_debug(f"  Evaluating {name}...")
    # Train the model on the full training+validation set
    model.fit(X_trainval, y_trainval)
    # Predict on the hold-out test set
    y_pred_test = model.predict(X_test)
    # Calculate the final R^2 score
    r2_test = r2_score(y_test, y_pred_test)
    print_debug(f"  → {name} Test R² = {r2_test:.4f}")
    
    # Generate and save the plot
    plot_predicted_vs_actual(y_test, y_pred_test, name, r2_test)

    # ▸ NEW: persist the trained model
    model_path = f"saved_models/{name}_best.joblib"
    dump(model, model_path)
    print_debug(f"  Saved {name} model → {model_path}")


# -----------------------------------------------------------------------------
# 11) Build & fit a stacking ensemble of all models
# -----------------------------------------------------------------------------
# print_debug("Building stacking ensemble…")
# # Instantiate each with its best params
# base_models = [
#     ("xgb", xgb.XGBRegressor(**best_xgb, device="cpu")),
#     ("lgb", lgb.LGBMRegressor(**best_lgb, device="cpu")),
#     ("cat", CatBoostRegressor(**best_cat, task_type="CPU", verbose=False)),
#     ("rf",  RandomForestRegressor(**best_rf)),
#     ("et",  ExtraTreesRegressor(**best_et)),
#     ("hgb", HistGradientBoostingRegressor(**best_hgb)),
#     ("mlp", MLPRegressor(**best_mlp)) # Added MLP to the stack
# ]
# stack = StackingRegressor(
#     estimators=base_models,
#     final_estimator=RidgeCV(),
#     cv=5,
#     n_jobs=-1, # Parallelize cross-validation
#     passthrough=False # Set to False to only see base model weights
# )
# stack.fit(X_trainval, y_trainval) # Train stack on the full training set
# print_debug("Stack trained.")

# -----------------------------------------------------------------------------
# 12) Final evaluation and visualizations
# -----------------------------------------------------------------------------
# print_debug("Evaluating on hold-out test set…")
# y_pred = stack.predict(X_test)
# final_r2 = r2_score(y_test, y_pred)
# print_debug(f"★ Test R² = {final_r2:.4f}")
# 
# # --- VISUALIZATION: Stacking Regressor Weights ---
# print_debug("Generating stacking weights plot...")
# # The coefficients of the final Ridge estimator show the weights of the base models
# stacking_weights = stack.final_estimator_.coef_
# model_names = [name for name, _ in base_models]
# 
# fig, ax = plt.subplots(figsize=(12, 7))
# sns.barplot(x=model_names, y=stacking_weights, ax=ax, palette='viridis')
# ax.set_title('Stacking Regressor - Base Model Weights', fontsize=16)
# ax.set_xlabel('Base Model', fontsize=12)
# ax.set_ylabel('Coefficient (Weight)', fontsize=12)
# ax.tick_params(axis='x', rotation=45)
# plt.tight_layout()
# plt.savefig("visualizations/stacking_weights.png", dpi=300)
# plt.close(fig)
# print_debug("Saved 'stacking_weights.png'")
# 
# 
# # --- VISUALIZATION: Actual vs. Predicted Plot ---
# print_debug("Generating actual vs. predicted plot...")
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
# # Add a y=x line for reference
# lims = [
#     np.min([ax.get_xlim(), ax.get_ylim()]),
#     np.max([ax.get_xlim(), ax.get_ylim()]),
# ]
# ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
# ax.set_aspect('equal')
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# ax.set_title('Final Model: Actual vs. Predicted Values', fontsize=16)
# ax.set_xlabel('Actual Values', fontsize=12)
# ax.set_ylabel('Predicted Values', fontsize=12)
# # Add R^2 score to the plot
# ax.text(0.05, 0.95, f'$R^2 = {final_r2:.4f}$', transform=ax.transAxes,
#         fontsize=14, verticalalignment='top',
#         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
# plt.tight_layout()
# plt.savefig("visualizations/actual_vs_predicted.png", dpi=300)
# plt.close(fig)
# print_debug("Saved 'actual_vs_predicted.png'")

print_debug("--- All visualizations saved in the 'visualizations' folder. ---")