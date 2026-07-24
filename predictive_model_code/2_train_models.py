import os
import pandas as pd
import numpy as np
import joblib
import optuna
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Use the matplotlib backend for Optuna to avoid needing plotly/kaleido
from optuna.visualization.matplotlib import plot_optimization_history

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

os.makedirs("saved_models", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)
sns.set_theme(style="whitegrid")

def main():
    print("Loading data...")
    df = pd.read_csv("cleaned_final_features.csv")
    TARGET = "CE_aver. (%)"
    drop_cols = ["CE_1 (%)", "CE_2 (%)", "CE_3 (%)", TARGET, "Additive_SMILES", "id"]
    drop_cols_existing = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols_existing).apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.20, random_state=42)

    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    def objective_xgb(trial):
        params = {
            "device": "cpu", 
            "objective": "reg:squarederror", 
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=False)
        return r2_score(y_val, model.predict(X_val_p))

    print("Running XGBoost Optuna Tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_xgb, timeout=120)
    
    # --- VISUALIZATION 1: Optuna History ---
    fig_optuna = plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig("visualizations/optuna_history.png", dpi=300)
    plt.close()

    best_params = study.best_trial.params
    best_params.update({"device": "cpu", "objective": "reg:squarederror", "random_state": 42})
    
    final_model = xgb.XGBRegressor(**best_params)
    X_trainval_p = preprocessor.fit_transform(X_trainval)
    final_model.fit(X_trainval_p, y_trainval)

    y_pred = final_model.predict(X_test_p)
    final_r2 = r2_score(y_test, y_pred)
    print(f"Hold-out Test R2: {final_r2:.4f}")

    # --- VISUALIZATION 2: Predicted vs Actual ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', color='#4e79a7')
    
    # y=x reference line
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title('XGBoost: Predicted vs Actual Coulombic Efficiency', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual CE (%)', fontsize=12)
    ax.set_ylabel('Predicted CE (%)', fontsize=12)
    
    ax.text(0.05, 0.95, f'$R^2 = {final_r2:.4f}$', transform=ax.transAxes,
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, ec='gray'))
    
    plt.tight_layout()
    plt.savefig("visualizations/predicted_vs_actual.png", dpi=300)
    plt.close()

    joblib.dump(preprocessor, "saved_models/preprocessor.joblib")
    joblib.dump(final_model, "saved_models/xgb_best_model.joblib")
    print("Model, preprocessor, and plots saved successfully.")

if __name__ == "__main__":
    main()