import os
import pandas as pd
from rdkit import Chem
# The mordred library is now required to calculate the specific descriptors your model needs.
# You can install it with: pip install mordred
from mordred import Calculator, descriptors
import joblib
import numpy as np
from tqdm import tqdm
import xgboost as xgb
import math
import numbers

# --- Visualization Libraries ---
# I've added these libraries to generate plots.
# You can install them with: pip install matplotlib seaborn
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import Draw

def _coerce_number(x):
    """
    Return a finite float or 0.0.
    Non-numeric, NaN, or ±inf will be converted to 0.0.
    """
    if isinstance(x, (int, float)) and math.isfinite(x):
        return float(x)
    return 0.0

# --- 1. Configuration ---
PREDICTIVE_MODEL_PATH = os.path.join("xgboost_model.joblib")
SOURCE_DATA_FILE = "output.csv" # Used to define the feature set
TOP_N_TO_DISPLAY = 20

# --- Generative Screening Settings ---
GENERATIVE_MODEL_DIR = "MODEL_DIR"
NUM_MOLECULES_TO_GENERATE = 500
SCREENING_ZN_MOLE = 1.0
SCREENING_ADDITIVE_MOLE = 5.0
SCREENING_LOG_MOLAR_RATIO = -2.0


# --- 2. Feature Calculation Setup (Based on SOURCE_DATA_FILE) ---

def get_feature_names_from_source(source_file_path):
    """
    Loads the source CSV to get the exact list of feature columns the model was trained on.
    """
    print(f"Reading feature names from '{source_file_path}'...")
    try:
        df = pd.read_csv(source_file_path)
        
        # Find the SMILES column to exclude it from features
        smiles_col = None
        for name in ['SMILES', 'smiles', 'canonical_smiles']:
            if name in df.columns:
                smiles_col = name
                break
        
        TARGET = "CE_aver. (%)"
        # Define all columns that are NOT features
        cols_to_drop = {"#", "CE_1 (%)", "CE_2 (%)", "CE_3 (%)", TARGET}
        if smiles_col:
            cols_to_drop.add(smiles_col)
            
        # We also need to remove the experimental conditions, as they are prepended manually.
        experimental_cols = {"Zn_mole (mmol)", "Additive_mole (%)", "LogMolarRatio"}
        cols_to_drop.update(experimental_cols)

        # The remaining columns are our molecular descriptor features, in the correct order
        feature_names = [col for col in df.columns if col not in cols_to_drop]
        print(f"Successfully loaded {len(feature_names)} molecular descriptor feature names.")
        return feature_names
    except FileNotFoundError:
        print(f"FATAL: Source data file '{source_file_path}' not found. Cannot determine feature set.")
        return None
    except Exception as e:
        print(f"FATAL: Error reading feature names from '{source_file_path}': {e}")
        return None

# Load the feature names that the predictive model expects
MOLECULAR_FEATURE_NAMES = get_feature_names_from_source(SOURCE_DATA_FILE)

# Create a Mordred calculator instance. This is done once for efficiency.
mordred_calculator = Calculator(descriptors, ignore_3D=True)

# --- This map helps translate between training column names and mordred names ---
MORDRED_NAME_MAP = {
    'MolWt': 'MW', 'LogP': 'LogP', 'TPSA': 'TopoPSA', 'HBD': 'nHBDon',
    'HBA': 'nHBAcc', 'RotB': 'nRot', 'Rings': 'nRing', 'TopoPSA(NO)': 'TopoPSA'
}

def calculate_features_for_screening(smiles_string, experimental_conditions):
    """
    Calculates the full feature vector for a newly generated SMILES string.
    It uses the MOLECULAR_FEATURE_NAMES list as the ground truth to ensure the
    vector matches the model's training data.
    """
    if not MOLECULAR_FEATURE_NAMES:
        print("FATAL: Feature names not loaded. Cannot calculate features.")
        return None
        
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    # Calculate all mordred descriptors for the molecule.
    calculated_descriptors_dict = mordred_calculator(mol).asdict()

    # Build the molecular descriptor part of the feature vector in the correct order.
    molecular_feature_vector = []
    for desc_name in MOLECULAR_FEATURE_NAMES:
        # If the name is in our map, use the mordred name, otherwise use the name directly.
        mordred_name = MORDRED_NAME_MAP.get(desc_name, desc_name)
        value = calculated_descriptors_dict.get(mordred_name)
        
        # CORRECTED LOGIC: Append the sanitized value only ONCE.
        molecular_feature_vector.append(_coerce_number(value))

    # The final feature vector must start with the experimental conditions.
    feature_list = [
        experimental_conditions['zn_mole'],
        experimental_conditions['additive_mole'],
        experimental_conditions['log_molar_ratio'],
        *molecular_feature_vector
    ]
    
    return np.array(feature_list).reshape(1, -1)


# --- 3. Visualization Functions ---

def plot_feature_importance(model, feature_names):
    """
    Creates and saves a bar plot of the top 20 feature importances from the trained model.
    """
    print("Generating feature importance plot...")
    try:
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(20) # Display top 20

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title('Top 20 Feature Importances for the Predictive Model')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature Name')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Saved feature importance plot to 'feature_importance.png'")
        plt.close()
    except Exception as e:
        print(f"\nWarning: Could not generate feature importance plot. Error: {e}")


def plot_score_distribution(df):
    """
    Creates and saves a histogram of the predicted scores.
    """
    print("\nGenerating predicted score distribution plot...")
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Predicted_Score'], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of Predicted Scores for {len(df)} Molecules')
        plt.xlabel('Predicted Score')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig('score_distribution.png')
        print("Saved score distribution plot to 'score_distribution.png'")
        plt.close()
    except Exception as e:
        print(f"\nWarning: Could not generate score distribution plot. Error: {e}")

def plot_top_molecules(df, top_n):
    """
    Creates and saves a grid image of the top N molecules.
    """
    print("\nGenerating image grid of top molecules...")
    try:
        top_df = df.head(top_n).copy()
        top_df['Molecule'] = top_df['SMILES'].apply(Chem.MolFromSmiles)
        
        # Filter out any SMILES that failed to convert
        top_df.dropna(subset=['Molecule'], inplace=True)

        legends = [f"Score: {score:.2f}" for score in top_df['Predicted_Score']]
        img = Draw.MolsToGridImage(
            top_df['Molecule'].tolist(),
            molsPerRow=4,
            subImgSize=(250, 250),
            legends=legends
        )
        img.save('top_molecules.png')
        print(f"Saved image of top {len(top_df)} molecules to 'top_molecules.png'")
    except Exception as e:
        print(f"\nWarning: Could not generate molecule grid image. Error: {e}")


def screen_with_models():
    """
    Executes the generative screening pipeline:
    1. Loads generative and predictive models.
    2. Generates new candidate molecules.
    3. Calculates features for each new molecule.
    4. Predicts their scores and ranks them.
    5. VISUALIZES the results.
    """
    print("--- Starting Generative Screening Workflow ---")

    # --- Step 1: Load Models ---
    print("\n--- Step 1: Loading Pre-trained Models ---")
    try:
        predictive_model = joblib.load(PREDICTIVE_MODEL_PATH)
        print(f"Predictive (XGBoost) model '{PREDICTIVE_MODEL_PATH}' loaded successfully.")
        from molecule_generation import load_model_from_directory
    except Exception as e:
        print(f"FATAL: Failed to load a required library or model file: {e}")
        return

    # --- VIZ: Plot Feature Importances ---
    experimental_cols = ["Zn_mole (mmol)", "Additive_mole (%)", "LogMolarRatio"]
    full_feature_names = experimental_cols + MOLECULAR_FEATURE_NAMES
    plot_feature_importance(predictive_model, full_feature_names)

    # --- Step 2: Generate Candidate Molecules ---
    print(f"\n--- Step 2: Generating {NUM_MOLECULES_TO_GENERATE} candidate molecules ---")
    generated_molecules = []
    try:
        with load_model_from_directory(GENERATIVE_MODEL_DIR) as generative_model:
            print("Generative (MoLeR) model loaded successfully.")
            generated_molecules = generative_model.sample(num_samples=NUM_MOLECULES_TO_GENERATE)
    except Exception as e:
        print(f"FATAL: Could not load or run the generative model. Error: {e}")
        return

    if not generated_molecules:
        print("FATAL: Molecule generation failed.")
        return
    print(f"Successfully generated {len(generated_molecules)} unique molecules.")

    # --- Step 3 & 4: Calculate Features and Predict Scores ---
    print("\n--- Step 3 & 4: Screening generated molecules ---")
    experimental_conditions = {
        'zn_mole': SCREENING_ZN_MOLE,
        'additive_mole': SCREENING_ADDITIVE_MOLE,
        'log_molar_ratio': SCREENING_LOG_MOLAR_RATIO
    }
    
    screened_results = []
    for smiles in tqdm(generated_molecules, desc="Screening Molecules"):
        features = calculate_features_for_screening(smiles, experimental_conditions)
        if features is not None:
            try:
                score = predictive_model.predict(features)[0]
                screened_results.append({"SMILES": smiles, "Predicted_Score": float(score)})
            except Exception as e:
                print(f"\nWarning: Could not score SMILES '{smiles}'. Error: {e}")
                if features is not None:
                    print(f"Model expected {predictive_model.n_features_in_} features, but got {features.shape[1]}.")
                break 

    if not screened_results:
        print("FATAL: Screening failed. No valid molecules could be scored.")
        return

    # --- Step 5: Rank and Display Results ---
    print("\n--- Step 5: Ranking Candidates ---")
    results_df = pd.DataFrame(screened_results)
    results_df.sort_values(by="Predicted_Score", ascending=False, inplace=True)
    
    print(f"\n--- Top {TOP_N_TO_DISPLAY} Generated Molecules (for conditions: {experimental_conditions}) ---")
    print(results_df.head(TOP_N_TO_DISPLAY).to_string())
    
    results_csv_path = "generative_screening_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nFull screening results saved to '{results_csv_path}'")
    
    # --- VIZ: Plot Score Distribution and Top Molecules ---
    plot_score_distribution(results_df)
    plot_top_molecules(results_df, TOP_N_TO_DISPLAY)

    print("\n--- Generative Screening Workflow Finished ---")


if __name__ == "__main__":
    # This script will generate new molecules and then score them.
    if MOLECULAR_FEATURE_NAMES:
        screen_with_models()
    else:
        print("\nExecution halted because the feature names could not be loaded from the source CSV.")