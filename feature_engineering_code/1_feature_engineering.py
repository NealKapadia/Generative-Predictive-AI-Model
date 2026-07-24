import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors
from mordred import Calculator, descriptors
 
 
warnings.filterwarnings("ignore")
 
 
# ============================================================
# Exact project locations
# ============================================================
 
BASE_DIR = Path(__file__).resolve().parent
 
INPUT_CSV = BASE_DIR / "s1.csv"
OUTPUT_CSV = BASE_DIR / "cleaned_final_features.csv"
PARTIAL_CSV = BASE_DIR / "cleaned_final_features_partial.csv"
FAILURE_CSV = BASE_DIR / "feature_engineering_failures.csv"
 
TMP_FOLDER = BASE_DIR / "dft_tmp"
TMP_FOLDER.mkdir(parents=True, exist_ok=True)
 
XTB_CMD = shutil.which("xtb")
 
 
# ============================================================
# RDKit descriptors
# ============================================================
 
def compute_rdkit_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
 
    if mol is None:
        raise ValueError("RDKit could not parse the SMILES.")
 
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
    }
 
 
# ============================================================
# Generate a 3D structure safely
# ============================================================
 
def generate_3d(smiles: str, xyz_path: Path):
    mol = Chem.MolFromSmiles(smiles)
 
    if mol is None:
        return False, "Invalid SMILES"
 
    mol = Chem.AddHs(mol)
 
    try:
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = 42
 
        embed_status = AllChem.EmbedMolecule(
            mol,
            parameters
        )
 
        # Retry using random coordinates when normal embedding fails
        if embed_status != 0:
            embed_status = AllChem.EmbedMolecule(
                mol,
                randomSeed=42,
                useRandomCoords=True
            )
 
        if embed_status != 0:
            return False, "RDKit 3D embedding failed"
 
        # Only use UFF when parameters exist for every atom
        if AllChem.UFFHasAllMoleculeParams(mol):
            try:
                AllChem.UFFOptimizeMolecule(
                    mol,
                    maxIters=500
                )
            except Exception as error:
                print(f"  UFF optimization skipped: {error}")
        else:
            print("  UFF parameters unavailable; using embedded geometry.")
 
        conformer = mol.GetConformer()
 
        xyz_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
 
        with xyz_path.open("w", encoding="utf-8") as file:
            file.write(f"{mol.GetNumAtoms()}\n")
            file.write(f"{smiles}\n")
 
            for atom in mol.GetAtoms():
                position = conformer.GetAtomPosition(
                    atom.GetIdx()
                )
 
                file.write(
                    f"{atom.GetSymbol()} "
                    f"{position.x:.6f} "
                    f"{position.y:.6f} "
                    f"{position.z:.6f}\n"
                )
 
        return True, None
 
    except Exception as error:
        return False, str(error)
 
 
# ============================================================
# Run xTB safely
# ============================================================
 
def run_xtb(
    xyz_path: Path,
    log_path: Path,
    n_cores: int = 4
):
    if XTB_CMD is None:
        return False, "xTB was not found in PATH"
 
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(n_cores)
 
    command = [
        XTB_CMD,
        xyz_path.name,
        "--gfn",
        "2"
    ]
 
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                cwd=str(xyz_path.parent),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                env=environment,
                timeout=600
            )
 
        if result.returncode != 0:
            return False, (
                f"xTB returned code {result.returncode}"
            )
 
        return True, None
 
    except subprocess.TimeoutExpired:
        return False, "xTB calculation timed out"
 
    except Exception as error:
        return False, str(error)
 
 
# ============================================================
# Read xTB output
# ============================================================
 
def parse_xtb_log(log_path: Path) -> dict:
    properties = {}
 
    if not log_path.exists():
        return properties
 
    text = log_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )
 
    patterns = {
        "homo_lumo_gap_eV": (
            r"HOMO-LUMO GAP\s+([-\d.]+)\s+eV"
        ),
        "dipole_tot_D": (
            r"molecular dipole:[\s\S]*?"
            r"full:\s+[-\d.]+\s+[-\d.]+\s+"
            r"[-\d.]+\s+([-\d.]+)"
        ),
    }
 
    for property_name, pattern in patterns.items():
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )
 
        if match:
            try:
                properties[property_name] = float(
                    match.group(1)
                )
            except ValueError:
                pass
 
    return properties
 
 
# ============================================================
# Save partial progress
# ============================================================
 
def save_partial(records):
    if records:
        pd.DataFrame(records).to_csv(
            PARTIAL_CSV,
            index=False
        )
 
 
# ============================================================
# Main workflow
# ============================================================
 
def main():
    print("=" * 70)
    print("STARTING FEATURE ENGINEERING")
    print("=" * 70)
 
    print("Input:", INPUT_CSV)
    print("Final output:", OUTPUT_CSV)
    print("Partial output:", PARTIAL_CSV)
    print("xTB:", XTB_CMD)
    print()
 
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file was not found: {INPUT_CSV}"
        )
 
    dataframe = pd.read_csv(INPUT_CSV)
 
    if "Additive_SMILES" not in dataframe.columns:
        raise KeyError(
            "s1.csv does not contain an "
            "'Additive_SMILES' column."
        )
 
    mordred_calculator = Calculator(
        descriptors,
        ignore_3D=True
    )
 
    records = []
    failures = []
 
    for position, (index, row) in enumerate(
        dataframe.iterrows(),
        start=1
    ):
        smiles = str(
            row["Additive_SMILES"]
        ).strip()
 
        print(
            f"Processing {position}/{len(dataframe)}: "
            f"{smiles}"
        )
 
        # Always begin with the original experimental data
        merged_record = row.to_dict()
 
        try:
            mol = Chem.MolFromSmiles(smiles)
 
            if mol is None:
                raise ValueError("Invalid SMILES")
 
            # RDKit descriptors
            try:
                rdkit_features = (
                    compute_rdkit_descriptors(smiles)
                )
 
                merged_record.update(rdkit_features)
 
            except Exception as error:
                print(
                    f"  RDKit descriptors failed: {error}"
                )
 
                failures.append({
                    "index": index,
                    "smiles": smiles,
                    "stage": "RDKit descriptors",
                    "error": str(error),
                })
 
            # Mordred descriptors
            try:
                mordred_result = mordred_calculator(mol)
 
                mordred_features = {
                    str(descriptor_name): value
                    for descriptor_name, value
                    in mordred_result.asdict().items()
                }
 
                merged_record.update(
                    mordred_features
                )
 
            except Exception as error:
                print(
                    f"  Mordred descriptors failed: {error}"
                )
 
                failures.append({
                    "index": index,
                    "smiles": smiles,
                    "stage": "Mordred",
                    "error": str(error),
                })
 
            # Individual folder for each xTB calculation
            molecule_folder = (
                TMP_FOLDER / f"molecule_{index:04d}"
            )
 
            molecule_folder.mkdir(
                parents=True,
                exist_ok=True
            )
 
            xyz_path = molecule_folder / "molecule.xyz"
            log_path = molecule_folder / "xtb.log"
 
            structure_ok, structure_error = generate_3d(
                smiles,
                xyz_path
            )
 
            if not structure_ok:
                print(
                    f"  3D generation skipped: "
                    f"{structure_error}"
                )
 
                failures.append({
                    "index": index,
                    "smiles": smiles,
                    "stage": "3D generation",
                    "error": structure_error,
                })
 
            else:
                xtb_ok, xtb_error = run_xtb(
                    xyz_path,
                    log_path,
                    n_cores=4
                )
 
                if xtb_ok:
                    xtb_features = parse_xtb_log(
                        log_path
                    )
 
                    merged_record.update(
                        xtb_features
                    )
 
                    print(
                        "  xTB completed; "
                        f"{len(xtb_features)} properties parsed."
                    )
 
                else:
                    print(
                        f"  xTB skipped or failed: "
                        f"{xtb_error}"
                    )
 
                    failures.append({
                        "index": index,
                        "smiles": smiles,
                        "stage": "xTB",
                        "error": xtb_error,
                    })
 
        except Exception as error:
            # This catches any unexpected molecule-level error.
            # The workflow continues with the next molecule.
            print(
                f"  Molecule failed but workflow "
                f"will continue: {error}"
            )
 
            failures.append({
                "index": index,
                "smiles": smiles,
                "stage": "general molecule processing",
                "error": str(error),
            })
 
        finally:
            # Always retain the original row, even when descriptors fail
            records.append(merged_record)
 
            # Save after every molecule
            save_partial(records)
 
            print(
                f"  Partial results saved: "
                f"{len(records)} rows\n"
            )
 
    if not records:
        raise RuntimeError(
            "No records were generated."
        )
 
    output_dataframe = pd.DataFrame(records)
 
    output_dataframe.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )
 
    # Remove columns containing no usable values
    output_dataframe = output_dataframe.dropna(
        axis=1,
        how="all"
    )
 
    # Keep these columns as identifiers/text
    protected_columns = [
        column
        for column in [
            "Additive_SMILES",
            "id"
        ]
        if column in output_dataframe.columns
    ]
 
    numeric_columns = (
        output_dataframe.columns.difference(
            protected_columns
        )
    )
 
    output_dataframe[numeric_columns] = (
        output_dataframe[numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
    )
 
    # Keep the original workflow's simple imputation method
    output_dataframe[numeric_columns] = (
        output_dataframe[numeric_columns]
        .fillna(0)
    )
 
    output_dataframe.to_csv(
        OUTPUT_CSV,
        index=False
    )
 
    pd.DataFrame(failures).to_csv(
        FAILURE_CSV,
        index=False
    )
 
    print()
    print("=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)
    print("Final output:", OUTPUT_CSV)
    print("File exists:", OUTPUT_CSV.exists())
    print("Rows saved:", len(output_dataframe))
    print("Columns saved:", len(output_dataframe.columns))
    print("Failures recorded:", len(failures))
    print("Failure report:", FAILURE_CSV)
 
 
if __name__ == "__main__":
    main()