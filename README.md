# iGRAIL v10 — Reproducible Experimental Framework

## iGRAIL: Blockchain-Orchestrated Generative Federated Learning for Privacy-Preserving Healthcare Analytics

This repository contains the experimental implementation of **iGRAIL**, a blockchain-orchestrated generative federated learning framework for privacy-preserving healthcare analytics.

The repository supports experiments with four healthcare datasets and compares three federated learning methods:

- FL
- BC-FL
- iGRAIL

The experiments evaluate predictive performance, communication-round behavior, client scalability, non-IID data distributions, differential privacy, security, attack robustness, and computational runtime.

The main Python file used to reproduce the final manuscript experiments is:

    igrail_experiments_v10.py

The final manuscript results should be generated using the `paper` preset. The `smoke` preset is provided only for quick validation of the environment and code.

---

## 1. Methods

### FL

FL uses standard sample-size weighted federated averaging.

### BC-FL

BC-FL uses cryptographic verification and blockchain-admitted trust-weighted aggregation.

### iGRAIL

iGRAIL extends the BC-FL security and trust layer with conditional WGAN-GP-based data augmentation and DP-SGD.

The manuscript should use these method definitions so that the mathematical description remains consistent with the experimental implementation.

---

## 2. Experimental Datasets

The final experiments use four datasets:

- Cancer
- Diabetes
- Healthcare
- Heart

The dataset configuration is defined in:

    DATASETS.json

The corresponding dataset files should be stored inside:

    data/

Recommended repository structure:

    igrail/
    |
    |-- igrail_experiments_v10.py
    |-- DATASETS.json
    |-- requirements.txt
    |-- README.md
    |
    |-- data/
    |   |-- cancer.csv
    |   |-- diabetes.csv
    |   |-- healthcare.csv
    |   |-- heart.csv
    |
    |-- results/
        |-- figures/
        |-- CSV result files

Before running the experiments, confirm that the dataset paths in `DATASETS.json` match the files inside the `data` directory.

If a dataset cannot legally be redistributed through GitHub, do not upload that dataset. Instead, provide its official source and preparation instructions and place the downloaded file in the location expected by `DATASETS.json`.

---

## 3. Creating the Python Environment

Python 3 should be installed before running the experiments.

To confirm the installed Python version:

    python3 --version

On some systems, the command may be:

    python --version

Create a virtual environment from the root directory of the repository:

    python3 -m venv .venv

On macOS or Linux, activate it using:

    source .venv/bin/activate

On Windows Command Prompt:

    .venv\Scripts\activate.bat

On Windows PowerShell:

    .venv\Scripts\Activate.ps1

After activation, the terminal should indicate that `.venv` is active.

---

## 4. Installing the Required Python Packages

Upgrade pip:

    python -m pip install --upgrade pip

Install the dependencies:

    pip install -r requirements.txt

The `requirements.txt` file should contain the dependencies required by `igrail_experiments_v10.py`.

For reproducibility, the GitHub release used for the manuscript should preserve the exact package versions used for the final experiment. Do not replace them with newer versions without rerunning and validating the experiments.

To record the exact environment after the final successful manuscript run, use:

    pip freeze > environment_freeze.txt

The resulting `environment_freeze.txt` can be included in the repository or archived with the manuscript release.

---

## 5. Verifying the Installation

After installing the dependencies, first perform a smoke test.

Run:

    python igrail_experiments_v10.py --preset smoke --datasets cancer diabetes healthcare heart

The smoke preset is intended only to confirm that:

- Python is configured correctly.
- Required packages can be imported.
- Dataset files can be loaded.
- The experiment pipeline can execute.
- Result directories can be created.
- Figures and CSV files can be generated.

The smoke test is not the final experiment.

**Do not report smoke-test results in the manuscript.**

---

## 6. Reproducing the Final Manuscript Experiment

After the smoke test completes successfully, run the full paper experiment:

    python igrail_experiments_v10.py --preset paper --datasets cancer diabetes healthcare heart

This is the main command for reproducing the final experimental results.

The paper preset uses the complete experimental configuration implemented in `igrail_experiments_v10.py`.

The final results should be taken only from this run.

---

## 7. Main Experimental Configuration

The final experiments evaluate FL, BC-FL, and iGRAIL.

The paper experiments use 20 actual federated learning communication rounds.

The main experiments use three clients unless the experiment specifically evaluates client scalability.

The client-scaling experiment evaluates:

    K = 2, 3, 4, 5, 6, 7, 8, 9, 10

Federated training begins at `K=2`.

If `K=1` appears on a figure axis, it is only a visual reference and is not a federated training configuration.

---

## 8. Non-IID Data Distribution

Non-IID client distributions are generated using a Dirichlet distribution.

The evaluated alpha values are:

    alpha = 0.1, 0.5, 1.0, 5.0

Smaller alpha values represent stronger heterogeneity among client data distributions.

Larger alpha values produce distributions closer to IID conditions.

---

## 9. Class-Imbalance Handling

The healthcare datasets contain different levels of class imbalance.

All compared methods use the same class-balanced binary cross-entropy loss.

This keeps the comparison between FL, BC-FL, and iGRAIL consistent and reduces the possibility that highly imbalanced datasets collapse into an all-negative classifier.

---

## 10. Differential Privacy Configuration

Differential privacy is evaluated for iGRAIL.

The evaluated delta values are:

    1e-6
    1e-7
    1e-8

The evaluated noise multipliers are:

    0.55
    0.75
    1.0
    1.35

The privacy budget epsilon is calculated using the RDP accountant implemented by the experiment.

FL and BC-FL are non-DP reference methods in this implementation. Therefore, fake epsilon values must not be assigned to FL or BC-FL.

Privacy comparisons should distinguish the non-private FL and BC-FL references from the differentially private iGRAIL configurations.

---

## 11. Predictive Metrics

The experiments evaluate the following predictive metrics:

- Accuracy
- Precision
- Recall
- Specificity
- Balanced Accuracy
- F1-score
- AUROC
- AUPRC
- MCC

Each final predictive figure uses only one metric on the y-axis.

The y-axis is labeled with the actual metric name and is never labeled only as `Score`.

---

## 12. Final Paper Figure Structure

The final manuscript figures are generated under:

    results/figures/13_FINAL_ONE_METRIC_MULTI_DATASET/

### A_client_scaling

For each metric:

    ClientScaling_Accuracy
    ClientScaling_Precision
    ClientScaling_Recall
    ClientScaling_Specificity
    ClientScaling_BalancedAcc
    ClientScaling_F1
    ClientScaling_AUROC
    ClientScaling_AUPRC
    ClientScaling_MCC

The x-axis represents the number of clients, `K`.

The evaluated values are:

    2, 3, 4, 5, 6, 7, 8, 9, 10

The axis can show ticks from 1 to 10, but `K=1` is only a visual reference.

### B_roundwise

For each metric:

    Roundwise_<metric>

The x-axis represents federated learning communication rounds.

The y-axis represents one predictive metric.

All four datasets and all three methods are included.

The paper run uses 20 actual communication rounds.

### C_selected_rounds

For each metric:

    SelectedRounds_<metric>

The selected communication rounds are:

    2, 5, 7, 10

All datasets and all methods are compared within the corresponding metric figure.

### D_noniid

For each metric:

    NonIID_<metric>

The x-axis represents Dirichlet alpha.

The figures compare all four datasets and all three methods under different levels of client-data heterogeneity.

### E_privacy

Two types of privacy figures are generated for each metric.

#### PrivacyTradeoff_<metric>

The x-axis represents measured epsilon.

The y-axis represents one selected predictive metric.

The curves represent each dataset under the evaluated delta configurations.

FL and BC-FL are not assigned epsilon values because they are not differentially private in this implementation.

#### PrivacyRounds_<metric>

The x-axis represents federated learning rounds.

The y-axis represents one predictive metric.

These figures contain dataset-specific FL references, dataset-specific BC-FL references, and representative iGRAIL privacy configurations.

### F_final_bars

For each metric:

    FinalDataset_<metric>_Bar

The x-axis contains:

    Cancer
    Diabetes
    Healthcare
    Heart

The bars represent:

    FL
    BC-FL
    iGRAIL

Bar charts are used because datasets are discrete categories rather than an ordered learning progression.

---

## 13. Multi-Dataset Comparison

Where appropriate, a single metric figure can contain curves for:

    Cancer-FL
    Cancer-BC-FL
    Cancer-iGRAIL

    Diabetes-FL
    Diabetes-BC-FL
    Diabetes-iGRAIL

    Healthcare-FL
    Healthcare-BC-FL
    Healthcare-iGRAIL

    Heart-FL
    Heart-BC-FL
    Heart-iGRAIL

This provides cross-dataset and cross-method comparison while keeping only one predictive quantity on the y-axis.

---

## 14. Curve Generation

Measured experimental data points remain visible using markers.

PCHIP interpolation is used only for drawing smooth curves through the measured points.

PCHIP does not replace or modify the experimental values stored in the CSV files.

The CSV files should therefore be treated as the primary numerical results.

Figures are visual representations of those results.

---

## 15. Runtime Evaluation

Runtime is separated into:

- Local processing
- Verification
- Aggregation
- Total runtime

Runtime is evaluated across different communication rounds.

The runtime comparison includes FL, BC-FL, and iGRAIL.

FL does not execute the additional verification stage used by the verification-enabled methods.

BC-FL adds cryptographic verification and trust-related operations.

iGRAIL adds its generative and privacy-preserving local processing on top of the BC-FL security and trust layer.

This separation allows the computational cost of each part of the framework to be examined independently.

---

## 16. Security and Attack Evaluation

The experimental framework also evaluates the behavior of the verification mechanism under security-related conditions.

The evaluated security scenarios include:

- Valid update
- Tampered update
- Replay
- Unauthorized update

Valid updates should be accepted.

Tampered, replayed, and unauthorized submissions should be rejected by the verification mechanism.

The attack-robustness experiment activates the attack beginning at communication round 6.

The attack results are used to compare conventional FL with the verification-enabled methods.

---

## 17. Reproducing the Results From a Fresh Clone

A researcher can reproduce the experiment using the following sequence.

Clone the repository:

    git clone <repository-url>

Enter the repository:

    cd <repository-directory>

Create the environment:

    python3 -m venv .venv

Activate it on macOS or Linux:

    source .venv/bin/activate

Upgrade pip:

    python -m pip install --upgrade pip

Install dependencies:

    pip install -r requirements.txt

Confirm that the required datasets are available under:

    data/

Run the validation experiment:

    python igrail_experiments_v10.py --preset smoke --datasets cancer diabetes healthcare heart

If the validation completes successfully, run the final paper experiment:

    python igrail_experiments_v10.py --preset paper --datasets cancer diabetes healthcare heart

The generated results can then be inspected under the `results` directory.

---

## 18. Reproducibility Checklist

Before reproducing the paper, confirm the following:

- `igrail_experiments_v10.py` is present.
- `DATASETS.json` is present.
- `requirements.txt` is present.
- All required datasets are available.
- Dataset paths match `DATASETS.json`.
- The Python virtual environment is active.
- Required packages have been installed.
- The smoke test completes without errors.
- The final experiment uses `--preset paper`.
- All four datasets are specified.
- Manuscript results are taken from the paper run rather than the smoke test.

---

## 19. Files Recommended for the Public Repository

The clean public repository should contain at least:

    README.md
    igrail_experiments_v10.py
    DATASETS.json
    requirements.txt
    data/
    results/

If the generated results are being released for exact manuscript verification, also retain the final CSV files used to construct the manuscript tables and figures.

An optional environment snapshot can also be included:

    environment_freeze.txt

Older experimental development files such as `igrail_experiments_v2.py`, `igrail_experiments_v3.py`, and other superseded versions are not required to reproduce the final v10 manuscript results.

---

## 20. Important Reproducibility Notes

Use `igrail_experiments_v10.py` for the final reported experiments.

Do not combine numerical results produced by older versions of the experiment code with v10 results.

Do not use smoke-test values in manuscript tables or figures.

Do not manually change generated CSV values before creating the manuscript tables.

Do not assign artificial privacy budgets to FL or BC-FL.

Do not describe BC-FL differently from its implementation in the experiment.

For exact numerical verification, use the generated CSV files rather than estimating values from plots.

---

## 21. Citation

If you use this implementation, experimental framework, or generated results in academic work, please cite the corresponding iGRAIL paper.

Paper:

**iGRAIL: Blockchain-Orchestrated Generative Federated Learning for Privacy-Preserving Healthcare Analytics**

The complete bibliographic citation can be added here after publication.

---

## 22. License

Add the selected software license before making the repository public.

For example, if an open-source license is selected, place the corresponding `LICENSE` file in the root directory and state the license here.

---

## 23. Contact

For questions about the implementation, experimental configuration, or reproduction of the reported results, please use the contact information provided in the corresponding paper or the GitHub repository.
