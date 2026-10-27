```markdown
# ConformalTriage: Verifiable Risk Control for Autonomous Pulmonary Nodule Sign-off

This repository contains the complete, end-to-end Python codebase and reproduction pipeline for the study: **"Verifiable Risk Control for Autonomous Pulmonary Nodule Sign-off: Consensus-Definition Leakage, CT Protocol Shift, and Finite-Sample Certification"**. 

We introduce **ConformalTriage**, a distribution-free safety audit toolkit designed to evaluate the deployment risks of autonomous AI triage in medical imaging (specifically, LIDC-IDRI and LNDb cohorts).

## 📂 Repository Structure
- `code/`: Core algorithms, conformal risk control (CRC) implementations, LTT PAC guarantees, and statistical evaluation scripts.
- `data/`: Placeholder for raw inputs (LIDC-IDRI metadata, LNDb annotations).
- `result/`: Generated CSV outputs, intermediate results, and final paper figures.

## 🚀 Reproducibility (Docker & Python)

We highly recommend using our Docker environment to reproduce the exact results. 

### 1. Environment Setup
```bash
conda create -n conformal_triage python=3.9
conda activate conformal_triage
pip install -r requirements.txt
```

### 2. Execution Pipeline
The experiments must be run in sequential order as they depend on upstream outputs.

```bash
# Core statistical audits & baseline calibrations (Sections 3.1 - 3.5)
python code/01_scorer_diagnostics.py
python code/02_definition_audit.py
...
# Pathology reconstruction and external validation (Sections 3.6 - 3.7)
python code/09_rebuild_nodule_level_pathology.py
python code/10_pathology_audit.py
python code/13_external_validation_lndb.py
...
```

## 📊 Data Availability
- **LIDC-IDRI**: Publicly available via [The Cancer Imaging Archive (TCIA)](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX).
- **LNDb**: Publicly available via [Zenodo](https://doi.org/10.5281/zenodo.8348419). 
*(Please download the corresponding CSVs and place them in the `data/` folder before running scripts 10 and 13.)*

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact
For questions regarding the code or the statistical framework, please contact:
**Yongjin Deng** (dengyj27@mail.sysu.edu.cn)
```
