# Exploratory characterization of chemical inhomogeneity in multicomponent alloys using information-theoretic self-supervised representation learning

Code and data-processing workflow used for self-supervised graph representation learning of Monte Carlo configurations of CoCuFeNiZn high-entropy alloys.

## Setup

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The manuscript dataset is provided separately as `manuscript_data.zip`.

Extract it into the repository root:

```text
SSL_Exploratory_Characterization/
├── config.yaml
├── manuscript_data/
├── notebooks/
└── src/
```

The default `config.yaml` is configured to use:

```yaml
data:
  root_path: manuscript_data
```

## Manuscript data

Each temperature directory contains:

```text
step50000.txt
recheck_energy.txt
```

`step50000.txt` contains the Monte Carlo atomic configuration used for graph construction.

`recheck_energy.txt` contains the Warren–Cowley data used in the manuscript analysis.

## Analysis

The manuscript analysis is contained in:

```text
notebooks/analysis.ipynb
```

## Citation

Associated manuscript: **[to be added]**
