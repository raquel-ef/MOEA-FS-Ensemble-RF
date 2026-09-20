# Multi-objective evolutionary feature selection for ensemble learning with random forests in time series forecasting

## Overview
This repository provides a novel multi-objective evolutionary feature selection (MOEFS) framework designed to enhance predictive accuracy, generalization, and interpretability in Time Series Forecasting (TSF). By partitioning training data and leveraging Multi-Objective Evolutionary Algorithms (MOEAs), the framework generates a diverse Pareto front of Random Forest models combined through a stacking-based ensemble.

## Project Summary

High-dimensional feature spaces in time series forecasting frequently lead to overfitting and reduced model interpretability. This project addresses these challenges by reformulating feature selection into a multi-objective optimization problem:

- **Dataset Partitioning**: The training dataset is split into distinct temporal partitions.
- **Multi-Objective Optimization**: Each partition corresponds to an objective function measuring model error (RMSE) for a given binary feature mask.
- **Pareto Ensemble Generation**: A single MOEA run discovers non-dominated feature subsets, producing diverse Random Forest (RF) base models.
- **Stacking Ensemble**: Predictions from all non-dominated RF models are aggregated through a meta-RF regressor to form the final forecast.
- **Selection-Frequency Interpretability**: A novel feature importance metric quantifies attribute relevance based on selection frequencies across the Pareto-optimal front.

## Key contributions
- **Efficient Data-Partitioned MOEFS**: Eliminates the need for complex, task-specific algorithmic designs. A single MOEA run across data partitions yields a Pareto set of diverse feature subsets, mitigating overfitting across varying temporal dynamics.
- **Algorithm-Agnostic Design**: Compatible with established Pareto-based (NSGA-II, NSGA-III) and decomposition-based (MOEA/D) optimization algorithms.
- **Selection Frequency Feature Importance**: Introduces a simple yet robust interpretability metric based on feature occurrence across non-dominated solutions, showing high consistency against permutation importance and correlation analysis.


## Project Structure

| File          | Purpose |
| ------------- | ------------- |
| `MultiObjectiveRFWrapper.py`  | Defines the multi-objective problem interface (inheriting from Platypus `Problem`). Evaluates binary feature masks by computing $K$-fold cross-validated RMSE across each data partition.  |
| `synthetic-MOEA-FS-4part.py`  | Main execution pipeline with 4-partition experiments. Handles time-lag creation, normalization, parallelized MOEA execution, stacking meta-model training, and feature importance extraction.  |

## Installation
Ensure you have Python 3.10 installed and install the following dependencies:
```sh
pip install -r requirements.txt
```


## Usage
1. **Prepare the dataset**. Place time series data in .csv format in the `/DATASET/` directory. Ensure the raw time series is structured appropriately for autoregressive transformation (lag creation is performed using the lags function during runtime).

2. **Execute the MOEFS framework**. Run the evolutionary feature selection and ensemble learning pipeline by specifying the dataset name (without extension):

    ```sh
    python synthetic-MOEA-FS-4part.py <dataset_name>
    ```

    This script executes [MultiObjectiveRFWrapper.py](/MultiObjectiveRFWrapper.py) across $K=4$ data partitions using parallelized NSGA-III optimization.

3. **Evaluate and analyze results**. The results (including intermediate populations, non-dominated Pareto solutions, stacking meta-model predictions, and selection-frequency feature importance tuples) will be exported as .pickle files in the `./results_embedded/` directory for downstream analysis and baseline comparisons.

## Citation
If you use this software in your work, please include the following citation:
```
@article{espinosa2025multi,
  title={Multi-objective evolutionary feature selection for ensemble learning with random forests in time series forecasting},
  author={Espinosa, Raquel and S{\'a}nchez, Gracia and Palma, Jos{\'e} and Jim{\'e}nez, Fernando},
  journal={Swarm and Evolutionary Computation},
  volume={99},
  pages={102211},
  year={2025},
  publisher={Elsevier}
}
```

## License
[MIT License](/LICENSE)


