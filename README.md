# SUPPORT2 Clinical Mortality Analysis

An end-to-end data science project exploring the SUPPORT2 clinical dataset and developing a machine-learning pipeline to predict in-hospital mortality.

## Overview

The project analyses clinical and demographic information from more than 9,000 seriously ill hospitalised patients.

The workflow includes:

- data cleaning and tidying;
- data quality assessment;
- exploratory data analysis;
- statistical visualization;
- feature preprocessing;
- machine-learning classification.

The prediction target is `hospdead`, which indicates whether a patient died during hospitalisation.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotnine
- Jupyter Notebook
- Git

## Project workflow

### 1. Data cleaning and tidying

The raw dataset was inspected and converted into a consistent, analysis-ready format.

The process included:

- missing-value analysis;
- duplicate detection;
- validation of categorical and numerical values;
- correction of inconsistent records;
- preparation of cleaned datasets.

### 2. Data quality assessment

The dataset was evaluated for completeness, consistency, validity, and possible analytical risks.

This included identifying:

- variables with substantial missingness;
- unusual or invalid values;
- class imbalance;
- potential target leakage;
- variables requiring transformation or encoding.

### 3. Data visualization

Visual analyses were produced to investigate:

- patient counts across disease categories;
- hospital survival and mortality by disease;
- mortality rates by sex and race;
- healthcare costs across demographic groups;
- relationships between clinical and demographic variables.

### 4. Mortality prediction

A classification pipeline was created to predict in-hospital mortality.

The modelling workflow included:

- feature selection;
- missing-value preprocessing;
- categorical encoding;
- train-test splitting;
- Random Forest classification;
- stratified cross-validation;
- hyperparameter optimisation;
- model evaluation.

## Repository contents

- `1.cleaning_and_tidying.ipynb` — data cleaning and tidying
- `2.data_quality_assess.ipynb` — data quality assessment
- `3.data_visualization.ipynb` — exploratory analysis and visualization
- `4.pre_process_and_data_analysis.ipynb` — preprocessing and mortality prediction
- `requirements.txt` — Python dependencies

## Running the project

Install the required packages:

```bash
pip install -r requirements.txt
