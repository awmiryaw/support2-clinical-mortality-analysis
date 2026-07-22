# Clinical Mortality Prediction using Machine Learning

An end-to-end data science project focused on predicting in-hospital mortality using the SUPPORT2 clinical dataset. The project covers data preprocessing, exploratory analysis, visualization, feature engineering, and machine learning.

## Features

- Data cleaning and preprocessing
- Data quality assessment
- Exploratory data analysis (EDA)
- Statistical data visualization
- Feature engineering
- Machine learning classification
- Model evaluation and optimization

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotnine
- Jupyter Notebook
- Git

## Repository Structure

- `1.cleaning_and_tidying.ipynb` – Data cleaning and preprocessing
- `2.data_quality_assess.ipynb` – Data quality assessment
- `3.data_visualization.ipynb` – Exploratory analysis and visualization
- `4.pre_process_and_data_analysis.ipynb` – Machine learning pipeline
- `requirements.txt` – Project dependencies

## Machine Learning Pipeline

The prediction workflow includes:

- Data preprocessing
- Missing value handling
- Categorical feature encoding
- Feature selection
- Train/test split
- Random Forest classification
- Stratified cross-validation
- Hyperparameter optimization
- Performance evaluation

## Installation

Clone the repository:

```bash
git clone https://github.com/awmiryaw/support2-clinical-mortality-analysis.git
cd support2-clinical-mortality-analysis
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in numerical order.

## Dataset

## Dataset

This project uses the SUPPORT2 dataset from the UCI Machine Learning Repository, containing information from over 9,000 hospitalized patients. The dataset is used for educational and research purposes to develop predictive models for in-hospital mortality.
Download the dataset and place it in the project root as `dataset.csv` before running the notebooks.
