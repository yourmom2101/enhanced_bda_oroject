# Property Price Prediction Model

## Overview
This project implements an advanced machine learning model for predicting property prices using historical sales data from 2002 to 2022. The model incorporates comprehensive data cleaning, feature engineering, and time series analysis to provide accurate price predictions.

## Features
- **Comprehensive Data Processing**
  - Handles data from 2002-2022
  - Robust cleaning and validation
  - Outlier detection and removal
  - Missing value handling

- **Advanced Feature Engineering**
  - Time-based features
  - Property characteristics
  - Market trend indicators
  - Location-based features

- **Model Architecture**
  - XGBoost-based regression
  - Time series cross-validation
  - Feature selection
  - Robust preprocessing pipeline

## Project Structure
```
property-price-prediction/
├── data/               # Data files
├── models/            # Saved model files
├── plots/             # Generated visualizations
├── notebooks/         # Jupyter notebooks
├── docs/              # Documentation
└── enhanced_property_model.py  # Main model script
```

## Performance Metrics
- R² Score: 0.9979
- Explained Variance: 0.9980
- RMSE: $3,630.45
- MAE: $1,602.17
- Mean Absolute Percentage Error: 0.70%

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/property-price-prediction.git
cd property-price-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your data files in the `data/` directory
2. Run the model:
```bash
python enhanced_property_model.py
```

## Data Requirements
The model expects the following data files:
- Historical data (2002-2018)
- Yearly data files (2019-2022)
- Required columns:
  - Sale_price
  - Fin_sqft
  - Lotsize
  - Year_Built
  - Fbath
  - Hbath
  - Bdrms
  - Stories
  - Sale_date
  - PropType
  - Taxkey

## Model Details
The model uses a pipeline approach with:
1. Data cleaning and preprocessing
2. Feature engineering
3. Feature selection
4. XGBoost regression
5. Time series cross-validation

## Visualizations
The model generates several visualizations:
- Actual vs Predicted prices
- Feature importance
- Price trends over time
- Residuals analysis
- Price distribution
- Correlation heatmap
- Feature vs Price scatter plots
- Error analysis

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or suggestions, please open an issue in the repository. 
Limitations and Potential Improvements
Limitations
Data Dependency:
The model relies heavily on the quality and consistency of the input data. Missing values, outliers, or errors can significantly impact performance.
It assumes that the features (e.g., Fin_sqft, Lotsize, Year_Built) are available and correctly formatted in new datasets.
Feature Engineering:
The current feature set is based on domain knowledge, but it may not capture all relevant relationships. For example, location-based features (e.g., proximity to amenities, schools, or public transport) are missing.
Some features (e.g., Property_Age, Total_Bathrooms) are derived from raw data, which may introduce noise if the raw data is inaccurate.
Model Complexity:
The model uses XGBoost, which is powerful but can be prone to overfitting if not tuned carefully. The current hyperparameters are set to reasonable defaults, but they may not be optimal for all datasets.
The model does not explicitly account for non-linear relationships or interactions between features, which may limit its performance on complex datasets.
Time Series Considerations:
While the model uses time-based features (e.g., Sale_Year, Sale_Month), it does not fully leverage time series techniques (e.g., ARIMA, LSTM) that could better capture temporal trends and seasonality.
Interpretability:
XGBoost is a "black-box" model, making it difficult to interpret how individual features influence predictions. This can be a limitation in scenarios where explainability is crucial.
Generalization:
The model may not generalize well to entirely new markets or regions where property dynamics differ significantly from the training data.
Potential Improvements
Enhanced Feature Engineering:
Location-Based Features: Incorporate geographic data (e.g., distance to schools, parks, public transport) to capture location-based influences on property prices.
Market Indicators: Include broader market indicators (e.g., interest rates, economic growth) to account for macroeconomic factors.
Interaction Terms: Create interaction features (e.g., Fin_sqft * Year_Built) to capture non-linear relationships.
Advanced Data Preprocessing:
Imputation Techniques: Use more sophisticated imputation methods (e.g., KNN imputation) to handle missing values.
Outlier Detection: Implement advanced outlier detection techniques (e.g., Isolation Forest) to better identify and handle anomalies.
Model Enhancements:
Hyperparameter Tuning: Use techniques like Bayesian optimization or grid search to find optimal hyperparameters for XGBoost.
Ensemble Methods: Combine multiple models (e.g., XGBoost, Random Forest, LightGBM) to improve robustness and accuracy.
Time Series Models: Integrate time series models (e.g., ARIMA, LSTM) to better capture temporal trends and seasonality.
Interpretability Improvements:
SHAP Values: Use SHAP (SHapley Additive exPlanations) to provide detailed feature importance and interpretability.
LIME: Implement LIME (Local Interpretable Model-agnostic Explanations) to explain individual predictions.
Cross-Validation and Validation:
Stratified Cross-Validation: Use stratified cross-validation to ensure balanced representation of different property types or regions.
External Validation: Validate the model on entirely new datasets to assess its generalization capabilities.
Deployment and Monitoring:
Model Monitoring: Implement continuous monitoring to detect drift or degradation in model performance over time.
Automated Retraining: Set up automated retraining pipelines to update the model with new data periodically.
