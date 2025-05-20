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