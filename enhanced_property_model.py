import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os
import shap
from lime import lime_tabular
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def standardize_column_names(df):
    """Standardize column names across different years."""
    column_mapping = {
        'FinishedSqft': 'Fin_sqft',
        'Rooms': 'Nr_of_rms',
        'taxkey': 'Taxkey',
        'nbhd': 'Nbhd',
        'PropertyID': 'PropertyID'  # Keep this for reference
    }
    return df.rename(columns=column_mapping)

def load_and_combine_data():
    """Load and combine data from all years."""
    print("Loading historical data (2002-2018)...")
    df_historical = pd.read_csv('data/2002-2018-property-sales-data.csv')
    
    # Load and combine recent years
    recent_years = range(2019, 2023)
    dfs = [df_historical]
    
    for year in recent_years:
        print(f"Loading {year} data...")
        df_year = pd.read_csv(f'data/{year}-property-sales-data.csv')
        df_year = standardize_column_names(df_year)
        dfs.append(df_year)
    
    # Combine all data
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Convert sale_date to datetime with flexible parsing
    print("Converting sale dates...")
    df_combined['Sale_date'] = pd.to_datetime(df_combined['Sale_date'], format='mixed')
    
    return df_combined

def engineer_features(df):
    """Create enhanced features including time-based ones."""
    df = df.copy()
    current_year = datetime.now().year
    
    # Basic features
    df['Property_Age'] = current_year - df['Year_Built']
    df['Total_Bathrooms'] = df['Fbath'] + df['Hbath']
    
    # Time-based features
    df['Sale_Year'] = df['Sale_date'].dt.year
    df['Sale_Month'] = df['Sale_date'].dt.month
    df['Sale_Quarter'] = df['Sale_date'].dt.quarter
    df['Days_Since_2002'] = (df['Sale_date'] - pd.Timestamp('2002-01-01')).dt.days
    
    # Market trend features
    df['Price_per_sqft'] = df['Sale_price'] / df['Fin_sqft']
    df['Log_Sqft'] = np.log1p(df['Fin_sqft'])
    df['Log_Lotsize'] = np.log1p(df['Lotsize'])
    df['Log_Price'] = np.log1p(df['Sale_price'])
    
    # Property features
    df['Bathrooms_per_Bedroom'] = df['Total_Bathrooms'] / df['Bdrms'].replace(0, 1)
    df['Sqft_per_Bedroom'] = df['Fin_sqft'] / df['Bdrms'].replace(0, 1)
    df['Lot_to_Sqft_Ratio'] = df['Lotsize'] / df['Fin_sqft']
    df['Age_Squared'] = df['Property_Age'] ** 2
    df['Total_Rooms'] = df['Bdrms'] + df['Total_Bathrooms']
    df['Room_Density'] = df['Total_Rooms'] / df['Fin_sqft']
    df['Sqft_per_Story'] = df['Fin_sqft'] / df['Stories'].replace(0, 1)
    df['Bathroom_Ratio'] = df['Total_Bathrooms'] / df['Total_Rooms'].replace(0, 1)
    
    return df

def create_feature_engineering_pipeline():
    """Create a pipeline for feature engineering."""
    return Pipeline([
        ('feature_engineering', FunctionTransformer(engineer_features))
    ])

def create_preprocessing_pipeline():
    """Create a preprocessing pipeline with imputation and scaling."""
    numeric_features = [
        'Fin_sqft', 'Lotsize', 'Property_Age', 'Total_Bathrooms', 'Bdrms', 'Stories',
        'Price_per_sqft', 'Bathrooms_per_Bedroom', 'Sqft_per_Bedroom', 'Lot_to_Sqft_Ratio',
        'Age_Squared', 'Log_Sqft', 'Log_Lotsize', 'Total_Rooms', 'Room_Density',
        'Sqft_per_Story', 'Bathroom_Ratio', 'Days_Since_2002', 'Sale_Year', 'Sale_Month',
        'Sale_Quarter', 'Log_Price'
    ]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    return preprocessor, numeric_features

def create_model_pipeline():
    """Create the complete model pipeline."""
    preprocessor, numeric_features = create_preprocessing_pipeline()
    
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selector', SelectFromModel(model, threshold='median')),
        ('model', model)
    ]), numeric_features

def clean_data(df):
    """Comprehensive data cleaning function."""
    df = df.copy()
    
    # 1. Ensure numeric columns are numeric
    print("Converting columns to numeric...")
    numeric_columns = ['Fin_sqft', 'Lotsize', 'Year_Built', 'Fbath', 'Hbath', 'Bdrms', 'Stories', 'Sale_price']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Replace negative values with NaN
            df[col] = df[col].replace([-np.inf, np.inf], np.nan)
            df[col] = df[col].mask(df[col] < 0, np.nan)
    
    # 2. Remove invalid property types
    print("Removing invalid property types...")
    valid_property_types = ['Single Family', 'Condo', 'Multi-Family', 'Residential']
    if 'PropType' in df.columns:
        df = df[df['PropType'].isin(valid_property_types)]
    
    # 3. Handle outliers
    print("Handling outliers...")
    # Remove properties with unrealistic sizes
    df = df[
        (df['Fin_sqft'] > 100) &  # Minimum reasonable size
        (df['Fin_sqft'] < 10000) &  # Maximum reasonable size
        (df['Lotsize'] > 100) &  # Minimum reasonable lot size
        (df['Lotsize'] < 100000) &  # Maximum reasonable lot size
        (df['Sale_price'] > 10000) &  # Minimum reasonable price
        (df['Sale_price'] < df['Sale_price'].quantile(0.99))  # Remove top 1% prices
    ]
    
    # 4. Handle year built
    print("Cleaning year built data...")
    if 'Year_Built' in df.columns:
        # Remove unrealistic years
        current_year = datetime.now().year
        df = df[
            (df['Year_Built'] > 1800) &  # Reasonable minimum year
            (df['Year_Built'] <= current_year)  # Can't be built in the future
        ]
    
    # 5. Handle bathrooms and bedrooms
    print("Cleaning bathroom and bedroom data...")
    if 'Bdrms' in df.columns:
        df = df[df['Bdrms'] <= 10]  # Maximum reasonable number of bedrooms
    if 'Fbath' in df.columns and 'Hbath' in df.columns:
        df = df[(df['Fbath'] + df['Hbath']) <= 10]  # Maximum reasonable number of bathrooms
    
    # 6. Handle stories
    print("Cleaning stories data...")
    if 'Stories' in df.columns:
        df = df[df['Stories'] <= 5]  # Maximum reasonable number of stories
    
    # 7. Remove duplicate sales
    print("Removing duplicate sales...")
    if 'Taxkey' in df.columns and 'Sale_date' in df.columns:
        df = df.drop_duplicates(subset=['Taxkey', 'Sale_date'], keep='first')
    
    print(f"Data cleaning complete. Remaining rows: {len(df)}")
    return df

def main():
    print("Loading and preprocessing data...")
    # Load and combine all data
    df = load_and_combine_data()
    
    # Apply comprehensive cleaning
    df = clean_data(df)
    
    # Create feature engineering pipeline
    feature_engineering_pipeline = create_feature_engineering_pipeline()
    df = feature_engineering_pipeline.fit_transform(df)
    
    # Create model pipeline
    model_pipeline, numeric_features = create_model_pipeline()
    
    # Prepare data
    X = df[numeric_features]
    y = df['Sale_price']
    
    # Split data chronologically
    train_size = int(len(df) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    print("Training model with time series cross-validation...")
    # Perform time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=tscv, scoring='r2')
    print(f"Time series cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Train final model
    model_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model_pipeline.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'R² Score': r2_score(y_test, y_pred),
        'Explained Variance': explained_variance_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'Mean Absolute Percentage Error': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model components
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model_pipeline, 'models/enhanced_property_model.joblib')
    print("\nModel components saved successfully!")
    
    # Generate visualizations
    generate_visualizations(X_test, y_test, y_pred, model_pipeline, df)

def generate_visualizations(X_test, y_test, y_pred, model_pipeline, df):
    """Generate and save enhanced model performance visualizations."""
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    plt.style.use('default')
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs Predicted Property Prices')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance
    plt.figure(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'Feature': model_pipeline.named_steps['feature_selector'].get_feature_names_out(),
        'Importance': model_pipeline.named_steps['model'].feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Price Trends Over Time
    plt.figure(figsize=(12, 6))
    df['Sale_Year'] = pd.to_datetime(df['Sale_date']).dt.year
    yearly_avg = df.groupby('Sale_Year')['Sale_price'].mean()
    plt.plot(yearly_avg.index, yearly_avg.values, marker='o')
    plt.title('Average Property Prices Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Price ($)')
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig('plots/price_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Residuals Plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig('plots/residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Sale_price'], bins=50, kde=True)
    plt.title('Distribution of Property Prices')
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig('plots/price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Feature vs Price Scatter Plots
    key_features = ['Fin_sqft', 'Lotsize', 'Year_Built', 'Total_Bathrooms', 'Bdrms']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(key_features):
        if idx < len(axes):
            sns.scatterplot(data=df, x=feature, y='Sale_price', alpha=0.5, ax=axes[idx])
            axes[idx].set_title(f'{feature} vs Price')
            axes[idx].set_ylabel('Price ($)')
            axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('plots/feature_vs_price.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Error Analysis
    plt.figure(figsize=(10, 6))
    error_percentage = np.abs((y_test - y_pred) / y_test) * 100
    sns.histplot(error_percentage, bins=50, kde=True)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Absolute Percentage Error (%)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plots/error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Save Model Metrics
    metrics = {
        'R² Score': r2_score(y_test, y_pred),
        'Explained Variance': explained_variance_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'Mean Absolute Percentage Error': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    with open('plots/model_metrics.txt', 'w') as f:
        f.write('Model Performance Metrics:\n')
        f.write('========================\n\n')
        for metric, value in metrics.items():
            f.write(f'{metric}: {value:.4f}\n')

if __name__ == "__main__":
    main() 