# -*- coding: utf-8 -*-
"""
Silver Price Forecasting with Prophet
Extracted and cleaned for clarity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

def main():
    print('='*50)
    print('Phase 1: Data Loading & Preparation')
    print('='*50)
    
    # 1. Load Data
    # Assuming the csv is in the same directory
    try:
        df = pd.read_csv('silver_prices_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Error: 'silver_prices_data.csv' not found.")
        return

    # Drop Volume if it's all NaN
    if 'Volume' in df.columns and df['Volume'].isnull().all():
        df.drop('Volume', axis=1, inplace=True)

    # 2. Prepare data for Prophet
    # Prophet requires columns 'ds' (Date) and 'y' (Value)
    prophet_df = df[['Price']].reset_index()
    prophet_df.columns = ['ds', 'y']
    prophet_df = prophet_df.dropna()

    print(f"Data prepared for Prophet. Rows: {len(prophet_df)}")
    print(prophet_df.head())
    print('\n')

    # 3. Train/Test Split
    train_size = int(len(prophet_df) * 0.8)
    train = prophet_df[:train_size]
    test = prophet_df[train_size:]

    print('='*50)
    print('Phase 2: Training & Evaluation')
    print('='*50)
    print(f'Training samples: {len(train)}')
    print(f'Testing samples: {len(test)}')

    # 4. Train Model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    model.fit(train)
    print('Model trained on training set.')

    # 5. Evaluate on Test Set
    test_forecast = model.predict(test[['ds']])
    
    y_true = test['y'].values
    y_pred = test_forecast['yhat'].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    print('\nEvaluation Metrics (Test Set):')
    print(f'   RMSE: ${rmse:.2f}')
    print(f'   MAE:  ${mae:.2f}')
    print(f'   MAPE: {mape:.2f}%')
    print(f'   R2:   {r2:.4f}')

    # Plot Actual vs Predicted
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test['ds'].values, y_true, label='Actual', color='black', linewidth=2)
    ax.plot(test['ds'].values, y_pred, label='Prophet Predicted', color='blue', alpha=0.7)
    ax.fill_between(test['ds'].values, test_forecast['yhat_lower'].values, test_forecast['yhat_upper'].values, alpha=0.2, color='blue')
    ax.set_title('Prophet: Actual vs Predicted (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    plt.tight_layout()
    plt.show(block=False) # block=False so it doesn't hang if run in some environments, but usually fine.
    
    print('\n' + '='*50)
    print('Phase 3: Future Forecasting (Including Q1 2026)')
    print('='*50)

    # 6. Retrain on Full Data
    model_full = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    model_full.fit(prophet_df)
    print("Model retrained on full dataset.")

    # 7. Make Future Dataframe
    # Forecasting 90 days into the future
    future_dates = model_full.make_future_dataframe(periods=90, freq='D')
    forecast = model_full.predict(future_dates)
    
    # Filter for Q1 2026
    start_date = pd.to_datetime('2026-01-01')
    end_date = pd.to_datetime('2026-03-31')
    forecast_q1_2026 = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]

    print(f"\nForecast generated for Q1 2026 ({len(forecast_q1_2026)} days).")
    
    # Show Q1 2026 Summary
    forecast_export = forecast_q1_2026[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_export.columns = ['Date', 'Predicted_Price', 'Lower_Bound', 'Upper_Bound']
    print(forecast_export.head())

    # 8. Plot Forecast
    fig2 = model_full.plot(forecast, figsize=(14, 6))
    plt.title('Silver Price Forecast (Prophet Full Model)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    # Highlight Q1 2026
    plt.axvspan(start_date, end_date, color='green', alpha=0.1, label='Q1 2026 Forecast')
    plt.legend()
    plt.xlim(pd.to_datetime('2025-06-01'), pd.to_datetime('2026-04-01'))
    plt.tight_layout()
    plt.show()

    # Plot Components
    fig3 = model_full.plot_components(forecast)
    plt.suptitle('Prophet Forecast Components', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # 9. Save Forecast
    output_filename = 'silver_price_forecast_2026_prophet.csv'
    forecast_export.to_csv(output_filename, index=False)
    print(f'\nForecast saved to {output_filename}')

if __name__ == "__main__":
    main()
