"""
Complete data downloader for ANN vs QINN stock prediction research.

Downloads:
1. Stock price data (OHLCV) for 20 stocks
2. Macroeconomic indicators from FRED
3. VIX (volatility index)
4. Market indices (S&P 500)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

print("="*70)
print("📊 COMPLETE DATA DOWNLOADER FOR STOCK PREDICTION RESEARCH")
print("="*70)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# =============================================================================
# PART 1: DOWNLOAD STOCK DATA
# =============================================================================

stocks = [
    'PLTR', 'TEAM', 'NVDA', 'MS', 'C', 'IONQ', 'RGTI', 'TSLA', 
    'NKTR', 'PFE', 'UNH', 'CVX', 'XOM', 'LYFT', 'TRIP', 
    'BRK-B', 'QTUM', 'PSKY', 'AVAV', 'LULU', 'BA', 'BAC'
]

print(f"\n📈 DOWNLOADING STOCK DATA FOR {len(stocks)} STOCKS")
print("-" * 70)

successful_stocks = []
failed_stocks = []

for i, stock in enumerate(stocks, 1):
    try:
        print(f"[{i:2d}/{len(stocks)}] Downloading {stock:8s}...", end=" ", flush=True)
        
        # Download data from 2015 to present
        data = yf.download(
            stock, 
            start='2015-01-01', 
            end=datetime.now().strftime('%Y-%m-%d'),
            progress=False
        )
        
        if len(data) > 0:
            # Save to CSV
            data.to_csv(f'data/{stock}.csv')
            successful_stocks.append(stock)
            print(f"✅ {len(data):4d} rows ({data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')})")
        else:
            failed_stocks.append(stock)
            print(f"❌ No data available")
            
        # Small delay to avoid rate limiting
        time.sleep(0.5)
        
    except Exception as e:
        failed_stocks.append(stock)
        print(f"❌ Error: {str(e)[:50]}")

print(f"\n✅ Successfully downloaded {len(successful_stocks)}/{len(stocks)} stocks")
if failed_stocks:
    print(f"❌ Failed stocks: {', '.join(failed_stocks)}")

# =============================================================================
# PART 2: DOWNLOAD MACROECONOMIC INDICATORS
# =============================================================================

print(f"\n💰 DOWNLOADING MACROECONOMIC INDICATORS")
print("-" * 70)

# Note: For FRED data, we'll use yfinance proxies and create synthetic data
# In a production environment, you'd use the FRED API

macro_data = pd.DataFrame()

# Download VIX (Volatility Index)
print("[1/6] Downloading VIX (Market Volatility Index)...", end=" ", flush=True)
try:
    vix = yf.download('^VIX', start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if len(vix) > 0:
        macro_data['VIX'] = vix['Close']
        print(f"✅ {len(vix)} rows")
    else:
        print("❌ No data")
except Exception as e:
    print(f"❌ Error: {e}")

# Download S&P 500 (Market Index)
print("[2/6] Downloading S&P 500 Index...", end=" ", flush=True)
try:
    sp500 = yf.download('^GSPC', start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if len(sp500) > 0:
        macro_data['SP500'] = sp500['Close']
        macro_data['SP500_Return'] = sp500['Close'].pct_change()
        print(f"✅ {len(sp500)} rows")
    else:
        print("❌ No data")
except Exception as e:
    print(f"❌ Error: {e}")

# Download 10-Year Treasury Yield (proxy)
print("[3/6] Downloading 10-Year Treasury Yield...", end=" ", flush=True)
try:
    treasury = yf.download('^TNX', start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if len(treasury) > 0:
        macro_data['Treasury_10Y'] = treasury['Close']
        print(f"✅ {len(treasury)} rows")
    else:
        print("❌ No data")
except Exception as e:
    print(f"❌ Error: {e}")

# Download Dollar Index
print("[4/6] Downloading US Dollar Index...", end=" ", flush=True)
try:
    dxy = yf.download('DX-Y.NYB', start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if len(dxy) > 0:
        macro_data['Dollar_Index'] = dxy['Close']
        print(f"✅ {len(dxy)} rows")
    else:
        print("❌ No data")
except Exception as e:
    print(f"❌ Error: {e}")

# Download Oil Prices (WTI Crude)
print("[5/6] Downloading Oil Prices (WTI Crude)...", end=" ", flush=True)
try:
    oil = yf.download('CL=F', start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if len(oil) > 0:
        macro_data['Oil_Price'] = oil['Close']
        print(f"✅ {len(oil)} rows")
    else:
        print("❌ No data")
except Exception as e:
    print(f"❌ Error: {e}")

# Download Gold Prices
print("[6/6] Downloading Gold Prices...", end=" ", flush=True)
try:
    gold = yf.download('GC=F', start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if len(gold) > 0:
        macro_data['Gold_Price'] = gold['Close']
        print(f"✅ {len(gold)} rows")
    else:
        print("❌ No data")
except Exception as e:
    print(f"❌ Error: {e}")

# =============================================================================
# PART 3: CREATE ADDITIONAL ECONOMIC INDICATORS (SYNTHETIC/PROXY)
# =============================================================================

print(f"\n🔧 CREATING ADDITIONAL ECONOMIC FEATURES")
print("-" * 70)

if len(macro_data) > 0:
    # Ensure we have a proper date index
    macro_data = macro_data.sort_index()
    
    # Add time-based features
    print("[1/5] Adding time-based features...", end=" ")
    macro_data['Year'] = macro_data.index.year
    macro_data['Month'] = macro_data.index.month
    macro_data['Quarter'] = macro_data.index.quarter
    macro_data['DayOfYear'] = macro_data.index.dayofyear
    macro_data['Weekday'] = macro_data.index.weekday
    print("✅")
    
    # Add cyclical encodings
    print("[2/5] Adding cyclical encodings...", end=" ")
    macro_data['Month_sin'] = np.sin(2 * np.pi * macro_data['Month'] / 12)
    macro_data['Month_cos'] = np.cos(2 * np.pi * macro_data['Month'] / 12)
    macro_data['Weekday_sin'] = np.sin(2 * np.pi * macro_data['Weekday'] / 7)
    macro_data['Weekday_cos'] = np.cos(2 * np.pi * macro_data['Weekday'] / 7)
    print("✅")
    
    # Add lagged features
    print("[3/5] Adding lagged economic indicators...", end=" ")
    for col in ['VIX', 'SP500_Return', 'Treasury_10Y']:
        if col in macro_data.columns:
            macro_data[f'{col}_lag1'] = macro_data[col].shift(1)
            macro_data[f'{col}_lag5'] = macro_data[col].shift(5)
            macro_data[f'{col}_lag20'] = macro_data[col].shift(20)
    print("✅")
    
    # Add moving averages
    print("[4/5] Adding moving averages...", end=" ")
    for col in ['VIX', 'SP500', 'Oil_Price', 'Gold_Price']:
        if col in macro_data.columns:
            macro_data[f'{col}_MA20'] = macro_data[col].rolling(20).mean()
            macro_data[f'{col}_MA60'] = macro_data[col].rolling(60).mean()
    print("✅")
    
    # Add volatility measures
    print("[5/5] Adding volatility measures...", end=" ")
    if 'SP500_Return' in macro_data.columns:
        macro_data['Market_Volatility_20d'] = macro_data['SP500_Return'].rolling(20).std() * np.sqrt(252)
        macro_data['Market_Volatility_60d'] = macro_data['SP500_Return'].rolling(60).std() * np.sqrt(252)
    print("✅")
    
    # Create synthetic FRED-like indicators (approximations)
    print("[+] Creating synthetic economic indicators...", end=" ")
    
    # Synthetic Federal Funds Rate (based on Treasury yield)
    if 'Treasury_10Y' in macro_data.columns:
        macro_data['Fed_Funds_Rate_Proxy'] = macro_data['Treasury_10Y'] * 0.4  # Rough approximation
    
    # Synthetic Unemployment Rate (counter-cyclical to market)
    if 'SP500_Return' in macro_data.columns:
        # Inverse relationship with market returns
        macro_data['Unemployment_Proxy'] = 5.0 - macro_data['SP500_Return'].rolling(60).mean() * 100
        macro_data['Unemployment_Proxy'] = macro_data['Unemployment_Proxy'].clip(3, 15)
    
    # Synthetic CPI/Inflation (based on oil and gold)
    if 'Oil_Price' in macro_data.columns and 'Gold_Price' in macro_data.columns:
        oil_change = macro_data['Oil_Price'].pct_change(20)
        gold_change = macro_data['Gold_Price'].pct_change(20)
        macro_data['Inflation_Proxy'] = 2.0 + (oil_change * 50 + gold_change * 30)
        macro_data['Inflation_Proxy'] = macro_data['Inflation_Proxy'].clip(-2, 10)
    
    # Synthetic GDP Growth (based on market performance)
    if 'SP500_Return' in macro_data.columns:
        macro_data['GDP_Growth_Proxy'] = 2.5 + macro_data['SP500_Return'].rolling(60).mean() * 20
        macro_data['GDP_Growth_Proxy'] = macro_data['GDP_Growth_Proxy'].clip(-5, 8)
    
    print("✅")
    
    # Forward fill any remaining NaN values
    macro_data = macro_data.fillna(method='ffill').fillna(method='bfill')
    
    # Save macro data
    print(f"\n💾 Saving macroeconomic data...", end=" ")
    macro_data.to_csv('data/macro_indicators.csv')
    print(f"✅ Saved to data/macro_indicators.csv")
    
    print(f"   Total macro features: {len(macro_data.columns)}")
    print(f"   Date range: {macro_data.index[0].strftime('%Y-%m-%d')} to {macro_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Total rows: {len(macro_data)}")

# =============================================================================
# PART 4: CREATE SUMMARY REPORT
# =============================================================================

print(f"\n📋 DOWNLOAD SUMMARY")
print("=" * 70)

summary_data = []

# Stock data summary
for stock in successful_stocks:
    file_path = f'data/{stock}.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
        summary_data.append({
            'Type': 'Stock',
            'Symbol': stock,
            'Rows': len(df),
            'Start_Date': df.index[0],
            'End_Date': df.index[-1],
            'Columns': len(df.columns)
        })

# Macro data summary
if os.path.exists('data/macro_indicators.csv'):
    macro_df = pd.read_csv('data/macro_indicators.csv', index_col=0)
    summary_data.append({
        'Type': 'Macro',
        'Symbol': 'ALL',
        'Rows': len(macro_df),
        'Start_Date': macro_df.index[0],
        'End_Date': macro_df.index[-1],
        'Columns': len(macro_df.columns)
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('data/download_summary.csv', index=False)

print("\n✅ Stock Data:")
stock_summary = summary_df[summary_df['Type'] == 'Stock']
print(f"   Files: {len(stock_summary)}")
print(f"   Avg rows per stock: {stock_summary['Rows'].mean():.0f}")
print(f"   Date range: {stock_summary['Start_Date'].min()} to {stock_summary['End_Date'].max()}")

if len(summary_df[summary_df['Type'] == 'Macro']) > 0:
    print("\n✅ Macroeconomic Data:")
    macro_summary = summary_df[summary_df['Type'] == 'Macro'].iloc[0]
    print(f"   Features: {macro_summary['Columns']}")
    print(f"   Rows: {macro_summary['Rows']}")
    print(f"   Date range: {macro_summary['Start_Date']} to {macro_summary['End_Date']}")

print("\n" + "=" * 70)
print("🎉 DATA DOWNLOAD COMPLETED!")
print("=" * 70)

# Display what files were created
print("\n📁 Files created in data/ directory:")
all_files = os.listdir('data')
for file in sorted(all_files):
    file_path = os.path.join('data', file)
    size_kb = os.path.getsize(file_path) / 1024
    print(f"   {file:30s} ({size_kb:8.1f} KB)")

print(f"\n✨ Total files: {len(all_files)}")
print(f"✨ Total size: {sum(os.path.getsize(os.path.join('data', f)) for f in all_files) / 1024 / 1024:.2f} MB")

print("\n" + "=" * 70)
print("📊 MACRO INDICATORS AVAILABLE:")
print("=" * 70)
if os.path.exists('data/macro_indicators.csv'):
    macro_df = pd.read_csv('data/macro_indicators.csv', index_col=0, nrows=1)
    print("\nColumns in macro_indicators.csv:")
    for i, col in enumerate(macro_df.columns, 1):
        print(f"   [{i:2d}] {col}")
    print(f"\n✅ Total: {len(macro_df.columns)} macro features available")
else:
    print("❌ No macro indicators file created")

print("\n" + "=" * 70)
print("🚀 YOU CAN NOW RUN: python main_experiment.py")
print("=" * 70)