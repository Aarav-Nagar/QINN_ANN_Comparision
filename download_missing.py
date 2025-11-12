import yfinance as yf

# Download BAC (Bank of America)
bac = yf.download('BAC', start='2015-01-01', end='2025-10-31')
bac.to_csv('data/BAC.csv')
print("✅ Downloaded BAC")

# Download BA (Boeing)
ba = yf.download('BA', start='2015-01-01', end='2025-10-31')
ba.to_csv('data/BA.csv')
print("✅ Downloaded BA")