import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# 1. Get Data
ticker = "^KLSE"
data = yf.download(ticker, start="1995-01-01", end="2026-03-01")
prices = data['Close'].resample('ME').last().ffill() # Monthly resample

# 2. Log transform to handle exponential growth/percentage changes
log_prices = np.log(prices)

# 3. Apply STL
# period=12 because we are looking for annual (monthly) seasonality
stl = STL(log_prices, period=12, robust=True)
res = stl.fit()

# 4. Plot the components
fig = res.plot()
plt.show()

# Extract the seasonal component
seasonal_component = res.seasonal

# Group by month to see the average 'strength' of the season
monthly_seasonality = seasonal_component.groupby(seasonal_component.index.month).mean()

# Convert back from log scale to see approximate % impact
impact_pct = (np.exp(monthly_seasonality) - 1) * 100
# 1. Calculate Monthly Returns from your original 'prices' series
monthly_returns = prices.pct_change()

# 2. Define a 'Win' (True if return > 0)
is_win = monthly_returns > 0

# 3. Group by month to get the Win Rate (%)
# we multiply by 100 to get a percentage
win_rate = is_win.groupby(is_win.index.month).mean() * 100

# 4. Combine with your existing 'impact_pct' (Seasonality Strength)
# 1. Flatten the seasonal strength and win rate to ensure they are 1D
# .squeeze() turns a (12, 1) into a (12,)
strength_1d = impact_pct.squeeze()
win_rate_1d = win_rate.squeeze()

# 2. Combine into DataFrame
seasonality_analysis = pd.DataFrame({
    'Seasonal Strength (%)': strength_1d,
    'Win Rate (%)': win_rate_1d
})

# 3. Format
seasonality_analysis.index.name = 'Month'
print(seasonality_analysis.round(2))


