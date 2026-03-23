import ee
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# 1. INITIALIZE & CONFIGURATION
ee.Initialize(project='numeric-marker-490315-v8')

# The "Big Three" Malaysian Export Hubs
MY_PORTS = {
    "Port Klang": {"lon": 101.326, "lat": 2.969},
    "Tanjung Pelepas (PTP)": {"lon": 103.545, "lat": 1.363},
    "Penang Port": {"lon": 100.370, "lat": 5.394}
}

def get_aggregated_data(ticker="USDMYR=X", start='2015-01-01', end='2026-03-15'):
    print(f"--- FETCHING AGGREGATED NATIONAL PORT DATA ---")
    
    # 1. Fetch Currency Data
    stock = yf.download(ticker, start=start, end=end)
    if isinstance(stock.columns, pd.MultiIndex): stock.columns = stock.columns.get_level_values(0)
    stock = stock[['Close']].copy()
    
    # 2. Loop through ports and fetch SAR data
    radar_dfs = []
    
    for port_name, coords in MY_PORTS.items():
        print(f"Extracting SAR data for {port_name}...")
        aoi = ee.Geometry.Point([coords['lon'], coords['lat']]).buffer(5000).bounds()
        col = (ee.ImageCollection('COPERNICUS/S1_GRD')
               .filterBounds(aoi).filterDate(start, end).sort('system:time_start'))

        def extract_radar_stats(img):
            date = img.date().format('YYYY-MM-dd')
            main_band = ee.String(ee.Algorithms.If(img.bandNames().contains('VV'), 'VV', img.bandNames().get(0)))
            stats = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=100)
            return ee.Feature(None, {'Date': date, f'{port_name}_dB': stats.get(main_band)})

        raw_results = col.map(extract_radar_stats).getInfo()
        
        # Clean and format the data for this specific port
        points = [{'Date': pd.to_datetime(f['properties']['Date']), 
                   port_name: f['properties'][f'{port_name}_dB']} 
                  for f in raw_results['features'] if f['properties'][f'{port_name}_dB'] is not None]
        
        df_port = pd.DataFrame(points).drop_duplicates('Date').set_index('Date')
        radar_dfs.append(df_port)
    
    # 3. Combine all port data
    print("\nMerging and calculating National Aggregate...")
    combined_radar = pd.concat(radar_dfs, axis=1).sort_index()
    
    # Interpolate missing days (since satellites pass over different ports on different days)
    combined_radar = combined_radar.interpolate(method='time').dropna()
    
    # CALCULATE THE AGGREGATE SIGNAL (Mean of all 3 ports)
    combined_radar['Aggregate_Radar_dB'] = combined_radar.mean(axis=1)
    
    # Merge with Currency
    merged = pd.merge_asof(stock.sort_index(), combined_radar, left_index=True, right_index=True, direction='backward')
    return merged.dropna()

# 1. INITIALIZE & CONFIGURATION
ee.Initialize(project='numeric-marker-490315-v8')

MY_PORTS = {
    "Port Klang": {"lon": 101.326, "lat": 2.969},
    "Tanjung Pelepas (PTP)": {"lon": 103.545, "lat": 1.363},
    "Penang Port": {"lon": 100.370, "lat": 5.394}
}

# (get_aggregated_data function remains the same as your previous version)

try:
    df = get_aggregated_data()

    # --- 1. SIGNAL & VOLATILITY CALCULATIONS ---
    WINDOW = 90
    df['Radar_Mean'] = df['Aggregate_Radar_dB'].rolling(WINDOW).mean()
    df['Radar_Std'] = df['Aggregate_Radar_dB'].rolling(WINDOW).std()
    df['Radar_Z'] = (df['Aggregate_Radar_dB'] - df['Radar_Mean']) / df['Radar_Std']
    
    # Calculate Adaptive Volatility for Peak Prominence
    # We use a 30-day window for more 'local' sensitivity to currency spikes
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Rolling_Vol'] = df['Log_Ret'].rolling(window=30).std()
    
    df = df.dropna()

    # Detect National Dips
    radar_valleys, _ = find_peaks(-df['Radar_Z'], distance=60, height=1.00) 
    radar_dip_dates = df.index[radar_valleys]

    print(f"\nDetected {len(radar_dip_dates)} Systemic National Dips.")

    # --- 2. ADAPTIVE LEAD TIME ANALYSIS ---
    lead_times = []
    
    for r_date in radar_dip_dates:
        # Get local volatility at the time of the dip to set dynamic prominence
        # Sensitivity factor of 0.5 - 1.0 is standard for 'Prominence'
        local_vol = df.loc[r_date, 'Rolling_Vol']
        dynamic_prominence = local_vol * 20.0
        
        # Search window for reaction: 90 days after dip
        search_end = r_date + pd.Timedelta(days=90)
        mask = (df.index > r_date) & (df.index <= search_end)
        search_period = df.loc[mask]
        
        if not search_period.empty:
            # Detect peaks in this specific window using dynamic prominence
            peaks, _ = find_peaks(search_period['Close'], prominence=dynamic_prominence)
            
            if len(peaks) > 0:
                first_peak_date = search_period.index[peaks[0]]
                lead_times.append((first_peak_date - r_date).days)
            else:
                lead_times.append(np.nan)
        else:
            lead_times.append(np.nan)

    avg_lead = np.nanmean(lead_times)
    med_lead = np.nanmedian(lead_times)

    print(f"--- VOLATILITY-ADJUSTED LEAD ANALYSIS ---")
    print(f"Average Lead: {avg_lead:.1f} days | Median Lead: {med_lead:.1f} days")

    # --- 3. BACKTEST (Using Detected Median Lead) ---
    optimal_hold = int(med_lead) if not np.isnan(med_lead) else 45
    
    performance_stats = []
    initial_capital = 10000
    equity_curve = [initial_capital]
    curve_dates = [df.index[0]]

    for r_date in radar_dip_dates:
        price_at_dip = df.loc[r_date, 'Close']
        future_date = r_date + pd.Timedelta(days=optimal_hold)
        
        closest_idx = df.index.get_indexer([future_date], method='nearest')[0]
        actual_exit_date = df.index[closest_idx]
        
        trade_return = (df.iloc[closest_idx]['Close'] - price_at_dip) / price_at_dip
        performance_stats.append(trade_return)
        
        new_balance = equity_curve[-1] * (1 + trade_return)
        equity_curve.append(new_balance)
        curve_dates.append(actual_exit_date)

    # 4. INSTITUTIONAL METRICS OUTPUT
    if performance_stats:
        perf_array = np.array(performance_stats)
        eq_series = pd.Series(equity_curve)
        max_dd = ((eq_series - eq_series.cummax()) / eq_series.cummax()).min() * 100
        
        print(f"\n--- MACRO AGGREGATE RESULTS ({optimal_hold}D HOLD) ---")
        print(f"Win Rate: {(perf_array > 0).mean()*100:.1f}%")
        print(f"Sharpe Ratio: {np.mean(perf_array)/np.std(perf_array):.4f}")
        print(f"Sortino Ratio: {np.mean(perf_array)/np.std(perf_array[perf_array<0]):.4f}")
        print(f"Max Drawdown: {max_dd:.2f}%")
        print(f"Final Value: ${equity_curve[-1]:,.2f}")

        # Plotting remains similar to your previous block
        plt.figure(figsize=(12, 6))
        plt.plot(curve_dates, equity_curve, marker='o', color='#2ca02c', label="Adaptive National Aggregate")
        plt.title(f"Volatility-Adjusted Alpha: {optimal_hold}-Day Target Window")
        plt.grid(True, alpha=0.3)
        plt.show()

except Exception as e:
    print(f"Error: {e}")

if performance_stats:
    perf_array = np.array(performance_stats)
    
    # Calculate the 'Magnitude' of winning vs losing trades
    wins = perf_array[perf_array > 0]
    losses = perf_array[perf_array < 0]
    
    avg_win = np.mean(wins) * 100 if len(wins) > 0 else 0
    avg_loss = np.mean(losses) * 100 if len(losses) > 0 else 0
    
    # The 'Expectancy Ratio'
    # (Win% * Avg Win) + (Loss% * Avg Loss)
    win_rate = len(wins) / len(perf_array)
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    print(f"\n--- MAGNITUDE ANALYSIS ---")
    print(f"Average Win: +{avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Trade Expectancy: {expectancy:.2f}% per signal")

if performance_stats:
    # 1. Create a detailed trade ledger
    trade_ledger = []
    for i, r_date in enumerate(radar_dip_dates):
        # Re-calculating for the ledger
        price_at_dip = df.loc[r_date, 'Close']
        future_date = r_date + pd.Timedelta(days=optimal_hold)
        closest_idx = df.index.get_indexer([future_date], method='nearest')[0]
        exit_price = df.iloc[closest_idx]['Close']
        ret = (exit_price - price_at_dip) / price_at_dip
        
        trade_ledger.append({
            'Entry Date': r_date.strftime('%Y-%m-%d'),
            'Exit Date': df.index[closest_idx].strftime('%Y-%m-%d'),
            'Return (%)': round(ret * 100, 2),
            'Radar Z-Score': round(df.loc[r_date, 'Radar_Z'], 2)
        })

    # 2. Sort by Return and get Top 5
    ledger_df = pd.DataFrame(trade_ledger)
    top_5_trades = ledger_df.sort_values(by='Return (%)', ascending=False).head(10)

    print("\n--- TOP 5 MACRO ALPHA EVENTS ---")
    print(top_5_trades.to_string(index=False))

# Sort by Return ascending to get the Worst 5
    worst_5_trades = ledger_df.sort_values(by='Return (%)', ascending=True).head(10)

    print("\n--- BOTTOM 5 MACRO FAILURE EVENTS ---")
    print(worst_5_trades.to_string(index=False))


# --- 5. PRICE ACTION & TRADE EXECUTION PLOT ---
plt.figure(figsize=(16, 8))

# Plot the USD/MYR Price
plt.plot(df.index, df['Close'], color='black', alpha=0.3, label='USDMYR=X Price')

# Plot Buy (Entry) and Sell (Exit) markers
entry_prices = []
exit_prices = []
exit_dates = []

for r_date in radar_dip_dates:
    # Entry Point (The Signal)
    price_at_dip = df.loc[r_date, 'Close']
    entry_prices.append(price_at_dip)
    
    # Exit Point (The 46-Day Target)
    future_date = r_date + pd.Timedelta(days=optimal_hold)
    closest_idx = df.index.get_indexer([future_date], method='nearest')[0]
    actual_exit_date = df.index[closest_idx]
    price_at_exit = df.iloc[closest_idx]['Close']
    
    exit_dates.append(actual_exit_date)
    exit_prices.append(price_at_exit)
    
    # Draw a line connecting Entry to Exit for visual clarity
    plt.plot([r_date, actual_exit_date], [price_at_dip, price_at_exit], 
             color='green' if price_at_exit > price_at_dip else 'red', 
             linewidth=2, alpha=0.6)

# Scatter plot the specific points
plt.scatter(radar_dip_dates, entry_prices, color='green', marker='^', s=100, label='Entry (Radar Dip Detected)', zorder=5)
plt.scatter(exit_dates, exit_prices, color='red', marker='v', s=100, label='Exit (46-Day Target)', zorder=5)

plt.title(f"Agentic Macro Fund: USD/MYR Signal Execution ({optimal_hold}-Day Structural Hold)")
plt.ylabel("Exchange Rate (MYR per 1 USD)")
plt.xlabel("Year")
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

