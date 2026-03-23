# Replace with your actual SerpApi Key
API_KEY = "cc678bb184df49f0c9589462f79c74146953024cdf03b28a3323b21638a78096" 
import serpapi
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import re
import numpy as np
from datetime import datetime
from scipy.stats import pearsonr # Added for p-value calculation

# The "Diversified Malaysia" Basket
assets = [
    "^KLSE",      # Top 30 Blue Chips
    "0800EA.KL"   # Bond Index (Your original)
]
results = []
def analyze_lead_lag(combined_df, ticker_name):
    print(f"\n--- 🕵️ ANALYZING LEAD-LAG (NORMALIZED TO T-0) FOR {ticker_name} ---")
    
    analysis_df = combined_df.copy()
    # Using raw values as requested
    analysis_df['S_Change'] = analysis_df['Sentiment'].diff()
    analysis_df['P_Return'] = analysis_df['Price'].pct_change()
    analysis_df = analysis_df.dropna()

    lags = range(-100, 101) # Updated to 101 to include exactly +100
    lag_results = []

    for lag in lags:
        shifted_p = analysis_df['P_Return'].shift(-lag)
        temp = pd.concat([analysis_df['S_Change'], shifted_p], axis=1).dropna()
        temp.columns = ['S', 'P']
        
        if len(temp) > 30:
            corr, p = pearsonr(temp['S'], temp['P'])
            lag_results.append({'lag': lag, 'corr': corr, 'p': p})

    lag_df = pd.DataFrame(lag_results)
    if lag_df.empty:
        print("❌ Not enough data for lead-lag analysis.")
        return

    # --- NEW: T-0 NORMALIZATION LOGIC ---
    # 1. Isolate the exact correlation at Lag 0
    t0_corr = lag_df.loc[lag_df['lag'] == 0, 'corr'].values[0]
    
    # 2. Subtract T-0 correlation from all lags (Baseline becomes 0)
    lag_df['norm_corr'] = lag_df['corr']
    
    # 3. Find the Golden Lag based on the highest absolute EXCESS correlation
    # We ignore lag 0 itself since it is now mathematically 0
    search_df = lag_df[lag_df['lag'] != 0]
    golden_row = search_df.loc[search_df['norm_corr'].abs().idxmax()]
    # ------------------------------------
    
    # Visualization
    plt.figure(figsize=(14, 6))
    # Color based on whether the lag is BETTER (blue) or WORSE (red) than T-0
    colors = ['#1E88E5' if c >= 0 else '#d32f2f' for c in lag_df['norm_corr']]
    
    plt.bar(lag_df['lag'], lag_df['norm_corr'], color=colors, alpha=0.8)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.8)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5) # T-0 Baseline
    
    plt.title(f"Relative Cross-Correlation: Sentiment vs {ticker_name}\nPeak Excess Lag: {int(golden_row['lag'])} days (Excess Corr vs T-0: {golden_row['norm_corr']:.4f})")
    plt.xlabel("Lag Days (Positive = Sentiment Leads Price)")
    plt.ylabel("Excess Correlation (Raw - T0 Baseline)")
    plt.grid(True, alpha=0.2)
    plt.show()

    print(f"🌟 BASELINE T-0 CORR: {t0_corr:.4f}")
    print(f"🌟 PEAK EXCESS LAG: {int(golden_row['lag'])} days")
    print(f"📊 EXCESS CORR: {golden_row['norm_corr']:.4f} (Raw: {golden_row['corr']:.4f})")
    print(f"🎯 P-VALUE (Raw): {golden_row['p']:.2e}")
    
    if golden_row['norm_corr'] > 0:
        if golden_row['lag'] > 0:
            print(f"💡 VERDICT: Sentiment leads price by {int(golden_row['lag'])} days with a STRONGER correlation than T-0.")
        else:
            print(f"💡 VERDICT: Price leads sentiment by {abs(int(golden_row['lag']))} days with a STRONGER correlation than T-0.")
    else:
        print("💡 VERDICT: No lag provides a better positive correlation than looking at T-0 exactly.")

def run_historical_overlay_agent(start_date, ticker="7052.KL", threshold=75):
    today_str = datetime.now().strftime('%Y-%m-%d')
    date_range = f"{start_date} {today_str}"
    
    print(f"--- 📊 FETCHING DATA: {ticker} vs EPF SENTIMENT ---")
    
    client = serpapi.Client(api_key=API_KEY)
    # kws = ["Tarikh Gaji", "Bonus 2026", "EPF Dividend"]
    kws = ["pengeluaran KWSP", "kedai pajak gadai", "EIS PERKESO", "EPF Withdraw", "Unemployment"]
    params = {
        "engine": "google_trends",
        "q": ",".join(kws),
        "geo": "MY",
        "date": date_range,
        "data_type": "TIMESERIES"
    }
    
    try:
        results = client.search(params)
        timeline_data = results.get("interest_over_time", {}).get("timeline_data", [])
        
        if not timeline_data:
            print("No trends data found.")
            return

        # 1. Parse Trends Data
        history = []
        for entry in timeline_data:
            avg_val = sum([v['extracted_value'] for v in entry['values']]) / len(entry['values'])
            clean_date = re.sub(r'[–-].*?,', ',', entry['date'])
            if "–" in clean_date or "-" in clean_date:
                clean_date = clean_date.split('–')[0].split('-')[0].strip()
            history.append({'Date': pd.to_datetime(clean_date), 'Sentiment': avg_val})
            
        trends_df = pd.DataFrame(history).set_index('Date')

        # 2. Fetch Stock Data
        stock_data = yf.download(ticker, start=start_date, progress=False)['Close']
        stock_data = pd.DataFrame(stock_data)

        # 3. ALIGNMENT FIX: Resample and Interpolate
        trends_daily = trends_df.resample('D').interpolate(method='linear')
        combined = trends_daily.join(stock_data, how='inner').dropna()
        
        combined.columns = ['Sentiment', 'Price']
        combined['Sentiment'] = np.log1p(combined['Sentiment'])
        
        # --- NEW: STATISTICAL CALCULATION ---
        # Returns (Correlation Coefficient, P-Value)
        corr_val, p_value = pearsonr(combined['Sentiment'], combined['Price'])
        significance = "STATISTICALLY SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
        # ------------------------------------
        # 4. Normalize (0 to 1 scale)
        norm_df = (combined - combined.min()) / (combined.max() - combined.min())

        # 5. Plotting
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.plot(norm_df.index, norm_df['Sentiment'], color='#1E88E5', lw=1.5, label='EPF Sentiment (Interpolated)')
        ax1.fill_between(norm_df.index, norm_df['Sentiment'], alpha=0.1, color='#1E88E5')
        ax1.set_ylabel('Search Intensity')
        
        ax2 = ax1.twinx()
        ax2.plot(norm_df.index, norm_df['Price'], color='#FFC107', lw=2.5, label=f'{ticker} Price')
        ax2.set_ylabel(f'Stock Price ({ticker})')

        # Update title with new stats
        plt.title(f"Liquidity Correlation: {ticker} vs. EPF Interest\nCorr: {corr_val:.4f} | P-Value: {p_value:.4e} ({significance})")
        
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        plt.grid(True, alpha=0.2)
        plt.show()

        print(f"✅ Correlation: {corr_val:.4f}")
        print(f"✅ P-Value: {p_value:.4e}")
        print(f"✅ Conclusion: {significance}")
        analyze_lead_lag(combined, ticker)

    except Exception as e:
        print(f"API Error: {e}")

# Run analysis
for a in assets:
    run_historical_overlay_agent(start_date="2021-05-01", ticker=a)

