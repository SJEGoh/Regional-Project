import torch
import pandas as pd
import re
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.dates as mdates
from scipy import stats
import numpy as np

# --- 1. CONFIGURATION ---
model_name = "ProsusAI/finbert"
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

def translate_text(text):
    if not text: return ""
    try:
        translator = GoogleTranslator(source='auto', target='en')
        chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
        return " ".join([translator.translate(c) for c in chunks])
    except Exception as e:
        print(f"   ⚠️ Translation failed: {e}")
        return text
    
def analyze_finbert_sliding_window(text):
    if not text or len(text) < 50: return 0.0
    segments = [text[i:i + 1500] for i in range(0, len(text), 1500)]
    all_logits = []
    for segment in segments:
        inputs = tokenizer(segment, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output = model(**inputs)
            all_logits.append(output.logits)
    if not all_logits: return 0.0
    avg_logits = torch.mean(torch.cat(all_logits), dim=0)
    probs = torch.nn.functional.softmax(avg_logits, dim=-1)
    return probs[0].item() - probs[1].item()

# --- 2. ASYNC SCRAPER ---
async def scrape_and_analyze_async(total_limit=200):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        page = await context.new_page()
        
        print("🚀 Accessing BNM...")
        await page.goto("https://www.bnm.gov.my/speeches-interviews", wait_until="networkidle")
        
        results = []
        page_num = 1
        
        while len(results) < total_limit:
            print(f"--- Scraping Page {page_num} ---")
            
            # 1. Parse current page content
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            rows = soup.find('tbody', id='myTable').find_all('tr')
            
            # 2. Process all rows on this page
            for row in rows:
                if len(results) >= total_limit: break
                
                cells = row.find_all('td')
                if not cells: continue
                
                date, title = cells[0].get_text(strip=True), cells[1].get_text(strip=True)
                href = cells[1].find('a')['href']
                link = href if href.startswith('http') else f"https://www.bnm.gov.my{href}"
                
                print(f"📄 [{len(results)+1}] Processing: {title[:50]}...")
                try:
                    # Use a new page or re-use existing one for details
                    detail_page = await context.new_page()
                    await detail_page.goto(link, wait_until="domcontentloaded")
                    detail_html = await detail_page.content()
                    s_soup = BeautifulSoup(detail_html, 'html.parser')
                    
                    content_div = s_soup.find('div', class_='article-content-cs')
                    text = " ".join([p.get_text() for p in content_div.find_all('p')]) if content_div else ""
                    
                    if "Ucapan" in title or any(w in text[:200] for w in ["ekonomi", "saya", "kita"]):
                        text = translate_text(text)
                    
                    sentiment = analyze_finbert_sliding_window(text)
                    results.append({
                        'Date': date, 'Sentiment': round(sentiment, 4), 'Title': title,
                        'Speaker': re.search(r"(?:by|by the)\s+([^,at|]+)", title).group(1).strip() if "by" in title.lower() else "BNM Representative"
                    })
                    await detail_page.close()
                except Exception as e:
                    print(f"   ❌ Failed: {e}")

            # 3. Handle Pagination
            if len(results) < total_limit:
                # Find the 'Next' button specifically using Playwright's locator
                next_button = page.locator("li.next:not(.disabled) a")
                
                if await next_button.count() > 0:
                    print("⏭️ Moving to next page...")
                    await next_button.click()
                    # Wait for table to reload
                    await page.wait_for_load_state("networkidle")
                    await asyncio.sleep(2) # Buffer for JS rendering
                    page_num += 1
                else:
                    print("🏁 No more pages available.")
                    break
        
        await browser.close()
        return pd.DataFrame(results)
# --- 3. HELPER FUNCTIONS ---
async def helper():
    df_final = await scrape_and_analyze_async(total_limit=780)
    df_sent = pd.DataFrame(df_final)
    df_sent['Date'] = pd.to_datetime(df_sent['Date'])

    start_date = df_sent['Date'].min() - pd.Timedelta(days=10)
    end_date = df_sent['Date'].max() + pd.Timedelta(days=2)

    fx_data = yf.download("MYR=X", start=start_date, end=end_date)
    fx_data.columns = fx_data.columns.get_level_values(0)
    fx_data = fx_data.reset_index()

    bond_data = yf.download("0800EA.KL", start=start_date, end=end_date)
    bond_data.columns = bond_data.columns.get_level_values(0)
    bond_data = bond_data.reset_index()

    def process_sentiment_df(df):
        weights = {'Prime Minis': 2.0, 'Governor': 1.5, 'Depu': 1.2, 'Assis': 1.0, 'BNM Representative': 0.7, 'T': 0.8}
        def get_weight(speaker):
            for key, val in weights.items():
                if key in speaker: return val
            return 0.7
        macro_keywords = ['opr', 'inflation', 'monetary', 'ringgit', 'exchange', 'economy', 'growth', 'outlook', 'stability', 'financial', 'sector', 'market', 'development', 'policy']
        df['Weight'] = df['Speaker'].apply(get_weight)
        df['Is_Macro'] = df['Title'].str.lower().apply(lambda x: any(k in x for k in macro_keywords))
        df.loc[df['Speaker'].str.contains('Governor'), 'Is_Macro'] = True
        df_macro = df[df['Is_Macro'] == True].copy()
        df_macro['Weighted_Sentiment'] = df_macro['Sentiment'] * df_macro['Weight']
        daily = df_macro.groupby('Date').apply(lambda x: x['Weighted_Sentiment'].sum() / x['Weight'].sum()).reset_index(name='Sentiment')
        return daily.sort_values('Date')

    df_clean = process_sentiment_df(df_sent)
    return df_clean, fx_data, bond_data

def helper2(df_clean, fx_data, bond_data):
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    fx_data['Date'] = pd.to_datetime(fx_data['Date']).dt.tz_localize(None)
    bond_data['Date'] = pd.to_datetime(bond_data['Date']).dt.tz_localize(None)

    # Plotting code remains the same as your request
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax1.stem(df_clean['Date'], df_clean['Sentiment'], linefmt='C0-', markerfmt='C0o', label='BNM Sentiment')
    ax1.set_ylabel('Weighted Macro Sentiment', color='C0', fontweight='bold')
    ax1.set_ylim(0, 1.2)

    ax2 = ax1.twinx()
    ax2.plot(fx_data['Date'].values, fx_data['Close'].values, color='red', linewidth=1.5, label='USD/MYR Spot', alpha=0.8)
    ax2.set_ylabel('USD/MYR Exchange Rate', color='red', fontweight='bold')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(bond_data['Date'].values, bond_data['Close'].values, color='green', linewidth=1.5, label='ABF Bond ETF (0800EA.KL)', linestyle='--')
    ax3.set_ylabel('Bond ETF Price (MYR)', color='green', fontweight='bold')

    ax1.xaxis_date() 
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()
    plt.title('BNM: Sentiment vs. Currency vs. Bond Performance', fontsize=16)
    plt.show()

    # Robust Horizon Evaluation
    def evaluate_horizons_robust(df_sent, mkt_data, horizons=[5, 30, 100], label="FX"):
        df_sent = df_sent.sort_values('Date')
        mkt_data = mkt_data.sort_values('Date')
        results = []
        for h in horizons:
            mkt_temp = mkt_data.copy()
            mkt_temp['Target_Date'] = mkt_temp['Date'] + pd.Timedelta(days=h)
            mkt_future = pd.merge_asof(mkt_temp[['Target_Date']], mkt_data[['Date', 'Close']], left_on='Target_Date', right_on='Date', direction='backward').rename(columns={'Close': 'Future_Price'})
            mkt_temp['Forward_Return'] = (mkt_future['Future_Price'] - mkt_temp['Close']) / mkt_temp['Close']
            merged = pd.merge_asof(df_sent, mkt_temp[['Date', 'Forward_Return']], on='Date', direction='backward')
            valid = merged.dropna(subset=['Forward_Return'])
            if len(valid) > 2:
                r, p = stats.pearsonr(valid['Sentiment'], valid['Forward_Return'])
                results.append({'Horizon': f'{h} Days', 'Asset': label, 'Corr': round(r, 4), 'p-value': round(p, 4)})
        return pd.DataFrame(results)

    print("\n--- Multi-Horizon Robust Analysis ---")
    fx_results = evaluate_horizons_robust(df_clean, fx_data, label="USD/MYR")
    bond_results = evaluate_horizons_robust(df_clean, bond_data, label="Bond ETF")
    print(pd.concat([fx_results, bond_results]))

# --- 4. EXECUTION ---
# Use this block to run the code
async def run_pipeline():
    df_clean, fx_data, bond_data = await helper()
    helper2(df_clean, fx_data, bond_data)

# If running in a Jupyter notebook cell:
# await run_pipeline()
