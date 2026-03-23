import os
import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

def sanitize_filename(filename):
    """Removes characters that aren't allowed in filenames."""
    # Limits length and removes special chars like / \ : * ? " < > |
    return re.sub(r'(?u)[^-\w.]', '_', filename)[:100]

def download_bnm_speeches(total_limit=200):
    # 1. Setup storage directory
    output_folder = "bnm_speeches"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created folder: {output_folder}")
    
    metadata = []
    
    with sync_playwright() as p:
        # 2. Launch Browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        page = context.new_page()
        
        print("🚀 Accessing BNM Speeches Archive...")
        page.goto("https://www.bnm.gov.my/speeches-interviews", wait_until="networkidle")
        
        page_num = 1
        
        while len(metadata) < total_limit:
            print(f"\n--- Scraping Index Page {page_num} ---")
            
            # Parse table on current page
            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')
            table_body = soup.find('tbody', id='myTable')
            if not table_body:
                print("⚠️ Could not find table body. Ending.")
                break
                
            rows = table_body.find_all('tr')
            
            for row in rows:
                if len(metadata) >= total_limit: break
                
                cells = row.find_all('td')
                if not cells or len(cells) < 2: continue
                
                date = cells[0].get_text(strip=True)
                title = cells[1].get_text(strip=True)
                anchor = cells[1].find('a')
                if not anchor: continue
                
                href = anchor['href']
                link = href if href.startswith('http') else f"https://www.bnm.gov.my{href}"
                
                print(f"📄 [{len(metadata)+1}] Downloading: {title[:60]}...")
                
                # Visit the specific speech page
                detail_page = context.new_page()
                try:
                    # 'domcontentloaded' is faster than 'networkidle' for just text
                    detail_page.goto(link, wait_until="domcontentloaded", timeout=30000)
                    detail_html = detail_page.content()
                    s_soup = BeautifulSoup(detail_html, 'html.parser')
                    
                    # Target the main article container
                    content_div = s_soup.find('div', class_='article-content-cs')
                    if content_div:
                        # Extract all text from paragraphs
                        text_content = "\n\n".join([p.get_text(strip=True) for p in content_div.find_all('p')])
                    else:
                        text_content = "Could not parse content div."

                    # 3. SAVE TO FILE
                    safe_title = sanitize_filename(title)
                    file_name = f"{date}_{safe_title}.txt"
                    file_path = os.path.join(output_folder, file_name)
                    
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(f"TITLE: {title}\n")
                        f.write(f"DATE: {date}\n")
                        f.write(f"URL: {link}\n")
                        f.write("-" * 50 + "\n\n")
                        f.write(text_content)

                    metadata.append({
                        'Date': date, 
                        'Title': title, 
                        'Link': link, 
                        'Local_Path': file_path
                    })
                    
                except Exception as e:
                    print(f"   ❌ Failed to download: {e}")
                finally:
                    detail_page.close() # Close the tab immediately to save RAM

            # 4. PAGINATION (Check for 'Next' button)
            if len(metadata) < total_limit:
                next_button = page.locator("li.next:not(.disabled) a")
                if next_button.count() > 0:
                    print("⏭️ Moving to next index page...")
                    next_button.click()
                    page.wait_for_load_state("networkidle")
                    page_num += 1
                else:
                    print("🏁 Reached end of archive.")
                    break
        
        browser.close()
    
    print(f"\n✅ Done! Downloaded {len(metadata)} speeches to '{output_folder}'.")
    return pd.DataFrame(metadata)

# Usage
df = download_bnm_speeches(total_limit=200)
