import asyncio
from playwright.async_api import async_playwright
from datetime import datetime, timedelta

start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 2, 28)

async def download_month_files(page, month_text):
    download_count = 0
    
    try:
        # Find and click dropdown trigger buttons
        dropdown_containers = await page.query_selector_all('div.dropdown')
        
        for container in dropdown_containers:
            try:
                # Find the dropdown trigger button within this container
                trigger = await container.query_selector('button.button.is-small:has(i.fas.fa-ellipsis-h)')
                
                if trigger:
                    # Force click with JavaScript to ensure dropdown opens
                    await trigger.evaluate('node => node.click()')
                    await page.wait_for_timeout(500)
                    
                    # Find all download buttons in this dropdown
                    download_buttons = await container.query_selector_all('a.dropdown-item:has(i.fas.fa-download)')
                    
                    for download_button in download_buttons:
                        try:
                            # Expect download and click
                            async with page.expect_download() as download_info:
                                await download_button.click()
                            
                            # Save the download
                            download = await download_info.value
                            download_count += 1
                            filename = f"rain_{month_text}{download_count}.zip"
                            await download.save_as(filename)
                            print(f"Downloaded: {filename}")
                            
                            # Wait a moment after download
                            await page.wait_for_timeout(100)
                        
                        except Exception as e:
                            print(f"Error downloading file in {month_text}: {e}")
            
            except Exception as e:
                print(f"Error processing dropdown in {month_text}: {e}")
    
    except Exception as e:
        print(f"Error in download process for {month_text}: {e}")
    
    return download_count

async def download_files():
    base_url = "https://history.colife.org.tw/#/?cd=%2F氣象%2F中央氣象署_雨量站"
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        # Navigate to the base URL
        await page.goto(base_url)
        await page.wait_for_load_state('networkidle')
        
        # Initial navigation steps
        initial_steps = ["氣象", "中央氣象署_雨量站"]
        
        for link_text in initial_steps:
            try:
                link = await page.query_selector(f'a.is-block.name:text("{link_text}")')
                if link:
                    await link.click()
                    await page.wait_for_timeout(1000)  # Wait for page to load
                else:
                    print(f"Could not find '{link_text}' link")
                    await browser.close()
                    return
            except Exception as e:
                print(f"Error clicking {link_text}: {e}")
                await browser.close()
                return
        
        current_date = start_date
        total_downloads = 0
        
        while current_date <= end_date:
            month_text = current_date.strftime("%Y%m")
            
            try:
                # Click on month
                month_link = await page.query_selector(f'a.is-block.name:text("{month_text}")')
                if month_link:
                    await month_link.click()
                    await page.wait_for_timeout(1000)  # Wait for page to load
                else:
                    print(f"Could not find '{month_text}' link")
                    break
                
                # Download files for this month
                month_downloads = await download_month_files(page, month_text)
                total_downloads += month_downloads
                print(f"Downloaded {month_downloads} files for {month_text}")
                
                # Go back to water resources sensor page
                water_sensor_link = await page.query_selector('a:text("中央氣象署_雨量站")')
                if water_sensor_link:
                    await water_sensor_link.click()
                    await page.wait_for_timeout(1000)
                else:
                    print("Could not return to water sensor page")
                    break
            
            except Exception as e:
                print(f"Error processing {month_text}: {e}")
            
            # Move to next month
            current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)
        
        print(f"Total files downloaded: {total_downloads}")
        await browser.close()

# For Jupyter Notebook or IPython
import nest_asyncio
nest_asyncio.apply()

# Run with get_event_loop
loop = asyncio.get_event_loop()
loop.run_until_complete(download_files())