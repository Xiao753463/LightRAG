import json
import time
import argparse
from datetime import datetime
from dateutil import relativedelta
from dateutil import parser
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service


def parse_relative_date(text):
    now = datetime.now()
    try:
        if '天前' in text:
            days = int(text.replace('天前', '').strip())
            return now - relativedelta.relativedelta(days=days)
        elif '個月前' in text:
            months = int(text.replace('個月前', '').strip())
            return now - relativedelta.relativedelta(months=months)
        elif '年前' in text:
            years = int(text.replace('年前', '').strip())
            return now - relativedelta.relativedelta(years=years)
    except:
        pass
    return now  # fallback


def get_reviews(store_name: str, start_date_str: str, end_date_str: str, driver_path: str = 'chromedriver') -> list:
    # 解析時間區間
    start_date = datetime.fromisoformat(start_date_str)
    end_date = datetime.fromisoformat(end_date_str)

    # 設定 Chrome 無頭模式
    options = Options()
    options.add_argument('--disable-gpu')
    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service, options=options)

    # 開啟 Google Maps
    driver.get('https://www.google.com/maps')
    wait = WebDriverWait(driver, 10)

    # 搜尋店面名稱
    search_box = wait.until(
        EC.element_to_be_clickable((By.ID, 'searchboxinput')))
    search_box.clear()
    search_box.send_keys(store_name)
    search_box.send_keys(Keys.ENTER)

    # 等待搜尋結果載入
    wait.until(EC.presence_of_element_located(
        (By.XPATH, "//button[contains(@aria-label, ' reviews') or contains(@aria-label, ' 則評論')]")))

    # 點擊「評論」按鈕
    reviews_button = driver.find_element(
        By.XPATH, "//button[contains(@aria-label, ' reviews') or contains(@aria-label, ' 則評論')]")
    print("找到評論按鈕，嘗試點擊")
    reviews_button.click()

    # 等待評論列表面板載入
    panel = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//div[@class='m6QErb DxyBCb kA9KIf dS8AEf XiKgde ']")))
    print("評論面板載入成功")

    # 不斷滾動至底部，確保載入所有評論
    last_height = driver.execute_script(
        "return arguments[0].scrollHeight", panel)
    while True:
        driver.execute_script(
            'arguments[0].scrollTo(0, arguments[0].scrollHeight);', panel)
        time.sleep(2)
        new_height = driver.execute_script(
            "return arguments[0].scrollHeight", panel)
        if new_height == last_height:
            break
        last_height = new_height

    # 擷取評論內容與時間
    reviews = []
    elems = panel.find_elements(
        By.XPATH, ".//div[contains(@class, 'jftiEf') and contains(@class, 'fontBodyMedium')]")
    print(f"找到 {len(elems)} 則總評論")
    for elem in elems:
        try:
            more_button = elem.find_element(
                By.XPATH, ".//button[contains(@aria-label, '顯示更多') or contains(text(), '更多')]")
            driver.execute_script("arguments[0].click();", more_button)
            time.sleep(0.5)
        except:
            pass  # 若無可點擊的按鈕則略過

        try:
            date_text = elem.find_element(
                By.XPATH, ".//span[contains(@class,'rsqaWe')]").text
            review_date = parse_relative_date(date_text)
        except Exception as e:
            print("解析時間失敗，略過評論：", e)
            continue
        if not (start_date <= review_date <= end_date):
            continue

        try:
            content = elem.find_element(
                By.XPATH, ".//span[@class='wiI7pd']").text
        except:
            try:
                content = elem.find_element(
                    By.XPATH, ".//span[@class='w8nwRe']").text
            except:
                content = ""
        print(f"通過時間篩選：{review_date.strftime('%Y-%m-%d')} 內容：{content[:30]}")
        reviews.append({
            'brand': store_name.replace(' ', ''),
            'review': content,
            'date': review_date.strftime('%Y-%m-%d')
        })

    driver.quit()
    return reviews


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='爬取 Google Maps 指定店面在設定時間區間的評論')
    parser.add_argument('--store', required=True, help='店面名稱')
    parser.add_argument('--start', required=True, help='開始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='結束日期 (YYYY-MM-DD)')
    parser.add_argument('--driver', default='chromedriver',
                        help='ChromeDriver 可執行檔路徑')
    args = parser.parse_args()

    results = get_reviews(args.store, args.start, args.end, args.driver)

    # 輸出為 JSON 檔案
    with open('reviews_output.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("已將結果儲存至 reviews_output.json")
