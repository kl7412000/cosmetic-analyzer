import requests
from bs4 import BeautifulSoup

BASE_URL = "https://incidecoder.com/ingredients"

def scrape(ingredient_name: str) -> dict | None:
    """
    從 INCI Decoder 爬取成分資料
    input:  成分名稱（英文）
    output: 原始資料 dict，找不到回傳 None
    """
    slug = ingredient_name.lower().replace(" ", "-")
    url = f"{BASE_URL}/{slug}"

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # 確認頁面存在
        title = soup.find("h1")
        if not title:
            return None

        name = title.get_text(strip=True)

        # 抓取 What-it-does（功能）
        functions = []
        for itemprop in soup.find_all("div", {"class": "itemprop"}):
            label = itemprop.find("span", {"class": "label"})
            if label and "What-it-does" in label.get_text():
                value = itemprop.find("span", {"class": "value"})
                if value:
                    functions = [a.get_text(strip=True) for a in value.find_all("a")]

        # 抓取 Also-called-like-this（別名）
        also_called = []
        for itemprop in soup.find_all("div", {"class": "itemprop"}):
            label = itemprop.find("span", {"class": "label"})
            if label and "Also-called" in label.get_text():
                value = itemprop.find("span", {"class": "value"})
                if value:
                    also_called = [v.strip() for v in value.get_text().split(",")]

        # 抓取 CAS number
        cas_number = ""
        cosing = soup.find("div", {"id": "cosing-data"})
        if cosing:
            for div in cosing.find_all("div"):
                text = div.get_text()
                if "CAS #:" in text:
                    cas_number = text.split("CAS #:")[1].split("|")[0].strip()

        # 抓取描述文字
        description = ""
        desc = soup.find("div", {"class": "ingredientDescription"})
        if desc:
            description = desc.get_text(strip=True)

        return {
            "source": "INCI Decoder",
            "url": url,
            "name": name,
            "also_called": also_called,
            "functions": functions,
            "description": description,
            "cas_number": cas_number,
        }

    except Exception as e:
        print(f"INCI Decoder 爬取失敗：{e}")
        return None