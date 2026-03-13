import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

BASE_URL = "https://ec.europa.eu/growth/tools-databases/cosing"

_NOISE = {
    "PDF", "CLOSE", "BACK", "HOME", "ADVANCED SEARCH",
    "REFERENCE DATA", "USER MANUAL", "SEARCH",
}

_STOP_WORDS = {
    "SCCS opinions",
    "Go back",
    "Cosmetics Regulation provisions",
    "Identified INGREDIENTS or substances e.g.",
    "Description",
    "CosIng - Cosmetics Ingredients",
    "Accessibility",
    "Chemical / IUPAC Name",
    "Note",
    "Other Directives / Regulations",
}


def _strip_h1_prefix(h1_text: str) -> str:
    """去掉 'Ingredient: ' 或 'Substance: ' 前綴"""
    for prefix in ("Ingredient: ", "Substance: "):
        if h1_text.startswith(prefix):
            return h1_text[len(prefix):]
    return h1_text


async def _get_detail_id(inci_name: str) -> str | None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        page.set_default_timeout(60000)
        try:
            await page.goto(f"{BASE_URL}/")
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(2000)
            try:
                await page.click("#accept")
                await page.wait_for_timeout(1000)
            except Exception:
                pass
            inputs = await page.query_selector_all("input[type=text]")
            if not inputs:
                return None
            await inputs[0].fill(inci_name)
            await page.click("button:has-text('Search')")
            await page.wait_for_timeout(3000)
            links = await page.query_selector_all("tbody td a")
            for link in links:
                text = await link.inner_text()
                href = await link.get_attribute("href")
                if text.strip().upper() == inci_name.upper() and href:
                    return href.split("/")[-1]
            return None
        except Exception as e:
            print(f"[CosIng] 搜尋失敗：{e}")
            return None
        finally:
            await browser.close()


async def _get_detail(detail_id: str, inci_name: str) -> dict | None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        page.set_default_timeout(60000)
        try:
            await page.goto(f"{BASE_URL}/details/{detail_id}")
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(3000)
            try:
                await page.click("#accept")
                await page.wait_for_timeout(1000)
            except Exception:
                pass
            content = await page.content()
        except Exception as e:
            print(f"[CosIng] 取得詳細資料失敗：{e}")
            return None
        finally:
            await browser.close()

    soup = BeautifulSoup(content, "html.parser")

    # 從 h1 判斷是 Ingredient 還是 Substance 頁面
    h1 = soup.find("h1")
    h1_text = h1.get_text(strip=True) if h1 else ""
    is_substance = h1_text.startswith("Substance:")
    parsed_inci = _strip_h1_prefix(h1_text).upper() if h1_text else inci_name.upper()

    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    cas_number = ""
    ec_number = ""
    functions = []
    annex_refs = []
    max_concentration = ""
    regulation = ""
    name_upper = parsed_inci.upper()

    i = 0
    while i < len(lines):
        line = lines[i]

        # CAS 號碼
        if line == "CAS #" and i + 1 < len(lines):
            val = lines[i + 1]
            if val not in ("EC #", "INCI Name", "Substance", "Description",
                           "Functions", "INN / ISO / AN",
                           "Identified INGREDIENTS or substances e.g."):
                cas_number = val
            i += 2
            continue

        # EC 號碼
        if line == "EC #" and i + 1 < len(lines):
            val = lines[i + 1]
            if val not in ("CAS #", "INCI Name", "Substance", "Description",
                           "Functions", "INN / ISO / AN",
                           "Identified INGREDIENTS or substances e.g."):
                ec_number = val
            i += 2
            continue

        # Functions（Ingredient 頁面才有）
        if line == "Functions" and not is_substance:
            j = i + 1
            while j < len(lines) and lines[j] not in _STOP_WORDS:
                val = lines[j].strip()
                if (val
                        and val == val.upper()
                        and len(val) > 2
                        and val not in _NOISE
                        and not val[0].isdigit()
                        and "/" not in val
                        and val != name_upper
                        and val != "PDF"
                        and val != "CAS #"
                        and val != "EC #"
                        and "%" not in val
                ):
                    functions.append(val)
                j += 1
            i = j
            continue

        # Substance 頁面：法規限制濃度
        if line == "Maximum concentration in ready for use preparation" and i + 1 < len(lines):
            max_concentration = lines[i + 1]
            i += 2
            continue

        # Substance 頁面：法規編號
        if line == "Regulation" and i + 1 < len(lines):
            regulation = lines[i + 1]
            i += 2
            continue

        # Annex / Ref #
        if line == "Annex / Ref #" and i + 1 < len(lines):
            val = lines[i + 1]
            if val and val != "Go back":
                annex_refs.append(val)
            i += 2
            continue

        i += 1

    # Substance 頁面把法規資訊整理進 annex_refs
    if is_substance and max_concentration:
        annex_refs.append(f"最高濃度：{max_concentration}")
    if is_substance and regulation:
        annex_refs.insert(0, f"法規：{regulation}")

    return {
        "source": "CosIng",
        "url": f"{BASE_URL}/details/{detail_id}",
        "inci_name": parsed_inci,
        "is_substance": is_substance,
        "cas_number": cas_number,
        "ec_number": ec_number,
        "functions": functions,
        "annex_refs": annex_refs,
    }


def scrape(inci_name: str) -> dict | None:
    """
    爬取 CosIng 成分資料（同步介面）。

    注意：請傳入 INCI 名稱，不是俗名。
    例如：scrape("Tocopherol") 而不是 scrape("Vitamin E")

    回傳：
        dict  — 找到資料
        None  — 找不到（交給 enricher.py 用 LLM 生成）
    """
    async def _run():
        detail_id = await _get_detail_id(inci_name)

        # 找不到時，嘗試去掉句點再搜一次（處理 "Alcohol Denat." 等情況）
        if not detail_id and inci_name.endswith("."):
            detail_id = await _get_detail_id(inci_name.rstrip("."))

        if not detail_id:
            print(f"[CosIng] 找不到成分：{inci_name}")
            return None
        return await _get_detail(detail_id, inci_name)

    return asyncio.run(_run())


if __name__ == "__main__":
    import json

    test_cases = [
        "Niacinamide",
        "Tocopherol",
        "Retinol",
        "Sodium Hyaluronate",
        "Salicylic Acid",
        "Zinc Oxide",
        "Methylparaben",
        "Phenoxyethanol",
        "Parfum",
        "Alcohol Denat.",
    ]

    for name in test_cases:
        print(f"\n{'='*50}")
        print(f"測試：{name}")
        result = scrape(name)
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("→ 找不到，需要 LLM fallback")