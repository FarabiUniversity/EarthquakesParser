"""Search API testers — все поисковые API в одном файле.

Для каждого API указано:
  - Минимальный тестовый скрипт
  - Стоимость запросов
  - Как получить ключ

Установка всех зависимостей:
  pip install requests google-generativeai tavily-python exa-py duckduckgo-search
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

QUERY = "earthquake damage statistics 2024"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. BRAVE SEARCH API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Free  — 2,000 запросов/мес, 1 req/sec
#   Paid  — от $3 за 1,000 запросов (CPM), до 20 req/sec
#   AI    — от $5/1K запросов (с LLM контекстом)
#
# Как получить ключ:
#   1. Зайти на https://api-dashboard.search.brave.com
#   2. Создать аккаунт (нужна кредитка даже для Free — только верификация)
#   3. Создать ключ → Select "Search API" → Скопировать токен
#   4. Записать в .env: BRAVE_API_KEY=...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_brave() -> None:
    """Test Brave Search API and print top 3 results."""
    key = os.environ["BRAVE_API_KEY"]
    r = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": key},
        params={"q": QUERY, "count": "5"},
    )
    data = r.json()
    for item in data.get("web", {}).get("results", [])[:3]:
        print(f"  {item['title']}\n  {item['url']}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. SERPAPI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Free  — 100 запросов/мес (без кредитки)
#   Paid  — от $75/мес за 5,000 запросов ($0.015/запрос)
#           до $0.0083/запрос на больших объёмах
#   80+ поисковых систем (Google, Bing, Yandex, Baidu, DuckDuckGo...)
#
# Как получить ключ:
#   1. Зарегистрироваться на https://serpapi.com
#   2. Dashboard → API Key → Скопировать
#   3. Записать в .env: SERPAPI_KEY=...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_serpapi() -> None:
    """Test SerpAPI and print top 3 organic results."""
    key = os.environ["SERPAPI_KEY"]
    r = requests.get(
        "https://serpapi.com/search",
        params={"q": QUERY, "api_key": key, "engine": "google", "num": "5"},
    )
    data = r.json()
    for item in data.get("organic_results", [])[:3]:
        print(f"  {item['title']}\n  {item['link']}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. SERPER.DEV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Free  — 2,500 запросов (единоразово при регистрации)
#   Paid  — от $50 за 50,000 запросов (~$1/1K)
#           на больших объёмах — до ~$0.30/1K
#   Только Google, быстрый (~2.9 сек)
#
# Как получить ключ:
#   1. Зарегистрироваться на https://serper.dev
#   2. Dashboard → API Key → Скопировать
#   3. Записать в .env: SERPER_API_KEY=...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_serper() -> None:
    """Test Serper.dev API and print top 3 organic results."""
    key = os.environ["SERPER_API_KEY"]
    r = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": QUERY, "num": 5},
    )
    data = r.json()
    for item in data.get("organic", [])[:3]:
        print(f"  {item['title']}\n  {item['link']}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. TAVILY (AI-native search)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Free  — 1,000 кредитов/мес (1 search = 1 кредит)
#   Pay-as-you-go — $0.008/кредит
#   Bootstrap — $75/мес
#   Growth    — $500/мес
#   Оптимизирован для RAG/LLM пайплайнов
#
# Как получить ключ:
#   1. Зарегистрироваться на https://app.tavily.com
#   2. API Keys → Create → Скопировать
#   3. Записать в .env: TAVILY_API_KEY=...
#   pip install tavily-python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_tavily() -> None:
    """Test Tavily AI search API and print top 3 results."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(QUERY, max_results=3)
    for item in response.get("results", []):
        print(f"  {item['title']}\n  {item['url']}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. EXA.AI (Semantic/Neural search)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Free  — $10 кредитов при регистрации (~2,000 запросов)
#   Paid  — ~$5/1K запросов (базовый)
#   Нейронный поиск на основе embeddings, "Find Similar"
#
# Как получить ключ:
#   1. Зарегистрироваться на https://dashboard.exa.ai
#   2. API Keys → Скопировать
#   3. Записать в .env: EXA_API_KEY=...
#   pip install exa-py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_exa() -> None:
    """Test Exa.ai neural search API and print top 3 results."""
    from exa_py import Exa

    exa = Exa(api_key=os.environ["EXA_API_KEY"])
    results = exa.search(QUERY, num_results=3, type="neural")
    for item in results.results:
        print(f"  {item.title}\n  {item.url}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. DUCKDUCKGO (free, no API key required)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   БЕСПЛАТНО — неофициальный API через duckduckgo-search
#   Без ключа, без регистрации
#   pip install duckduckgo-search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_duckduckgo() -> None:
    """Test DuckDuckGo search (no API key required) and print top 3 results."""
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(QUERY, max_results=3))
    for item in results:
        print(f"  {item.get('title')}\n  {item.get('href')}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. YANDEX XML (Yandex Search API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Free  — 0 (есть лимиты, зависят от «ИКС» сайта)
#           Обычно ~1,000 запросов/сутки для нового аккаунта
#   Paid  — от $0.25 до $4 за 1K запросов
#   Ответ в XML (не JSON). Требуется логотип Яндекса при показе.
#
# Как получить ключ:
#   1. Иметь Яндекс-аккаунт
#   2. Зайти на https://xml.yandex.ru
#   3. Привязать сайт (можно любой)
#   4. Получить user + key
#   5. Записать в .env: YANDEX_USER=... YANDEX_KEY=...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_yandex() -> None:
    """Test Yandex XML search API and print status and response preview."""
    user = os.environ["YANDEX_USER"]
    key = os.environ["YANDEX_KEY"]
    r = requests.get(
        "https://yandex.com/search/xml",
        params={
            "user": user,
            "key": key,
            "query": QUERY,
            "l10n": "en",
            "groupby": "attr=d.mode=deep.groups-on-page=3",
        },
    )
    print(f"  Status: {r.status_code}")
    print(f"  Response (first 500 chars):\n  {r.text[:500]}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. SEARCHAPI.IO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Free  — 100 запросов (при регистрации)
#   Paid  — от $40/мес за 10,000 запросов ($4/1K)
#           до $1/1K на 5M запросов/мес
#   Google, Bing, Baidu, Scholar, YouTube и др.
#   99.9% SLA, юридическая защита ($2M legal protection)
#
# Как получить ключ:
#   1. Зарегистрироваться на https://www.searchapi.io
#   2. Dashboard → API Key
#   3. Записать в .env: SEARCHAPI_KEY=...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_searchapi() -> None:
    """Test SearchAPI.io and print top 3 organic results."""
    key = os.environ["SEARCHAPI_KEY"]
    r = requests.get(
        "https://www.searchapi.io/api/v1/search",
        params={"q": QUERY, "api_key": key, "engine": "google", "num": "5"},
    )
    data = r.json()
    for item in data.get("organic_results", [])[:3]:
        print(f"  {item['title']}\n  {item['link']}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. GEMINI (Google Generative AI with grounding)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Free  — 1,500 запросов/день (Gemini 1.5 Flash)
#   Paid  — от $0.075/1M токенов (Flash)
#   Grounding через Google Search — дополнительно
#
# Как получить ключ:
#   1. Зайти на https://aistudio.google.com
#   2. Get API Key → Создать
#   3. Записать в .env: GEMINI_API_KEY=...
#   pip install google-generativeai
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_gemini() -> None:
    """Test Google Gemini API with search grounding and print the response."""
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(QUERY)
    print(f"  {str(response.text)[:500]}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. DATAFORSEO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Free  — $1 кредит при регистрации (~тест)
#   Paid  — от $0.0006/запрос ($0.60/1K) — один из самых дешёвых
#           $60 за 100K запросов
#   SEO-ориентированный: ранжирование, ключевые слова, SERP
#
# Как получить ключ:
#   1. Зарегистрироваться на https://dataforseo.com
#   2. Dashboard → API Credentials (login + password, Basic Auth)
#   3. Записать в .env: DATAFORSEO_LOGIN=... DATAFORSEO_PASSWORD=...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_dataforseo() -> None:
    """Test DataForSEO SERP API and print top 3 organic results."""
    login = os.environ["DATAFORSEO_LOGIN"]
    password = os.environ["DATAFORSEO_PASSWORD"]
    r = requests.post(
        "https://api.dataforseo.com/v3/serp/google/organic/live/advanced",
        auth=(login, password),
        json=[{"keyword": QUERY, "location_code": 2398, "language_code": "en"}],
    )
    data = r.json()
    tasks = data.get("tasks", [{}])
    if tasks and tasks[0].get("result"):
        for item in tasks[0]["result"][0].get("items", [])[:3]:
            if item.get("type") == "organic":
                print(f"  {item.get('title')}\n  {item.get('url')}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. SCRAPINGDOG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Free  — 1,000 кредитов при регистрации
#   Paid  — от $40/мес за 5,000 запросов
#           на масштабе — до $0.00029/запрос (самый дешёвый на рынке)
#   Универсальный SERP + скрапинг Amazon, LinkedIn и пр.
#
# Как получить ключ:
#   1. Зарегистрироваться на https://www.scrapingdog.com
#   2. Dashboard → API Key
#   3. Записать в .env: SCRAPINGDOG_KEY=...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_scrapingdog() -> None:
    """Test ScrapingDog Google SERP API and print top 3 results."""
    key = os.environ["SCRAPINGDOG_KEY"]
    r = requests.get(
        "https://api.scrapingdog.com/google",
        params={"api_key": key, "query": QUERY, "results": "5"},
    )
    data = r.json()
    for item in data.get("organic_results", [])[:3]:
        print(f"  {item.get('title')}\n  {item.get('link')}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. PERPLEXITY SONAR API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   sonar       — $1/1K запросов (быстрый, лёгкий)
#   sonar-pro   — $3/1K запросов (глубокий поиск)
#   Возвращает готовый ответ с цитатами (не список ссылок!)
#
# Как получить ключ:
#   1. Зарегистрироваться на https://www.perplexity.ai
#   2. Settings → API → Generate API Key
#   3. Записать в .env: PERPLEXITY_API_KEY=...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_perplexity() -> None:
    """Test Perplexity Sonar API and print first 500 chars of the answer."""
    key = os.environ["PERPLEXITY_API_KEY"]
    r = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "sonar",
            "messages": [{"role": "user", "content": QUERY}],
        },
    )
    data = r.json()
    print(f"  {data['choices'][0]['message']['content'][:500]}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. YOU.COM WEB SEARCH API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Цены по запросу (enterprise-oriented)
#   Акцент на точность и цитируемость
#   2-3x быстрее конкурентов (по их заявлению)
#
# Как получить ключ:
#   1. Зарегистрироваться на https://api.you.com
#   2. Dashboard → API Key
#   3. Записать в .env: YOU_API_KEY=...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_you() -> None:
    """Test YOU.com web search API and print top 3 hits."""
    key = os.environ["YOU_API_KEY"]
    r = requests.get(
        "https://api.ydc-index.io/search",
        headers={"X-API-Key": key},
        params={"query": QUERY},
    )
    data = r.json()
    for hit in data.get("hits", [])[:3]:
        print(f"  {hit.get('title')}\n  {hit.get('url')}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14. LINKUP WEB SEARCH API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Free  — €5 кредитов при регистрации
#   Paid  — pay-as-you-go, цены по запросу
#   SERP + Web Search + LLM коннекторы (LangChain, MCP)
#
# Как получить ключ:
#   1. Зарегистрироваться на https://app.linkup.so
#   2. Dashboard → API Key
#   3. Записать в .env: LINKUP_API_KEY=...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_linkup() -> None:
    """Test Linkup web search API and print top 3 results."""
    key = os.environ["LINKUP_API_KEY"]
    r = requests.post(
        "https://api.linkup.so/v1/search",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={"q": QUERY, "depth": "standard", "outputType": "searchResults"},
    )
    data = r.json()
    for item in data.get("results", [])[:3]:
        print(f"  {item.get('name')}\n  {item.get('url')}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 15. SEARXNG (Self-hosted, бесплатный!)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   100% БЕСПЛАТНО — open source, self-hosted
#   Агрегирует результаты из 247 поисковых систем
#   Запуск: docker run -p 8080:8080 searxng/searxng
#
# Установка:
#   docker run -d -p 8080:8080 searxng/searxng
#   Или свой VPS / Raspberry Pi
#   Не нужен API-ключ! Просто запрос к localhost.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_searxng() -> None:
    """Test self-hosted SearXNG instance and print top 3 results."""
    base_url = os.environ.get("SEARXNG_URL", "http://localhost:8080")
    r = requests.get(
        f"{base_url}/search",
        params={
            "q": QUERY,
            "format": "json",
            "engines": "google,bing,duckduckgo",
        },
    )
    data = r.json()
    for item in data.get("results", [])[:3]:
        print(f"  {item['title']}\n  {item['url']}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 16. FIRECRAWL SEARCH API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Стоимость:
#   Free  — 500 кредитов (бессрочно)
#   Starter — $19/мес (3,000 кредитов)
#   Standard — $83/мес (100,000 кредитов)
#   Growth   — $333/мес (500,000 кредитов)
#   Поиск + полное извлечение контента в Markdown (для LLM)
#
# Как получить ключ:
#   1. Зарегистрироваться на https://www.firecrawl.dev
#   2. Dashboard → API Key
#   3. Записать в .env: FIRECRAWL_API_KEY=...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_firecrawl() -> None:
    """Test Firecrawl search API and print top 3 results with Markdown content."""
    key = os.environ["FIRECRAWL_API_KEY"]
    r = requests.post(
        "https://api.firecrawl.dev/v1/search",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={"query": QUERY, "limit": 3},
    )
    data = r.json()
    for item in data.get("data", [])[:3]:
        print(f"  {item.get('title', 'N/A')}\n  {item.get('url')}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# СВОДНАЯ ТАБЛИЦА СТОИМОСТИ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRICING_TABLE = (
    "\nAPI              | Free Tier            | Price/1K req     | Key?\n"
    "-----------------|----------------------|------------------|----------\n"
    "Brave Search     | 2,000/mo             | from $3/1K       | Yes\n"
    "SerpAPI          | 100/mo               | $15->$8.3/1K     | Yes\n"
    "Serper.dev       | 2,500 (once)         | $1->$0.30/1K     | Yes\n"
    "Tavily           | 1,000 credits/mo     | $8/1K PAYG       | Yes\n"
    "Exa.ai           | $10 credits          | ~$5/1K           | Yes\n"
    "DuckDuckGo       | unlimited            | FREE             | No\n"
    "Yandex XML       | ~1,000/day           | $0.25-$4/1K      | Yes\n"
    "SearchAPI.io     | 100 (once)           | $4->$1/1K        | Yes\n"
    "Gemini           | 1,500/day (Flash)    | $0.075/1M tokens | Yes\n"
    "DataForSEO       | $1 credit            | from $0.60/1K    | Yes (Basic)\n"
    "ScrapingDog      | 1,000 credits        | $8->$0.29/1K     | Yes\n"
    "Perplexity Sonar | —                    | $1/1K (sonar)    | Yes\n"
    "YOU.com          | —                    | on request       | Yes\n"
    "Linkup           | EUR 5 credits        | on request       | Yes\n"
    "SearXNG          | unlimited self-host  | FREE             | No\n"
    "Firecrawl        | 500 credits (perm.)  | ~$0.83/1K        | Yes\n"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL_TESTS = {
    "brave": test_brave,
    "serpapi": test_serpapi,
    "serper": test_serper,
    "tavily": test_tavily,
    "exa": test_exa,
    "duckduckgo": test_duckduckgo,
    "yandex": test_yandex,
    "searchapi": test_searchapi,
    "gemini": test_gemini,
    "dataforseo": test_dataforseo,
    "scrapingdog": test_scrapingdog,
    "perplexity": test_perplexity,
    "you": test_you,
    "linkup": test_linkup,
    "searxng": test_searxng,
    "firecrawl": test_firecrawl,
}

if __name__ == "__main__":
    import sys

    print(PRICING_TABLE)

    if len(sys.argv) > 1:
        names: list[str] = list(sys.argv[1:])
    else:
        names = list(ALL_TESTS.keys())

    for name in names:
        if name == "pricing":
            continue
        fn = ALL_TESTS.get(name)
        if not fn:
            print(f"❌ Unknown API: {name}")
            continue
        print(f"{'═' * 60}")
        print(f"▶ Testing: {name.upper()}")
        print(f"{'═' * 60}")
        try:
            fn()
            print("✅ OK\n")
        except KeyError as e:
            print(f"⚠️  Env var not set: {e} — пропускаю\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
