# Veritatis Fact-Checker

RAG-система на LangGraph для оценки достоверности утверждений об землетрясениях.
Принимает произвольное утверждение, ищет похожие записи в Tier 2 Milvus-коллекции и возвращает оценку правдивости от **0.0** до **1.0**.

---

## Содержание

1. [Архитектура](#архитектура)
2. [Компоненты](#компоненты)
   - [Milvus Tier 2](#milvus-tier-2)
   - [Retriever — query_collection](#retriever--query_collection)
   - [LangGraph Tool — retrieve_facts](#langgraph-tool--retrieve_facts)
   - [ReAct-агент](#react-агент)
   - [Structured output — FactCheckResult](#structured-output--factcheckresult)
3. [Поток данных](#поток-данных)
4. [Ключевое ограничение](#ключевое-ограничение)
5. [Конфигурация](#конфигурация)
6. [API-эндпоинт](#api-эндпоинт)
7. [Использование напрямую из Python](#использование-напрямую-из-python)
8. [Зависимости](#зависимости)

---

## Архитектура

```
Утверждение (claim)
       │
       ▼
 EmbeddingGenerator              sentence-transformers/all-MiniLM-L6-v2
 embed(claim) → vector[384]
       │
       ▼
 Milvus veritatis_tier2          COSINE HNSW, top-10
 vector_search(tier=2)
       │
       ▼  список metadata-документов
      {iid, credibility_score, date, domain, distance}
       │
       ▼
 LangGraph ReAct Agent           gpt-4 via локальный OpenAI-прокси
 System prompt + retrieve_facts tool
       │
       ▼
 FactCheckResult (Pydantic)
  • score    : float 0.0–1.0
  • reasoning: str
  • sources  : list[str]
```

---

## Компоненты

### Milvus Tier 2

**Коллекция:** `veritatis_tier2`

Tier 2 содержит записи, которые прошли проверку достоверности источника (`credibility_score >= 0.7`).
Текст статей **не хранится** в Milvus — только эмбеддинг и метаданные.

| Поле               | Тип                    | Описание                                        |
|--------------------|------------------------|-------------------------------------------------|
| `iid`              | VARCHAR (PK, max 64)   | UUID из `parsed_content.id` (Supabase)          |
| `embedding`        | FLOAT_VECTOR (dim=384) | Нормализованный вектор, all-MiniLM-L6-v2        |
| `credibility_score`| FLOAT                  | Оценка достоверности источника (0.0–1.0)        |
| `date`             | INT64                  | Дата события в epoch milliseconds               |
| `domain`           | VARCHAR (max 500)      | Домен источника (напр. `usgs.gov`)              |

Индекс: **HNSW**, метрика: **COSINE**.

---

### Retriever — `query_collection`

`veritatis/veritatis/agent.py`

```python
def query_collection(claim: str, top_k: int = 10) -> list[dict[str, Any]]:
```

Обёртка над `plain_search.vector_search(tier=2)`.
Внутри делает:
1. `embedding_generator.embed(claim)` — строит вектор запроса.
2. `Collection.search(...)` по `veritatis_tier2` с `metric_type=COSINE, ef=128`.
3. Возвращает топ-`top_k` словарей:

```python
{
    "iid": "3f2a...",
    "tier": 2,
    "credibility_score": 0.82,
    "date": 1693526400000,     # epoch ms
    "domain": "usgs.gov",
    "distance": 0.74           # COSINE similarity, выше = релевантнее
}
```

---

### LangGraph Tool — `retrieve_facts`

```python
@tool
def retrieve_facts(claim: str) -> list[dict]:
    """Retrieve up to 10 relevant earthquake records from the credible (Tier 2) database."""
    return query_collection(claim)
```

Агент может вызвать инструмент **несколько раз** — например, переформулировав запрос, если первый прогон вернул мало результатов или все `distance < 0.3`.

---

### ReAct-агент

Строится через `langgraph.prebuilt.create_react_agent`:

```python
agent = create_react_agent(
    llm,                          # ChatOpenAI → локальный gpt-4
    tools=[retrieve_facts],
    prompt=_SYSTEM_PROMPT,
    response_format=FactCheckResult,
)
```

**Системный промпт** объясняет агенту:
- что поля `distance` и `credibility_score` означают;
- как на их основании выставить `score`;
- когда нужно повторить поиск;
- что оценка основывается **только** на данных из БД, а не на предобученных знаниях.

**Логика оценки**, которую агент применяет при рассуждении:

| Сигнал | Влияние на score |
|--------|------------------|
| `distance ≥ 0.6` + `credibility_score ≥ 0.7` | сильное подтверждение |
| Много записей из разных `domain` | независимые источники → score выше |
| `distance < 0.3` у всех результатов | тема не покрыта базой → score ≈ 0.5 (неизвестно) |
| Единственная запись с низким score | score снижается |

---

### Structured output — `FactCheckResult`

```python
class FactCheckResult(BaseModel):
    score: float          # 0.0 (ложь/не подтверждено) → 1.0 (хорошо подтверждено)
    reasoning: str        # пошаговое объяснение оценки
    sources: list[str]    # домены задействованных документов
```

LangGraph извлекает structured output через `model.with_structured_output(FactCheckResult)` на финальном шаге агента. Результат доступен в `result["structured_response"]`.

---

## Поток данных

```
1. fact_check("Earthquake M7.5 hit Turkey in 2023")
       │
2.     └─► LangGraph: HumanMessage → агент выбирает действие
       │
3.         retrieve_facts("Earthquake M7.5 hit Turkey in 2023")
               │
               └─► embed(claim) → вектор 384d
               └─► Milvus COSINE search → top-10 hits
               └─► return list[dict]  (metadata only, no text)
       │
4.     агент рассуждает над metadata:
           distance=[0.81, 0.76, 0.71, ...], credibility=[0.87, 0.92, ...]
           domains=["usgs.gov", "emsc.eu", "reuters.com", ...]
       │
5.     при необходимости — повторный retrieve_facts с уточнённым запросом
       │
6.     финальный шаг: structured output → FactCheckResult
               score=0.88
               reasoning="Found 8 records with distance ≥ 0.65 ..."
               sources=["usgs.gov", "emsc.eu", "reuters.com"]
       │
7.     return FactCheckResult
```

---

## Ключевое ограничение

> **Текст статей не хранится в Milvus.**

Агент рассуждает исключительно по **метаданным**:
- насколько семантически похож вектор записи на вектор утверждения (`distance`),
- насколько был оценён источник (`credibility_score`),
- откуда пришла запись (`domain`).

Это означает:
- Конкретные цифры (магнитуда, координаты) **не верифицируются** напрямую.
- Система оценивает, **насколько хорошо утверждение покрыто** корпусом Tier 2.
- При пустой базе любой score будет около 0.5 (неопределённость).

---

## Конфигурация

Все параметры переопределяются через переменные окружения:

| Переменная             | Дефолт                          | Описание                            |
|------------------------|---------------------------------|-------------------------------------|
| `OPENAI_BASE_URL`      | `http://192.168.8.22:9999/v1`   | URL OpenAI-совместимого LLM-сервера |
| `OPENAI_API_KEY`       | `api-key`                       | API-ключ (для локального сервера — любой) |
| `FACT_CHECK_MODEL`     | `gpt-4`                         | Имя модели                          |
| `FACT_CHECK_MAX_TOKENS`| `80000`                         | Максимум токенов ответа             |
| `MILVUS_HOST`          | `localhost`                     | Хост Milvus                         |
| `MILVUS_PORT`          | `19530`                         | Порт Milvus                         |
| `EMBED_MODEL_NAME`     | `sentence-transformers/all-MiniLM-L6-v2` | Модель эмбеддингов      |

В `.env` файле (для dev-окружения, `ENV=development`):

```env
OPENAI_BASE_URL=http://192.168.8.22:9999/v1
OPENAI_API_KEY=api-key
FACT_CHECK_MODEL=gpt-4
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

---

## API-эндпоинт

### `POST /fact-check`

**Описание:** принимает утверждение, запускает RAG fact-checker, возвращает оценку правдивости.

#### Запрос

```http
POST /fact-check
Content-Type: application/json
```

```json
{
  "claim": "A magnitude 7.5 earthquake struck Turkey in February 2023"
}
```

| Поле    | Тип    | Обязателен | Описание                            |
|---------|--------|------------|-------------------------------------|
| `claim` | string | да         | Утверждение для проверки (любой язык) |

#### Ответ `200 OK`

```json
{
  "claim": "A magnitude 7.5 earthquake struck Turkey in February 2023",
  "score": 0.88,
  "reasoning": "Retrieved 9 records with high similarity (distance ≥ 0.70). Sources include usgs.gov (credibility 0.92) and emsc.eu (credibility 0.89). Multiple independent domains confirm earthquake activity in Turkey in early 2023. Score reflects strong coverage in the Tier 2 database.",
  "sources": ["usgs.gov", "emsc.eu", "reuters.com", "bbc.com"]
}
```

| Поле        | Тип          | Описание                                     |
|-------------|--------------|----------------------------------------------|
| `claim`     | string       | Исходное утверждение (echo)                  |
| `score`     | float 0–1    | Оценка правдивости                           |
| `reasoning` | string       | Объяснение агента                            |
| `sources`   | list[string] | Домены документов, повлиявших на оценку      |

#### Интерпретация `score`

| Диапазон  | Значение                                     |
|-----------|----------------------------------------------|
| 0.8 – 1.0 | Хорошо подтверждено несколькими источниками  |
| 0.6 – 0.8 | Умеренно подтверждено                        |
| 0.4 – 0.6 | Неопределённо — мало данных или смешанные сигналы |
| 0.2 – 0.4 | Слабо подтверждено                           |
| 0.0 – 0.2 | Не подтверждено / противоречит базе          |

#### Ответ `500 Internal Server Error`

```json
{
  "detail": "Connection refused: Milvus not available"
}
```

#### Пример cURL

```bash
curl -X POST http://localhost:8000/fact-check \
  -H "Content-Type: application/json" \
  -d '{"claim": "A magnitude 7.5 earthquake struck Turkey in February 2023"}'
```

#### Пример Python (httpx)

```python
import httpx

resp = httpx.post(
    "http://localhost:8000/fact-check",
    json={"claim": "A magnitude 7.5 earthquake struck Turkey in February 2023"},
    timeout=120,
)
data = resp.json()
print(f"Score: {data['score']}")
print(f"Reasoning: {data['reasoning']}")
print(f"Sources: {data['sources']}")
```

---

## Использование напрямую из Python

Без HTTP — вызов модуля напрямую:

```python
from veritatis.agent import fact_check

result = fact_check("A magnitude 7.5 earthquake struck Turkey in February 2023")

print(result.score)      # 0.88
print(result.reasoning)  # "Found 9 records with distance ≥ 0.70..."
print(result.sources)    # ["usgs.gov", "emsc.eu", ...]
```

Только retriever (без LLM):

```python
from veritatis.agent import query_collection

docs = query_collection("earthquake Turkey 2023", top_k=10)
for doc in docs:
    print(doc["domain"], doc["distance"], doc["credibility_score"])
```

---

## Зависимости

Модуль использует следующие пакеты (добавлены в `veritatis/pyproject.toml` и `requirements.txt`):

| Пакет             | Версия     | Роль                                    |
|-------------------|------------|-----------------------------------------|
| `langgraph`       | ≥ 0.2.55   | ReAct-агент, граф вычислений            |
| `langchain-openai`| ≥ 0.1.0    | `ChatOpenAI` с поддержкой `base_url`    |
| `langchain-core`  | ≥ 0.2.0    | `@tool` декоратор, базовые типы         |
| `pymilvus`        | 2.6.3      | Поиск в Milvus (уже был в зависимостях) |
| `sentence-transformers` | 2.2.2 | Генерация эмбеддингов               |
| `pydantic`        | ≥ 2.0      | Structured output (`FactCheckResult`)   |

---

## Связанные файлы

| Файл                                  | Роль                                          |
|---------------------------------------|-----------------------------------------------|
| `veritatis/veritatis/agent.py`        | Весь код агента: retriever, tool, LLM, граф   |
| `veritatis/veritatis/plain_search.py` | `vector_search()` — низкоуровневый поиск      |
| `veritatis/veritatis/embeddings.py`   | `EmbeddingGenerator` — SentenceTransformer    |
| `veritatis/veritatis/vector_stores.py`| Схема коллекций, `ensure_collection_loaded`   |
| `veritatis/api/main.py`               | FastAPI приложение, эндпоинт `POST /fact-check`|
