# Veritatis Fact-Checker

RAG-система для оценки достоверности утверждений об землетрясениях.
Принимает произвольное утверждение, ищет похожие записи в Tier 2 Milvus-коллекции и возвращает оценку правдивости от **0.0** до **1.0**.

---

## Содержание

1. [Архитектура](#архитектура)
2. [Компоненты](#компоненты)
   - [Milvus Tier 2](#milvus-tier-2)
   - [Retriever — query_collection](#retriever--query_collection)
   - [LLM reasoning](#llm-reasoning)
   - [Парсинг ответа](#парсинг-ответа)
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
 query_collection(claim)
       │
       ▼  список metadata-документов
      {iid, credibility_score, date, domain, distance}
       │
       ▼
 ChatOpenAI.invoke(prompt)       gpt-4 via локальный OpenAI-прокси
 system prompt + records as JSON
       │
       ▼
 _extract_final_content()        strip <|channel|>final<|message|> framing
 _parse_json_response()          extract JSON from response text
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

Retrieval выполняется **в Python до вызова LLM** — никаких tool call со стороны модели не требуется.

---

### LLM reasoning

`fact_check()` формирует один prompt, объединяя системный промпт, утверждение и JSON-список извлечённых записей, и вызывает `ChatOpenAI.invoke()` напрямую:

```python
prompt = (
    f"{_SYSTEM_PROMPT}\n\n"
    f"CLAIM: {claim}\n\n"
    f"RETRIEVED RECORDS FROM DATABASE:\n{records_text}\n\n"
    "Based solely on the records above, produce your verdict.\n"
    f"Respond with ONLY a valid JSON object in this exact format:\n{_JSON_SCHEMA}"
)
response = llm.invoke(prompt)
```

**Системный промпт** объясняет модели:
- что поля `distance` и `credibility_score` означают;
- как на их основании выставить `score`;
- что оценка основывается **только** на данных из БД, а не на предобученных знаниях.

**Логика оценки**, которую модель применяет при рассуждении:

| Сигнал | Влияние на score |
|--------|------------------|
| `distance ≥ 0.6` + `credibility_score ≥ 0.7` | сильное подтверждение |
| Много записей из разных `domain` | независимые источники → score выше |
| `distance < 0.3` у всех результатов | тема не покрыта базой → score ≈ 0.5 |
| Единственная запись с низким score | score снижается |

---

### Парсинг ответа

Локальная LLM оборачивает ответ в channel-формат:

```
<|channel|>analysis<|message|>...<|end|>
<|start|>assistant<|channel|>final<|message|>{"score": ..., "reasoning": ..., "sources": [...]}
```

Две вспомогательные функции обрабатывают это:

**`_extract_final_content(text)`** — извлекает контент из канала `final`. Если маркер отсутствует, зачищает все `<|...|>` токены.

**`_parse_json_response(text)`** — ищет JSON-блок в тексте (поддерживает ` ```json ``` ` и голый `{...}`).

---

### Structured output — `FactCheckResult`

```python
class FactCheckResult(BaseModel):
    score: float          # 0.0 (ложь/не подтверждено) → 1.0 (хорошо подтверждено)
    reasoning: str        # пошаговое объяснение оценки
    sources: list[str]    # домены задействованных документов
```

Объект собирается вручную после парсинга JSON из ответа модели:

```python
return FactCheckResult(
    score=float(data["score"]),
    reasoning=str(data["reasoning"]),
    sources=list(data.get("sources", [])),
)
```

---

## Поток данных

```
1. fact_check("Earthquake M7.8 hit Turkey in February 2023")
       │
2.     └─► query_collection(claim, top_k=10)
               │
               └─► embed(claim) → вектор 384d
               └─► Milvus COSINE search veritatis_tier2 → top-10 hits
               └─► return list[dict]  (metadata only, no text)
       │
3.     prompt = system_prompt + claim + JSON(records)
       llm.invoke(prompt)  →  raw LLM response
       │
4.     _extract_final_content(raw)
           strip <|channel|>final<|message|> framing
       _parse_json_response(text)
           find and parse {...} JSON block
       │
5.     return FactCheckResult(
           score=0.92,
           reasoning="All three Tier 2 records have high credibility...",
           sources=["usgs.gov", "reuters.com", "earthquaketrack.com"]
       )
```

---

## Ключевое ограничение

> **Текст статей не хранится в Milvus.**

Модель рассуждает исключительно по **метаданным**:
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
  "score": 0.92,
  "reasoning": "All three Tier 2 records have high credibility (≥0.88) and strong semantic similarity to the claim (distances 0.68–0.84), indicating robust support for a magnitude 7.8 earthquake in Turkey on Feb 6, 2023.",
  "sources": ["usgs.gov", "reuters.com", "earthquaketrack.com"]
}
```

| Поле        | Тип          | Описание                                     |
|-------------|--------------|----------------------------------------------|
| `claim`     | string       | Исходное утверждение (echo)                  |
| `score`     | float 0–1    | Оценка правдивости                           |
| `reasoning` | string       | Объяснение модели                            |
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

print(result.score)      # 0.92
print(result.reasoning)  # "All three Tier 2 records have high credibility..."
print(result.sources)    # ["usgs.gov", "reuters.com", "earthquaketrack.com"]
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

| Пакет             | Версия     | Роль                                    |
|-------------------|------------|-----------------------------------------|
| `langchain-openai`| ≥ 0.1.0    | `ChatOpenAI` с поддержкой `base_url`    |
| `langchain-core`  | ≥ 0.2.0    | `@tool` декоратор, базовые типы         |
| `pymilvus`        | 2.6.3      | Поиск в Milvus (уже был в зависимостях) |
| `sentence-transformers` | 2.2.2 | Генерация эмбеддингов               |
| `pydantic`        | ≥ 2.0      | `FactCheckResult` dataclass             |

> `langgraph` остаётся в `requirements.txt` как транзитивная зависимость, но больше не используется в `agent.py`.

---

## Связанные файлы

| Файл                                  | Роль                                              |
|---------------------------------------|---------------------------------------------------|
| `veritatis/veritatis/agent.py`        | Retriever, LLM вызов, парсинг ответа, `FactCheckResult` |
| `veritatis/veritatis/plain_search.py` | `vector_search()` — низкоуровневый поиск          |
| `veritatis/veritatis/embeddings.py`   | `EmbeddingGenerator` — SentenceTransformer        |
| `veritatis/veritatis/vector_stores.py`| Схема коллекций, `ensure_collection_loaded`       |
| `veritatis/api/main.py`               | FastAPI приложение, эндпоинт `POST /fact-check`   |
