# VectorConsensus — Векторный Консенсус

## Обзор

`vector_consensus` — модуль для поиска наиболее релевантного вектора из набора **без запроса**.
Вместо классического семантического поиска (запрос → ближайшие векторы) этот модуль сравнивает векторы между собой и определяет, какой из них наиболее **центральный** (максимально похож на всех остальных) и **детализированный** (содержит наибольшее количество информации).

Основное применение — автоматический отбор кандидатов из Tier 1 для продвижения в Tier 2.

## Архитектура

```
vector_consensus.py
├── cosine_similarity()              Косинусное сходство двух векторов
├── compute_pairwise_similarities()  Матрица NxN попарных сходств (numpy dot-product)
├── calculate_centrality_scores()    Централность: среднее сходство с остальными (векторизовано)
├── calculate_detail_scores()        Детализированность: длина + лексическое разнообразие
├── fetch_all_vectors()              Загрузка векторов из Milvus (эмбеддинги или перегенерация)
├── find_most_relevant_vector()      Основной API: поиск лучшего вектора в коллекции
└── find_best_vector_with_embeddings() Поиск лучшего вектора из готового списка
```

### Место в тир-системе

```
Tier 1: veritatis_tier1_lake
│  (все входящие данные)
│
└─► find_most_relevant_vector()
        │  Выбирает наиболее центральный и детализированный вектор
        │
        ▼
Tier 2: veritatis_tier2_arena
│  (кандидаты на верификацию)
│
└─► ручная проверка / дальнейший анализ
        │
        ▼
Tier 3: veritatis_tier3_sanctum
   (верифицированные факты)
```

---

## Алгоритм

Итоговая оценка каждого вектора складывается из трёх факторов (третий опциональный):

```
combined_score = centrality_weight   × centrality_score
               + detail_weight       × detail_score
               + credibility_weight  × credibility_score
```

По умолчанию веса 0.6 / 0.4 / 0.0 — полная обратная совместимость с предыдущим поведением.

| Фактор | По умолчанию | Описание |
|--------|-------------|----------|
| `centrality_score` | 60% | Среднее косинусное сходство с остальными векторами коллекции |
| `detail_score` | 40% | Комбинация длины контента и лексического разнообразия (устраняет повторяющийся мусор) |
| `credibility_score` | 0% | Достоверность источника из Milvus (уже нормализована в [0, 1]) |

### Формула `detail_score`

```
raw = 0.7 × normalized_length + 0.3 × lexical_diversity
detail_score = (raw − min_raw) / (max_raw − min_raw)
```

где `lexical_diversity = уникальные_слова / всего_слов`.

### Шаги вычисления

1. Загрузить записи из коллекции Milvus (получаем эмбеддинги и скалярные поля)
2. По списку `iid` запросить в Supabase таблицу `parsed_content` и получить `main_text`
3. Построить матрицу NxN косинусных сходств через numpy dot-product (векторизовано)
4. Для каждого вектора: `centrality = (сумма строки − 1) / (N − 1)`
5. Для каждого вектора: `detail_score` по формуле выше
6. Вычислить `combined_score` и отсортировать по убыванию

---

## Структура данных

### `VectorAnalysis`

Результат анализа одного вектора.

```python
@dataclass
class VectorAnalysis:
    # Поля из Milvus
    iid: str
    embedding: List[float]
    credibility_score: float
    date: int
    domain: str

    # Поля из Supabase
    main_text: str

    # Вычисленные оценки
    centrality_score: float          # Среднее сходство с остальными (0.0–1.0)
    detail_score: float              # Длина + лексическое разнообразие (0.0–1.0)
    combined_score: float            # Взвешенная сумма              (0.0–1.0)

    # Метаданные
    avg_similarity_to_others: float  # То же, что centrality_score
    main_text_length: int           # Длина main_text в символах
```

---

## API Reference

### `find_most_relevant_vector()`

Основная функция. Загружает коллекцию из Milvus и возвращает лучший вектор.

```python
find_most_relevant_vector(
    collection_name: str,
    centrality_weight: float = 0.6,
    detail_weight: float = 0.4,
    credibility_weight: float = 0.0,
    limit: Optional[int] = None,
    offset: int = 0,
) -> Tuple[VectorAnalysis, List[VectorAnalysis]]
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `collection_name` | `str` | — | Имя коллекции Milvus |
| `centrality_weight` | `float` | `0.6` | Вес центральности (все три веса должны давать 1.0) |
| `detail_weight` | `float` | `0.4` | Вес детализированности |
| `credibility_weight` | `float` | `0.0` | Вес достоверности источника |
| `limit` | `int \| None` | `None` | Максимальное количество векторов для анализа |
| `offset` | `int` | `0` | Смещение при выборке |

**Возвращает:** `(best_vector, all_vectors_ranked)` — лучший вектор и полный список, отсортированный по `combined_score` по убыванию.

---

### `find_best_vector_with_embeddings()`

Удобная функция для работы с уже загруженными векторами (без обращения к Milvus).

```python
find_best_vector_with_embeddings(
    vectors_with_embeddings: List[Dict[str, Any]],
    centrality_weight: float = 0.6,
    detail_weight: float = 0.4,
    credibility_weight: float = 0.0,
) -> Tuple[VectorAnalysis, List[VectorAnalysis]]
```

Каждый элемент `vectors_with_embeddings` должен содержать ключи:
`iid`, `main_text`, `credibility_score`, `date`, `domain`, `embedding`.

---

### `fetch_all_vectors()`

Загружает записи из Milvus.

```python
fetch_all_vectors(
    collection_name: str,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]
```

> **Примечание:** В Veritatis текст не хранится в Milvus. Для `detail_score` используется `parsed_content.main_text` из Supabase.

---

### Низкоуровневые функции

```python
cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float
```
Косинусное сходство двух нормализованных L2-векторов. Возвращает значение в диапазоне `[0.0, 1.0]`.

```python
compute_pairwise_similarities(embeddings: List[np.ndarray]) -> np.ndarray
```
Строит симметричную матрицу NxN попарных сходств. Вычисляет только верхний треугольник, нижний заполняет зеркально — `O(N²/2)`.

```python
calculate_centrality_scores(similarity_matrix: np.ndarray) -> List[float]
```
Для каждой строки матрицы вычисляет среднее по всем столбцам, исключая диагональ (self-similarity).

```python
calculate_detail_scores(texts: List[str]) -> List[float]
```
Min-max нормализация длин текстов в диапазон `[0.0, 1.0]`. Если все тексты одинаковой длины — возвращает `1.0` для всех.

---

## Примеры использования

### Базовый анализ Tier 1

```python
from veritatis.vector_consensus import find_most_relevant_vector

best, all_ranked = find_most_relevant_vector("veritatis_tier1_lake")

print(f"Лучший вектор: {best.iid}")
print(f"  combined_score: {best.combined_score:.3f}")
print(f"  centrality:     {best.centrality_score:.3f}")
print(f"  detail:         {best.detail_score:.3f}")
print(f"  main_text: {best.main_text[:200]}...")
```

### Акцент на консенсусе (центральности)

```python
best, all_ranked = find_most_relevant_vector(
    "veritatis_tier1_lake",
    centrality_weight=0.9,
    detail_weight=0.1,
)
```

Используйте, когда важно найти вектор, наиболее близкий ко всем остальным — «типичный» документ группы.

### Акцент на детализированности

```python
best, all_ranked = find_most_relevant_vector(
    "veritatis_tier1_lake",
    centrality_weight=0.2,
    detail_weight=0.8,
)
```

Используйте, когда важен максимально подробный документ независимо от его позиции в кластере.

### Анализ подмножества (первые 100 записей)

```python
best, all_ranked = find_most_relevant_vector(
    "veritatis_tier1_lake",
    limit=100,
)
```

### Работа с готовыми векторами (без Milvus)

```python
from veritatis.vector_consensus import find_best_vector_with_embeddings

vectors = [
    {
        "iid": "vec_001",
        "main_text": "Землетрясение магнитудой 6.2 произошло в регионе...",
        "credibility_score": 0.85,
        "date": 0,
        "domain": "example.com",
        "embedding": [0.1, 0.2, ...],  # 384-мерный вектор
    },
    # ...
]

best, all_ranked = find_best_vector_with_embeddings(vectors)
```

### Вывод топ-5 и продвижение в Tier 2 (через API)

```bash
curl -X POST http://localhost:8000/consensus/analyze \
  -H "Content-Type: application/json" \
  -d '{"top_n": 5, "threshold": 0.7}'
```

### Вывод топ-5 и продвижение в Tier 2 (Python)

```python
from veritatis.vector_consensus import find_most_relevant_vector

best, all_ranked = find_most_relevant_vector("veritatis_tier1_lake")

print("Топ-5 векторов:")
for i, v in enumerate(all_ranked[:5], 1):
    print(f"  {i}. {v.iid}  score={v.combined_score:.3f}  len={v.main_text_length}")
```

---

## Производительность

| Количество векторов | Приблизительное время |
|--------------------|----------------------|
| 10 | < 0.1 сек |
| 100 | ~ 1 сек |
| 500 | ~ 5 сек |
| 1 000 | ~ 10 сек |
| 5 000 | ~ 4–5 мин |

Сложность: **O(N²)** по числу попарных сравнений, реализовано через numpy dot-product — значительно быстрее, чем Python-цикл. При больших коллекциях рекомендуется использовать параметр `limit`.

---

## Граничные случаи

| Ситуация | Поведение |
|----------|-----------|
| Один вектор в коллекции | Возвращается с оценками `1.0` для всех полей |
| Все тексты одинаковой длины | `detail_score = 1.0` для всех; победителя определяет `centrality_score` |
| Пустая коллекция | Выбрасывает `ValueError` |
| `centrality_weight + detail_weight + credibility_weight ≠ 1.0` | Выбрасывает `AssertionError` |
| Все векторы идентичны | `centrality_score ≈ 1.0` для всех; победителя определяет `detail_score` |

---

## Тестирование

```bash
# Unit-тесты (без Milvus)
cd veritatis
pytest tests/test_vector_consensus.py -v

# Интеграционные тесты (требуют запущенный Milvus)
pytest tests/test_vector_consensus.py -v -m integration

# Демо-скрипт с наглядным выводом
python tests/demo_vector_consensus.py
```

---

## API эндпойнт: POST /consensus/analyze

Единственный рабочий способ запустить консенсусный анализ через HTTP.

```http
POST /consensus/analyze
Content-Type: application/json

{
    "limit": 100,
    "top_n": 10,
    "threshold": 0.7,
    "centrality_weight": 0.6,
    "detail_weight": 0.4,
    "credibility_weight": 0.0
}
```

**Параметры тела запроса:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `limit` | `int \| null` | `null` | Максимум векторов из Tier 1 (null = все) |
| `top_n` | `int` | `10` | Сколько векторов включить в ранжированный список |
| `threshold` | `float` | `0.7` | Порог для поля `above_threshold_count` в статистике |
| `centrality_weight` | `float` | `0.6` | Вес центральности |
| `detail_weight` | `float` | `0.4` | Вес детализированности |
| `credibility_weight` | `float` | `0.0` | Вес достоверности источника |

Сумма трёх весов должна равняться 1.0, иначе возвращается HTTP 422.

**Ответ:**

```json
{
    "analyzed_count": 50,
    "best_vector": {
        "iid": "abc123",
        "main_text": "Землетрясение магнитудой 7.8...",
        "credibility_score": 0.85,
        "date": 0,
        "domain": "news.example.com",
        "centrality_score": 0.823,
        "detail_score": 0.756,
        "combined_score": 0.796,
        "main_text_length": 1234
    },
    "moved_to_tier2": true,
    "top_n": [
        {"rank": 1, "iid": "abc123", "combined_score": 0.796, ...},
        ...
    ],
    "stats": {
        "avg_combined_score": 0.65,
        "min_combined_score": 0.32,
        "max_combined_score": 0.796,
        "avg_centrality_score": 0.70,
        "avg_detail_score": 0.55,
        "above_threshold_count": 12,
        "threshold_used": 0.7
    }
}
```

> **Важно:** лучший вектор **всегда** перемещается в Tier 2 независимо от его score.
> `threshold` используется только для поля `above_threshold_count` в статистике.

---

## CLI-инструмент: analyze_tier1.py

Скрипт `scripts/analyze_tier1.py` анализирует векторы из Tier 1 и опционально перемещает лучшие в Tier 2.

### Быстрый старт

```bash
# 1. Запустить Milvus (если ещё не запущен)
cd veritatis && docker-compose up -d

# 2. Базовый анализ (все векторы)
python scripts/analyze_tier1.py

# 3. Анализ с ограничением (быстрее для больших коллекций)
python scripts/analyze_tier1.py --limit 100

# 4. Акцент на консенсус
python scripts/analyze_tier1.py --centrality-weight 0.9 --detail-weight 0.1

# 5. Акцент на детальность
python scripts/analyze_tier1.py --centrality-weight 0.2 --detail-weight 0.8

# 6. Переместить лучшие в Tier 2
python scripts/analyze_tier1.py --move-to-tier2 --threshold 0.7

# 6b. Добавить вес достоверности источника
python scripts/analyze_tier1.py --centrality-weight 0.55 --detail-weight 0.25 --credibility-weight 0.20

# 7. Показать топ-20
python scripts/analyze_tier1.py --top-n 20
```

### Все параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--limit N` | Анализировать первые N векторов | Все |
| `--centrality-weight W` | Вес центральности (0–1) | 0.6 |
| `--detail-weight W` | Вес детальности (0–1) | 0.4 |
| `--credibility-weight W` | Вес достоверности источника (0–1) | 0.0 |
| `--top-n N` | Показать топ-N результатов | 10 |
| `--move-to-tier2` | Переместить лучшие в Tier 2 | false |
| `--threshold T` | Порог для перемещения (0–1) | 0.7 |

**Важно:** `centrality-weight + detail-weight + credibility-weight` должны в сумме давать 1.0.

### Типичные сценарии

```bash
# Быстрая проверка качества данных
python scripts/analyze_tier1.py --limit 50 --top-n 5

# Найти консенсусные новости
python scripts/analyze_tier1.py --centrality-weight 0.9 --detail-weight 0.1 --top-n 5

# Найти самые подробные репортажи
python scripts/analyze_tier1.py --centrality-weight 0.1 --detail-weight 0.9 --top-n 5

# Автоматическая фильтрация в Tier 2
python scripts/analyze_tier1.py --move-to-tier2 --threshold 0.75
```

### Интерпретация результатов

| Метрика | Диапазон | Значение |
|---------|----------|----------|
| `centrality_score` | 0.0–0.3 | Вектор — outlier |
| `centrality_score` | 0.6–1.0 | Высокая согласованность |
| `detail_score` | 0.0 | Самый короткий / однообразный текст |
| `detail_score` | 1.0 | Самый длинный и разнообразный текст |
| `combined_score` | < 0.5 | Не рекомендуется для Tier 2 |
| `combined_score` | ≥ 0.7 | Рекомендуется для Tier 2 |

---

## Зависимости

| Пакет | Версия | Назначение |
|-------|--------|-----------|
| `numpy` | любая | Матричные вычисления |
| `pymilvus` | 2.6.3+ | Работа с коллекциями Milvus |
| `sentence-transformers` | 2.2.2+ | Генерация эмбеддингов |

---

## См. также

- `veritatis/vector_stores.py` — управление коллекциями Milvus (Tier 1/2/3)
- `veritatis/embeddings.py` — модель генерации эмбеддингов
- `veritatis/search.py` — поиск с фильтрацией по релевантности (классический запрос → результаты)
