# VectorConsensus — Векторный Консенсус

## Обзор

`vector_consensus` — модуль для поиска наиболее релевантного вектора из набора **без запроса**.
Вместо классического семантического поиска (запрос → ближайшие векторы) этот модуль сравнивает векторы между собой и определяет, какой из них наиболее **центральный** (максимально похож на всех остальных) и **детализированный** (содержит наибольшее количество информации).

Основное применение — автоматический отбор кандидатов из Tier 1 для продвижения в Tier 2.

## Архитектура

```
vector_consensus.py
├── cosine_similarity()              Косинусное сходство двух векторов
├── compute_pairwise_similarities()  Матрица NxN попарных сходств
├── calculate_centrality_scores()    Централность: среднее сходство с остальными
├── calculate_detail_scores()        Детализированность: нормализованная длина текста
├── fetch_all_vectors()              Загрузка векторов из Milvus + перегенерация эмбеддингов
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

Итоговая оценка каждого вектора складывается из двух независимых факторов:

```
combined_score = centrality_weight × centrality_score
               + detail_weight     × detail_score
```

| Фактор | По умолчанию | Описание |
|--------|-------------|----------|
| `centrality_score` | 60% | Среднее косинусное сходство с остальными векторами коллекции |
| `detail_score` | 40% | Нормализованная длина контента (0.0 — самый короткий, 1.0 — самый длинный) |

### Шаги вычисления

1. Загрузить все записи из коллекции Milvus (без эмбеддингов — Milvus их не возвращает)
2. Перегенерировать эмбеддинги из текста через `sentence-transformers/all-MiniLM-L6-v2`
3. Построить матрицу NxN попарных косинусных сходств
4. Для каждого вектора: `centrality = среднее(сходство со всеми остальными)`
5. Для каждого вектора: `detail = (длина - min) / (max - min)`
6. Вычислить `combined_score` и отсортировать по убыванию

---

## Структура данных

### `VectorAnalysis`

Результат анализа одного вектора.

```python
@dataclass
class VectorAnalysis:
    # Поля из Milvus
    id: str
    content: str
    source_url: str
    credibility_score: float
    ingested_timestamp: int
    supabase_id: str
    embedding: List[float]

    # Вычисленные оценки
    centrality_score: float          # Среднее сходство с остальными (0.0–1.0)
    detail_score: float              # Нормализованная длина текста  (0.0–1.0)
    combined_score: float            # Взвешенная сумма              (0.0–1.0)

    # Метаданные
    avg_similarity_to_others: float  # То же, что centrality_score
    content_length: int              # Длина контента в символах
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
    limit: Optional[int] = None,
    offset: int = 0,
) -> Tuple[VectorAnalysis, List[VectorAnalysis]]
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `collection_name` | `str` | — | Имя коллекции Milvus |
| `centrality_weight` | `float` | `0.6` | Вес центральности (сумма весов должна равняться 1.0) |
| `detail_weight` | `float` | `0.4` | Вес детализированности |
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
) -> Tuple[VectorAnalysis, List[VectorAnalysis]]
```

Каждый элемент `vectors_with_embeddings` должен содержать ключи:
`id`, `content`, `source_url`, `credibility_score`, `ingested_timestamp`, `supabase_id`, `embedding`.

---

### `fetch_all_vectors()`

Загружает записи из Milvus и перегенерирует эмбеддинги.

```python
fetch_all_vectors(
    collection_name: str,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]
```

> **Важно:** Milvus не возвращает векторы при `query()`. Функция автоматически перегенерирует эмбеддинги из поля `content` через ту же модель, что использовалась при индексации — результаты остаются консистентны.

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
calculate_detail_scores(contents: List[str]) -> List[float]
```
Min-max нормализация длин текстов в диапазон `[0.0, 1.0]`. Если все тексты одинаковой длины — возвращает `1.0` для всех.

---

## Примеры использования

### Базовый анализ Tier 1

```python
from veritatis.vector_consensus import find_most_relevant_vector

best, all_ranked = find_most_relevant_vector("veritatis_tier1_lake")

print(f"Лучший вектор: {best.id}")
print(f"  combined_score: {best.combined_score:.3f}")
print(f"  centrality:     {best.centrality_score:.3f}")
print(f"  detail:         {best.detail_score:.3f}")
print(f"  контент: {best.content[:200]}...")
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
        "id": "vec_001",
        "content": "Землетрясение магнитудой 6.2 произошло в регионе...",
        "source_url": "https://example.com/news/1",
        "credibility_score": 0.85,
        "ingested_timestamp": 1700000000,
        "supabase_id": "uuid-001",
        "embedding": [0.1, 0.2, ...],  # 384-мерный вектор
    },
    # ...
]

best, all_ranked = find_best_vector_with_embeddings(vectors)
```

### Вывод топ-5 и продвижение в Tier 2

```python
from veritatis.vector_consensus import find_most_relevant_vector
from veritatis.vector_stores import MilvusRecordStore

best, all_ranked = find_most_relevant_vector("veritatis_tier1_lake")

store = MilvusRecordStore()

print("Топ-5 векторов:")
for i, v in enumerate(all_ranked[:5], 1):
    print(f"  {i}. {v.id}  score={v.combined_score:.3f}  len={v.content_length}")

# Переместить лучший вектор в Tier 2
if best.combined_score >= 0.7:
    store.move_record(
        best.id,
        source_collection="veritatis_tier1_lake",
        target_collection="veritatis_tier2_arena",
    )
    print(f"Вектор {best.id} перемещён в Tier 2")
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

Сложность: **O(N²)** по числу попарных сравнений. При больших коллекциях рекомендуется использовать параметр `limit`.

---

## Граничные случаи

| Ситуация | Поведение |
|----------|-----------|
| Один вектор в коллекции | Возвращается с оценками `1.0` для всех полей |
| Все тексты одинаковой длины | `detail_score = 1.0` для всех; победителя определяет `centrality_score` |
| Пустая коллекция | Выбрасывает `ValueError` |
| `centrality_weight + detail_weight ≠ 1.0` | Выбрасывает `AssertionError` |
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

## CLI-инструмент

Для анализа Tier 1 из командной строки используйте `scripts/analyze_tier1.py`.
Подробнее: [README_ANALYZE_TIER1.md](README_ANALYZE_TIER1.md).

```bash
# Базовый анализ
python scripts/analyze_tier1.py

# Анализ 200 записей с уклоном в центральность
python scripts/analyze_tier1.py --limit 200 --centrality-weight 0.8 --detail-weight 0.2

# Автоматически переместить лучшие векторы в Tier 2
python scripts/analyze_tier1.py --move-to-tier2 --threshold 0.7
```

---

## Зависимости

| Пакет | Версия | Назначение |
|-------|--------|-----------|
| `numpy` | любая | Матричные вычисления |
| `pymilvus` | 2.6.3+ | Работа с коллекциями Milvus |
| `sentence-transformers` | 2.2.2+ | Генерация эмбеддингов |

---

## См. также

- [README_ANALYZE_TIER1.md](README_ANALYZE_TIER1.md) — руководство по CLI-инструменту анализа Tier 1
- `veritatis/vector_stores.py` — управление коллекциями Milvus (Tier 1/2/3)
- `veritatis/embeddings.py` — модель генерации эмбеддингов
- `veritatis/search.py` — поиск с фильтрацией по релевантности (классический запрос → результаты)
