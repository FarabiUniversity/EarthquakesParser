# Vector Consensus Analysis - Документация

## Описание

Модуль `vector_consensus.py` предназначен для поиска **самого релевантного и подробного вектора** из набора векторов путём **сравнения их между собой** (без начального query).

### Задача

Из N векторов в Tier 1 найти **один лучший вектор**, который:
1. **Центральный** - наиболее близок ко всем остальным векторам (консенсус)
2. **Подробный** - содержит самый длинный/детальный текст

## Основные функции

### 1. `find_most_relevant_vector()`

Основная функция для поиска лучшего вектора из коллекции Milvus.

```python
from veritatis.vector_consensus import find_most_relevant_vector

# Найти лучший вектор из Tier 1
best, all_ranked = find_most_relevant_vector(
    collection_name="veritatis_tier1_lake",
    centrality_weight=0.6,  # Вес центральности (60%)
    detail_weight=0.4,      # Вес детальности (40%)
    limit=100               # Проанализировать первые 100 векторов
)

print(f"Лучший вектор: {best.id}")
print(f"  Центральность: {best.centrality_score:.3f}")
print(f"  Детальность: {best.detail_score:.3f}")
print(f"  Общий балл: {best.combined_score:.3f}")
print(f"  Текст: {best.content[:100]}...")
```

**Параметры:**
- `collection_name` - название коллекции Milvus (например, `"veritatis_tier1_lake"`)
- `centrality_weight` - вес центральности (0.0-1.0, по умолчанию 0.6)
- `detail_weight` - вес детальности (0.0-1.0, по умолчанию 0.4)
- `limit` - максимум векторов для анализа (None = все)
- `offset` - пропустить первые N векторов

**Возвращает:**
- `best` - объект `VectorAnalysis` с лучшим вектором
- `all_ranked` - список всех векторов отсортированных по убыванию score

### 2. `find_best_vector_with_embeddings()`

Функция для работы с уже загруженными векторами (без обращения к Milvus).

```python
from veritatis.vector_consensus import find_best_vector_with_embeddings

# Если у вас уже есть векторы с embeddings
vectors = [
    {
        "id": "v1",
        "content": "Текст новости про землетрясение...",
        "source_url": "http://example.com/1",
        "credibility_score": 0.8,
        "ingested_timestamp": 1000,
        "supabase_id": "s1",
        "embedding": [0.1, 0.2, ...],  # 384-мерный вектор
    },
    # ... больше векторов
]

best, all_ranked = find_best_vector_with_embeddings(
    vectors_with_embeddings=vectors,
    centrality_weight=0.6,
    detail_weight=0.4,
)
```

## Структура VectorAnalysis

Результат анализа возвращается в виде объекта `VectorAnalysis`:

```python
@dataclass
class VectorAnalysis:
    # Метаданные из Milvus
    id: str
    content: str
    source_url: str
    credibility_score: float
    ingested_timestamp: int
    supabase_id: str
    embedding: List[float]

    # Вычисленные оценки
    centrality_score: float        # 0.0-1.0: средняя схожесть со всеми
    detail_score: float            # 0.0-1.0: нормализованная длина текста
    combined_score: float          # Взвешенная сумма двух оценок

    # Дополнительная информация
    avg_similarity_to_others: float
    content_length: int
```

## Как работает алгоритм

### Шаг 1: Получение векторов

```python
# Загружаются все векторы из коллекции
# Embeddings регенерируются из content (тот же embedding model)
records = fetch_all_vectors(collection_name, limit, offset)
```

### Шаг 2: Вычисление попарных схожестей

```python
# Косинусное сходство между всеми парами векторов
# Матрица NxN где [i,j] = similarity(vector_i, vector_j)
similarity_matrix = compute_pairwise_similarities(embeddings)
```

**Пример матрицы для 3 векторов:**
```
       v1    v2    v3
v1   [1.0   0.9   0.5]
v2   [0.9   1.0   0.6]
v3   [0.5   0.6   1.0]
```

### Шаг 3: Вычисление центральности

```python
# Центральность = средняя схожесть вектора со всеми ДРУГИМИ
# Высокая центральность = вектор "в центре" кластера
centrality_scores = calculate_centrality_scores(similarity_matrix)
```

**Пример:**
- `v1: (0.9 + 0.5) / 2 = 0.70` - средняя центральность
- `v2: (0.9 + 0.6) / 2 = 0.75` - **самый центральный**
- `v3: (0.5 + 0.6) / 2 = 0.55` - наименее центральный

### Шаг 4: Вычисление детальности

```python
# Детальность = нормализованная длина content
# Самый длинный текст = 1.0, самый короткий = 0.0
detail_scores = calculate_detail_scores(contents)
```

**Пример:**
- `v1: len=50 chars  → score=0.0`  (самый короткий)
- `v2: len=100 chars → score=0.5`
- `v3: len=150 chars → score=1.0`  (самый длинный)

### Шаг 5: Объединение оценок

```python
# Взвешенная сумма (по умолчанию 60% центральность, 40% детальность)
combined_score = centrality_weight * centrality + detail_weight * detail
```

**Пример (0.6 / 0.4):**
- `v1: 0.6*0.70 + 0.4*0.0 = 0.42`
- `v2: 0.6*0.75 + 0.4*0.5 = 0.65` - **победитель!**
- `v3: 0.6*0.55 + 0.4*1.0 = 0.73` - если важнее детальность

## Примеры использования

### Пример 1: Базовый поиск

```python
from veritatis.vector_consensus import find_most_relevant_vector

# Найти лучший вектор с балансом центральности и детальности
best, all_ranked = find_most_relevant_vector(
    "veritatis_tier1_lake",
    centrality_weight=0.6,
    detail_weight=0.4
)

print(f"\n🏆 Лучший вектор: {best.id}")
print(f"📊 Центральность: {best.centrality_score:.3f}")
print(f"📝 Детальность: {best.detail_score:.3f}")
print(f"⭐ Общий балл: {best.combined_score:.3f}")
print(f"📄 Длина текста: {best.content_length} символов")
print(f"🔗 Источник: {best.source_url}")
```

### Пример 2: Акцент на центральности (консенсус)

```python
# Если важнее найти "консенсусный" вектор
best, all_ranked = find_most_relevant_vector(
    "veritatis_tier1_lake",
    centrality_weight=0.9,  # 90% веса на центральность
    detail_weight=0.1,      # 10% на детальность
)

print(f"Самый центральный: {best.id}")
print(f"Средняя схожесть: {best.avg_similarity_to_others:.3f}")
```

### Пример 3: Акцент на детальности

```python
# Если важнее найти самый подробный
best, all_ranked = find_most_relevant_vector(
    "veritatis_tier1_lake",
    centrality_weight=0.2,  # 20% на центральность
    detail_weight=0.8,      # 80% на детальность
)

print(f"Самый подробный: {best.id}")
print(f"Длина: {best.content_length} символов")
```

### Пример 4: Анализ топ-5 векторов

```python
best, all_ranked = find_most_relevant_vector("veritatis_tier1_lake")

print("\n📋 Топ-5 векторов:\n")
for i, vec in enumerate(all_ranked[:5], 1):
    print(f"{i}. {vec.id}")
    print(f"   Центральность: {vec.centrality_score:.3f}")
    print(f"   Детальность: {vec.detail_score:.3f}")
    print(f"   Общий балл: {vec.combined_score:.3f}")
    print(f"   Текст: {vec.content[:80]}...")
    print()
```

### Пример 5: Фильтрация для Tier 2

```python
from veritatis.vector_stores import MilvusRecordStore

# Найти лучший вектор
best, all_ranked = find_most_relevant_vector("veritatis_tier1_lake")

# Переместить в Tier 2 только если score выше порога
THRESHOLD = 0.7

if best.combined_score >= THRESHOLD:
    record_store = MilvusRecordStore()
    success = record_store.move_record(
        collection_from="veritatis_tier1_lake",
        collection_to="veritatis_tier2_arena",
        record_id=best.id
    )

    if success:
        print(f"✅ Вектор {best.id} перемещён в Tier 2")
        print(f"   Score: {best.combined_score:.3f}")
else:
    print(f"❌ Вектор {best.id} не прошёл фильтр (score={best.combined_score:.3f} < {THRESHOLD})")
```

## Тестирование

### Запуск юнит-тестов

```bash
cd /home/mukhit/PycharmProjects/EarthquakesParser/veritatis

# Только юнит-тесты (без Milvus)
pytest tests/test_vector_consensus.py -v -m "not integration"

# Все тесты включая интеграционные (требуется запущенный Milvus)
pytest tests/test_vector_consensus.py -v
```

### Запуск демо-скрипта

```bash
# После запуска docker-compose
docker-compose up -d

# Запустить демо
python tests/demo_vector_consensus.py
```

## Зависимости

- `pymilvus>=2.6.3` - работа с Milvus
- `numpy` - векторные операции
- `sentence-transformers` - embedding модель
- `pytest` - для тестов

## Технические детали

### Косинусное сходство

```python
def cosine_similarity(vec1, vec2):
    # Для нормализованных векторов (L2 norm = 1)
    # косинусное сходство = скалярное произведение
    similarity = np.dot(vec1, vec2)
    return max(0.0, min(1.0, similarity))  # Clamp [0, 1]
```

### Регенерация embeddings

Поскольку Milvus query() не возвращает embedding векторы, мы **регенерируем их из content**:

```python
from veritatis.embeddings import embedding_generator

# Batch embedding для эффективности
embeddings = embedding_generator.embed_batch(contents)
```

Это безопасно, потому что:
- Используется тот же embedding model (`all-MiniLM-L6-v2`)
- Content идентичен оригиналу
- Модель детерминистична

## FAQ

**Q: Почему веса по умолчанию 0.6/0.4?**

A: Это баланс между:
- **Центральность (60%)** - находим "консенсус" среди новостей
- **Детальность (40%)** - предпочитаем более полные описания

Можно настроить под задачу!

**Q: Что если все векторы очень похожи?**

A: Если все векторы имеют схожесть >0.9, то centrality_score будет высоким у всех. В этом случае detail_score станет решающим фактором.

**Q: Что если все векторы очень разные?**

A: Если векторы ортогональны (схожесть ~0), то centrality_score будет низким у всех. Опять же detail_score решит.

**Q: Можно ли использовать другие метрики?**

A: Да! Можно модифицировать:
- `calculate_centrality_scores()` - другая агрегация (медиана, взвешенная)
- `calculate_detail_scores()` - использовать complexity метрики
- Добавить `credibility_score` в комбинацию

**Q: Как быстро это работает?**

A: Сложность O(N²) для попарных схожестей, где N = количество векторов.
- 100 векторов: ~1 секунда
- 1000 векторов: ~10 секунд
- 10000 векторов: ~100 секунд

Для больших коллекций используйте `limit` параметр.

## Связь с Tier-системой

```
┌─────────────────────────────────────────────┐
│  Tier 1: Lacus Factorum (озеро фактов)     │
│  - Все новые вектора из парсера             │
│  - Неверифицированные                       │
└─────────────────┬───────────────────────────┘
                  │
                  │ vector_consensus анализ
                  │ find_most_relevant_vector()
                  │
                  ▼
         ┌────────────────┐
         │ Лучший вектор  │
         │ combined_score │
         └────────┬───────┘
                  │
                  │ Если score >= threshold
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Tier 2: Arena Veritatis (арена истины)     │
│  - Кандидаты для верификации               │
│  - Прошли фильтрацию                        │
└─────────────────────────────────────────────┘
```

## Автор

Создано для проекта Veritatis - RAG система для анализа новостей о землетрясениях.
