# Credibility Score Computation — Вычисление Достоверности

## Обзор

Модуль `credibility` вычисляет оценку достоверности (**credibility score**) для всех записей в Tier 1 на основе принципа **консенсуса**: чем больше других похожих статей говорят то же самое, тем выше достоверность этой записи.

**Ключевая идея**: если несколько независимых источников сообщают о похожем событии, это повышает вероятность того, что информация правдива.

## Архитектура

```
credibility.py
├── compute_credibility_scores()     Главная функция: вычисление credibility для всех записей
├── _get_record_embedding()          Получение эмбеддинга для конкретной записи
└── (использует существующие модули):
    ├── SimilarityDetector           Поиск групп похожих записей
    ├── find_best_vector_with_embeddings  Ранжирование по централизованности
    └── MilvusRecordStore.update_credibility_scores  Обновление Milvus
```

### Зависимости модулей

```
similarity.py (PR #47)
│  Находит группы похожих статей через косинусное расстояние
│
└─► SimilarityGroup: {anchor_iid, similar_records[]}
        │
        ▼
vector_consensus.py (PR #49)
│  Находит "центральный" вектор в группе (ближайший к остальным)
│
└─► VectorAnalysis: {iid, combined_score, ...}
        │
        ▼
credibility.py (текущая реализация)
│  Комбинирует оба модуля: группировка + ранжирование
│
└─► Обновляет credibility_score в Milvus
```

---

## Алгоритм

### Шаг 1: Группировка похожих статей

Использует `SimilarityDetector` для поиска групп статей с высоким косинусным сходством:

```python
detector = SimilarityDetector(
    collection_name="veritatis_tier1_lake",
    similarity_threshold=0.85,  # Минимальное сходство
    min_group_size=2,           # Минимум 2 записи в группе
)
groups = detector.find_similar_groups()
```

**Пример группы**:
```
Group 1:
  - iid1: "Землетрясение магнитудой 5.2 в Алматы"
  - iid2: "В Алматы произошло землетрясение силой 5.2 балла"
  - iid3: "Сейсмическая активность в Алматы: магнитуда 5.2"
  → Все три статьи говорят об одном событии
```

### Шаг 2: Ранжирование внутри группы

Для каждой группы применяется **векторный консенсус** (`find_best_vector_with_embeddings`):

```python
best, all_ranked = find_best_vector_with_embeddings(
    vectors_with_embeddings,
    centrality_weight=0.6,  # Насколько вектор близок к остальным
    detail_weight=0.4,      # Качество текста (длина + разнообразие)
)
```

**Централизованность (centrality)**: среднее косинусное сходство с другими векторами в группе.

**Детализация (detail)**: комбинация длины текста и лексического разнообразия.

**Combined score** = `0.6 × centrality + 0.4 × detail`

**Пример результата**:
```
iid1: combined_score = 0.92  (самая центральная статья)
iid2: combined_score = 0.85  (чуть менее центральная)
iid3: combined_score = 0.78  (периферийная)
```

### Шаг 3: Присвоение credibility_score

Для каждой записи в группе:
```
credibility_score = combined_score из векторного консенсуса
```

Для записей **без группы** (нет похожих статей):
```
credibility_score = 0.5  (нейтральное значение)
```

**Интерпретация**:
- **0.7 - 1.0**: Высокая достоверность (много подтверждающих источников)
- **0.5**: Нейтральная (нет ни подтверждений ни опровержений)
- **0.0 - 0.5**: Низкая достоверность (мало поддержки или противоречивая информация)

### Шаг 4: Обновление Milvus

Все вычисленные значения сохраняются в Milvus:

```python
store.update_credibility_scores(
    collection_name,
    updates=[
        {"iid": "uuid1", "credibility_score": 0.92},
        {"iid": "uuid2", "credibility_score": 0.85},
        ...
    ]
)
```

---

## Использование

### CLI Script (Рекомендуется для автоматизации)

Самый быстрый и простой способ для запуска из командной строки или cron jobs:

```bash
# Базовое использование
python scripts/compute_credibility.py

# С настройками
python scripts/compute_credibility.py \
  --threshold 0.85 \
  --min-group-size 2 \
  --verbose
```

**Преимущества**:
- ✅ Прямой вызов без HTTP overhead
- ✅ Удобно для cron/scheduled tasks
- ✅ Детальный вывод в консоль
- ✅ Легко интегрируется в CI/CD

**См. также**: [scripts/README.md](../scripts/README.md#compute_credibilitypy)

---

## API

### REST Endpoint (для веб-интеграции)

**POST** `/credibility/compute`

Вычисляет и обновляет credibility_score для всех записей в Tier 1.

#### Параметры

```json
{
  "similarity_threshold": 0.85,
  "min_group_size": 2,
  "centrality_weight": 0.6,
  "detail_weight": 0.4,
  "neutral_score": 0.5,
  "collection": "veritatis_tier1_lake"
}
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `similarity_threshold` | float | 0.85 | Минимальное косинусное сходство (0-1) для группировки записей |
| `min_group_size` | int | 2 | Минимальное количество записей для формирования группы |
| `centrality_weight` | float | 0.6 | Вес централизованности (должна сумма с detail_weight = 1.0) |
| `detail_weight` | float | 0.4 | Вес детализации (качество текста) |
| `neutral_score` | float | 0.5 | Оценка для записей без группы |
| `collection` | string | "veritatis_tier1_lake" | Имя коллекции Milvus |

#### Ответ

```json
{
  "status": "success",
  "message": "Updated credibility scores for 156 records",
  "stats": {
    "total_records": 156,
    "groups_found": 23,
    "records_in_groups": 89,
    "records_without_groups": 67,
    "updated_count": 156,
    "parameters": {
      "similarity_threshold": 0.85,
      "min_group_size": 2,
      "centrality_weight": 0.6,
      "detail_weight": 0.4,
      "neutral_score": 0.5
    }
  }
}
```

#### Пример запроса

```bash
curl -X POST http://localhost:8000/credibility/compute \
  -H "Content-Type: application/json" \
  -d '{
    "similarity_threshold": 0.85,
    "min_group_size": 2
  }'
```

---

### Python API

#### Функция: `compute_credibility_scores()`

```python
from veritatis.credibility import compute_credibility_scores

stats = compute_credibility_scores(
    collection_name="veritatis_tier1_lake",
    similarity_threshold=0.85,
    min_group_size=2,
    centrality_weight=0.6,
    detail_weight=0.4,
    neutral_score=0.5,
)

print(f"Обновлено записей: {stats['updated_count']}")
print(f"Найдено групп: {stats['groups_found']}")
print(f"Записей в группах: {stats['records_in_groups']}")
print(f"Записей без групп: {stats['records_without_groups']}")
```

**Возвращаемое значение**:

```python
{
    "total_records": int,           # Всего записей в коллекции
    "groups_found": int,            # Количество найденных групп
    "records_in_groups": int,       # Записей в группах
    "records_without_groups": int,  # Записей-одиночек
    "updated_count": int,           # Обновлённых записей
    "parameters": dict,             # Использованные параметры
}
```

---

## Примеры использования

### Случай 1: Новости о землетрясении

**Входные данные** (100 статей в Tier 1):
- 40 статей о землетрясении в Алматы (5.2 магнитуды)
- 25 статей о наводнении в Актау
- 15 статей о погоде в Астане
- 20 несвязанных статей (разные темы)

**Результат после вычисления**:

```
Группа 1 (Алматы землетрясение): 40 статей
  - iid_a1: credibility_score = 0.94 (самая центральная)
  - iid_a2: credibility_score = 0.91
  - ...
  - iid_a40: credibility_score = 0.76

Группа 2 (Актау наводнение): 25 статей
  - iid_b1: credibility_score = 0.88
  - ...

Группа 3 (Астана погода): 15 статей
  - iid_c1: credibility_score = 0.82
  - ...

Без группы: 20 статей
  - iid_x1: credibility_score = 0.5 (нейтрально)
  - iid_x2: credibility_score = 0.5
  - ...
```

**Интерпретация**:
- Землетрясение в Алматы имеет самую высокую достоверность (40 источников)
- Одиночные статьи получают нейтральную оценку (недостаточно подтверждений)

---

### Случай 2: Фейковая новость

**Входные данные**:
- 1 статья о "метеорите упавшем на Байконур"
- 99 других статей о других событиях

**Результат**:
```
iid_fake: credibility_score = 0.5
```

**Почему 0.5?** Нет других похожих источников, поэтому нельзя ни подтвердить ни опровергнуть. Нейтральная оценка говорит: "требуется дополнительная проверка".

---

### Случай 3: Противоречивые источники

**Входные данные**:
- 10 статей: "Землетрясение магнитудой 5.2"
- 3 статьи: "Землетрясение магнитудой 4.8"

**Результат**:
```
Группа 1 (5.2): 10 статей
  - credibility_score: 0.85-0.92 (высокая, больше источников)

Группа 2 (4.8): 3 статьи
  - credibility_score: 0.72-0.78 (ниже, меньше источников)
```

**Интерпретация**: Версия "5.2" более достоверна из-за большего количества подтверждений.

---

## Технические детали

### Схема Milvus

```python
veritatis_tier1_lake:
  - iid: VARCHAR (primary key)        # UUID из parsed_content.id
  - embedding: FLOAT_VECTOR(384)      # Эмбеддинг текста
  - credibility_score: FLOAT          # 0.0 - 1.0
  - date: INT64                       # Unix timestamp
  - domain: VARCHAR                   # Домен источника
```

**Важно**:
- `content` (текст статьи) НЕ хранится в Milvus
- Текст запрашивается из Supabase `parsed_content.main_text` по `iid`

### Схема Supabase

```sql
parsed_content:
  - id: UUID (primary key)           # = iid в Milvus
  - main_text: TEXT                  # Полный текст статьи
  - search_result_id: UUID
  - parsed_at: TIMESTAMPTZ
  ...
```

### Поток данных

```
1. Milvus: Получить все iid + embeddings
           ↓
2. Milvus: Найти группы похожих через косинусное расстояние
           ↓
3. Supabase: Запросить main_text для всех iid
           ↓
4. Vector Consensus: Ранжировать записи в каждой группе
           ↓
5. Milvus: Обновить credibility_score
```

---

## Настройка параметров

### similarity_threshold (Порог сходства)

Влияет на размер групп:

| Значение | Поведение |
|----------|-----------|
| **0.95** | Очень строгое сходство → много маленьких групп или одиночек |
| **0.85** | Высокое сходство → средние группы (рекомендуемое) |
| **0.70** | Умеренное сходство → большие группы, возможны ложные совпадения |

**Рекомендация**: Начать с `0.85`, затем анализировать результаты.

### min_group_size (Минимальный размер группы)

| Значение | Поведение |
|----------|-----------|
| **2** | Группы из 2+ записей (рекомендуемое) |
| **3** | Только группы из 3+ записей (строже) |
| **5** | Только крупные кластеры (очень строго) |

**Рекомендация**: `2` для максимального покрытия.

### centrality_weight / detail_weight

Контролируют баланс между "консенсусом" и "качеством текста":

| Соотношение | Приоритет |
|-------------|-----------|
| **0.6 / 0.4** | Больше важен консенсус (рекомендуемое) |
| **0.7 / 0.3** | Ещё больше консенсус |
| **0.5 / 0.5** | Равный баланс |
| **0.4 / 0.6** | Приоритет качеству текста |

**Рекомендация**: `0.6 / 0.4` — консенсус важнее деталей.

### neutral_score

Оценка для одиночных записей:

| Значение | Интерпретация |
|----------|---------------|
| **0.5** | Нейтрально — "неизвестно" (рекомендуемое) |
| **0.3** | Подозрительно — "по умолчанию недоверие" |
| **0.7** | Оптимистично — "по умолчанию доверие" |

**Рекомендация**: `0.5` — честное "не знаем".

---

## Мониторинг и отладка

### Логирование

Модуль использует стандартный Python logging:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

**Логи при выполнении**:
```
INFO - Starting credibility score computation for 'veritatis_tier1_lake'
INFO - Found 23 similarity groups
INFO - Processing group 1/23: anchor=abc123..., size=15
INFO - Group 1 processed: 15 records ranked (best score: 0.923)
INFO - Found 67 records without groups (assigning neutral score 0.5)
INFO - Updating credibility scores for 156 records...
INFO - Successfully updated 156 records
INFO - Credibility computation complete: 156 total, 89 in groups, 67 without groups
```

### Проверка результатов

```python
from pymilvus import Collection
from veritatis.vector_stores import ensure_collection_loaded

ensure_collection_loaded("veritatis_tier1_lake")
collection = Collection("veritatis_tier1_lake")

# Получить распределение scores
results = collection.query(
    expr="credibility_score >= 0",
    output_fields=["iid", "credibility_score"],
    limit=1000
)

scores = [r["credibility_score"] for r in results]
print(f"Среднее: {sum(scores) / len(scores):.3f}")
print(f"Мин: {min(scores):.3f}")
print(f"Макс: {max(scores):.3f}")

# Записи с высокой достоверностью
high_cred = [r for r in results if r["credibility_score"] > 0.8]
print(f"Записей с credibility > 0.8: {len(high_cred)}")
```

---

## Тестирование

### Unit-тесты

```bash
cd veritatis
source .venv/bin/activate
python -m pytest tests/test_credibility.py -v
```

**Покрытие**:
- ✅ Одна группа + одиночки
- ✅ Несколько групп
- ✅ Нет групп (все одиночки)

### Интеграционное тестирование

```python
# Запустить на реальных данных (малая выборка)
from veritatis.credibility import compute_credibility_scores

stats = compute_credibility_scores(
    collection_name="veritatis_tier1_lake",
    similarity_threshold=0.90,  # Строгий порог для теста
    min_group_size=3,           # Только крупные группы
)

print(stats)
```

---

## Оптимизация производительности

### Для больших коллекций (>10,000 записей)

1. **Batch processing**: Обрабатывать по частям
2. **Индексирование**: Убедиться что Milvus индекс создан
3. **Параллелизация**: Обрабатывать группы параллельно

### Пример батчинга

```python
# TODO: Реализовать в будущем
# Обработать первые 1000, затем следующие 1000...
```

---

## Частые вопросы (FAQ)

### Q: Почему credibility_score = 0.5 для одиночек?

**A**: `0.5` — нейтральное значение. Это означает "недостаточно информации для оценки", а не "недостоверно". Отсутствие подтверждений не означает ложность.

### Q: Что если две группы противоречат друг другу?

**A**: Каждая группа получает свои оценки независимо. Более крупная группа будет иметь в среднем более высокие оценки из-за большего числа подтверждений.

### Q: Как часто нужно пересчитывать credibility?

**A**:
- **После каждой массовой загрузки** новых данных
- **Периодически** (например, раз в день) при постоянном потоке
- **По требованию** для критических обновлений

### Q: Можно ли вручную изменить credibility_score?

**A**: Да, через endpoint `/update_credibility`:

```bash
curl -X POST http://localhost:8000/update_credibility \
  -H "Content-Type: application/json" \
  -d '{
    "updates": [
      {"iid": "uuid-here", "credibility_score": 0.95}
    ],
    "tier": 1
  }'
```

---

## Дорожная карта (будущие улучшения)

- [ ] Автоматический пересчёт при новых загрузках
- [ ] Batch processing для коллекций >100k записей
- [ ] Кэширование результатов групп
- [ ] Метрики и дашборд для мониторинга распределения scores
- [ ] Интеграция с веб-интерфейсом для визуализации групп
- [ ] Экспериментальные веса (ML-оптимизация)

---

## Связанная документация

- [VECTOR_CONSENSUS.md](./VECTOR_CONSENSUS.md) — Векторный консенсус (основа для ранжирования)
- [SUPABASE_ARCHITECTURE.md](./SUPABASE_ARCHITECTURE.md) — Архитектура БД
- Similarity Detection (PR #47) — Поиск похожих записей

---

## Авторы и история изменений

**Версия 1.0** (2026-04-05):
- Первая реализация credibility score computation
- REST API endpoint `/credibility/compute`
- Unit тесты с покрытием основных сценариев
- Интеграция с SimilarityDetector и VectorConsensus

---

## Лицензия

См. основной файл LICENSE проекта.
