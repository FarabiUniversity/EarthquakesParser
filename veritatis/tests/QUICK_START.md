# Vector Consensus - Быстрый старт

## 🚀 Как проверить что код работает

### Шаг 1: Дождаться установки зависимостей

```bash
# Проверить что pip install завершился
ps aux | grep "pip install" | grep -v grep
# Если пусто - установка завершена ✅
```

### Шаг 2: Запустить демо-скрипт

```bash
cd /home/mukhit/PycharmProjects/EarthquakesParser/veritatis

# Запустить демонстрацию
python tests/demo_vector_consensus.py
```

**Ожидаемый вывод:**
```
🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬
  ДЕМОНСТРАЦИЯ VECTOR CONSENSUS ANALYSIS
🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬

========================================
  1. Вычисление попарных схожестей
========================================
...

✅ ВСЕ ДЕМО ЗАВЕРШЕНЫ УСПЕШНО!
```

### Шаг 3: Запустить юнит-тесты

```bash
# Только юнит-тесты (без Milvus)
pytest tests/test_vector_consensus.py -v -m "not integration"

# Ожидаемый вывод:
# test_identical_vectors PASSED
# test_orthogonal_vectors PASSED
# test_three_vectors PASSED
# ... и т.д.
```

### Шаг 4: Запустить интеграционные тесты (с Milvus)

```bash
# Сначала запустить Milvus через Docker
docker-compose up -d

# Подождать пока Milvus запустится (~30 секунд)
sleep 30

# Запустить все тесты
pytest tests/test_vector_consensus.py -v

# Ожидаемый вывод:
# test_find_best_from_test_collection PASSED ✅
```

## 📝 Основные файлы

| Файл | Описание |
|------|----------|
| `veritatis/vector_consensus.py` | Основной модуль с алгоритмом |
| `tests/test_vector_consensus.py` | Юнит и интеграционные тесты |
| `tests/demo_vector_consensus.py` | Демонстрация работы |
| `tests/README_VECTOR_CONSENSUS.md` | Полная документация |
| `tests/QUICK_START.md` | Этот файл |

## 🔧 Как использовать в коде

### Вариант 1: Анализ коллекции Milvus

```python
from veritatis.vector_consensus import find_most_relevant_vector

# Найти лучший вектор из Tier 1
best, all_ranked = find_most_relevant_vector(
    collection_name="veritatis_tier1_lake",
    centrality_weight=0.6,  # 60% вес центральности
    detail_weight=0.4,      # 40% вес детальности
)

print(f"🏆 Лучший: {best.id}")
print(f"📊 Центральность: {best.centrality_score:.3f}")
print(f"📝 Детальность: {best.detail_score:.3f}")
print(f"⭐ Общий балл: {best.combined_score:.3f}")
```

### Вариант 2: Анализ готовых векторов

```python
from veritatis.vector_consensus import find_best_vector_with_embeddings

vectors = [
    {
        "id": "v1",
        "content": "Текст...",
        "source_url": "http://...",
        "embedding": [0.1, 0.2, ...],  # 384-мерный
        # ... другие поля
    },
    # ... больше векторов
]

best, all_ranked = find_best_vector_with_embeddings(
    vectors,
    centrality_weight=0.6,
    detail_weight=0.4,
)
```

## 🎯 Примеры задач

### Задача 1: Фильтрация для Tier 2

```python
from veritatis.vector_consensus import find_most_relevant_vector
from veritatis.vector_stores import MilvusRecordStore

# Анализируем Tier 1
best, all_ranked = find_most_relevant_vector("veritatis_tier1_lake")

# Переносим только лучшие (score >= 0.7)
THRESHOLD = 0.7
record_store = MilvusRecordStore()

for vec in all_ranked:
    if vec.combined_score >= THRESHOLD:
        record_store.move_records(
            "veritatis_tier1_lake",
            "veritatis_tier2_arena",
            vec.id
        )
        print(f"✅ {vec.id} → Tier 2 (score={vec.combined_score:.3f})")
```

### Задача 2: Акцент на консенсус

```python
# Если важнее найти "консенсусный" вектор (среднее мнение)
best, _ = find_most_relevant_vector(
    "veritatis_tier1_lake",
    centrality_weight=0.9,  # 90% центральность
    detail_weight=0.1,
)

print(f"Консенсус: {best.id}")
print(f"Средняя схожесть: {best.avg_similarity_to_others:.3f}")
```

### Задача 3: Акцент на детальность

```python
# Если важнее найти самый подробный
best, _ = find_most_relevant_vector(
    "veritatis_tier1_lake",
    centrality_weight=0.2,
    detail_weight=0.8,  # 80% детальность
)

print(f"Самый подробный: {best.id}")
print(f"Длина: {best.content_length} символов")
```

## 🐛 Отладка

### Проблема: ModuleNotFoundError

```bash
# Убедиться что зависимости установлены
pip list | grep pymilvus
pip list | grep numpy
pip list | grep sentence-transformers

# Если нет - установить
pip install -r requirements.txt
```

### Проблема: Milvus connection error

```bash
# Проверить что Milvus запущен
docker ps | grep milvus

# Если нет - запустить
docker-compose up -d

# Проверить логи
docker-compose logs milvus
```

### Проблема: Тесты падают

```bash
# Запустить с подробным выводом
pytest tests/test_vector_consensus.py -vv -s

# Запустить только один тест
pytest tests/test_vector_consensus.py::TestCosineSimilarity::test_identical_vectors -v
```

## 📚 Дополнительная информация

- **Полная документация**: `tests/README_VECTOR_CONSENSUS.md`
- **Исходный код**: `veritatis/vector_consensus.py`
- **Тесты**: `tests/test_vector_consensus.py`

## ✅ Чеклист проверки

- [ ] Установлены зависимости (`pip install -r requirements.txt`)
- [ ] Демо работает (`python tests/demo_vector_consensus.py`)
- [ ] Юнит-тесты проходят (`pytest -m "not integration"`)
- [ ] Milvus запущен (`docker-compose up -d`)
- [ ] Интеграционные тесты проходят (`pytest`)
- [ ] Код работает с реальными данными

## 🎓 Обучение

1. Сначала прочитать: `tests/README_VECTOR_CONSENSUS.md`
2. Запустить демо: `python tests/demo_vector_consensus.py`
3. Посмотреть тесты: `tests/test_vector_consensus.py`
4. Попробовать на реальных данных

Удачи! 🚀
