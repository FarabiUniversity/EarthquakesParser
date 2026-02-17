# Анализ реальных данных из Tier 1

## 🚀 Быстрый старт

### 1. Запустить Milvus (если ещё не запущен)

```bash
cd /home/mukhit/PycharmProjects/EarthquakesParser/veritatis
docker-compose up -d
```

### 2. Базовый анализ (все векторы)

```bash
/usr/bin/python3 scripts/analyze_tier1.py
```

Это покажет:
- 🏆 Самый релевантный вектор
- 📋 Топ-10 векторов
- 📊 Статистику

### 3. Анализ с ограничением (первые 100 векторов)

```bash
/usr/bin/python3 scripts/analyze_tier1.py --limit 100
```

**Зачем нужен лимит?**
- Быстрее для больших коллекций (100 векторов ~1-2 секунды)
- Сложность O(N²), поэтому 1000 векторов = ~10-20 секунд

### 4. Анализ с акцентом на консенсус

```bash
/usr/bin/python3 scripts/analyze_tier1.py \
  --centrality-weight 0.9 \
  --detail-weight 0.1
```

Найдёт вектор который **ближе всего ко всем остальным** (консенсус).

### 5. Анализ с акцентом на детальность

```bash
/usr/bin/python3 scripts/analyze_tier1.py \
  --centrality-weight 0.2 \
  --detail-weight 0.8
```

Найдёт **самый подробный** вектор (длинный текст).

### 6. Переместить лучшие в Tier 2

```bash
/usr/bin/python3 scripts/analyze_tier1.py \
  --move-to-tier2 \
  --threshold 0.7
```

Все векторы со score ≥ 0.7 будут перемещены в `veritatis_tier2_arena`.

### 7. Показать топ-20

```bash
/usr/bin/python3 scripts/analyze_tier1.py --top-n 20
```

## 📊 Пример вывода

```
================================================================================
  🔍 АНАЛИЗ ВЕКТОРОВ ИЗ TIER 1
================================================================================

📡 Подключение к Milvus...
✅ Подключено к Milvus

🧮 Параметры анализа:
   Центральность: 60.0%
   Детальность: 40.0%
   Лимит векторов: 100

⏳ Загрузка и анализ векторов...

✅ Анализ завершён! Проанализировано векторов: 95

================================================================================
  🏆 ЛУЧШИЙ ВЕКТОР
================================================================================
ID: earthquake_turkey_20230206_001
Source: http://news.example.com/turkey-earthquake

📊 Оценки:
   Центральность: 0.823 (схожесть с другими)
   Детальность:   0.756 (относительная длина)
   Общий балл:    0.796

📝 Детали:
   Длина текста: 1234 символов
   Credibility: 0.85
   Timestamp: 1675728000

📄 Содержание:
   Мощное землетрясение магнитудой 7.8 произошло в Турции...

================================================================================
  📋 ТОП-10 ВЕКТОРОВ
================================================================================
🏆 earthquake_turkey_20230206_001
    Балл: 0.796 (центр: 0.823, детал: 0.756)
    Длина: 1234 символов
    ...
```

## ⚙️ Все параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--limit N` | Анализировать первые N векторов | Все |
| `--centrality-weight W` | Вес центральности (0-1) | 0.6 |
| `--detail-weight W` | Вес детальности (0-1) | 0.4 |
| `--top-n N` | Показать топ-N результатов | 10 |
| `--move-to-tier2` | Переместить лучшие в Tier 2 | false |
| `--threshold T` | Порог для перемещения (0-1) | 0.7 |

**Важно:** `centrality-weight + detail-weight` должны в сумме давать 1.0

## 🎯 Типичные сценарии

### Сценарий 1: Быстрая проверка качества данных

```bash
# Посмотреть на 50 векторов
/usr/bin/python3 scripts/analyze_tier1.py --limit 50 --top-n 5
```

### Сценарий 2: Найти консенсусные новости

```bash
# Акцент на центральность - найти "среднее" мнение
/usr/bin/python3 scripts/analyze_tier1.py \
  --centrality-weight 0.9 \
  --detail-weight 0.1 \
  --top-n 5
```

### Сценарий 3: Найти самые подробные репортажи

```bash
# Акцент на детальность - найти длинные статьи
/usr/bin/python3 scripts/analyze_tier1.py \
  --centrality-weight 0.1 \
  --detail-weight 0.9 \
  --top-n 5
```

### Сценарий 4: Автоматическая фильтрация в Tier 2

```bash
# Найти лучшие и переместить в Tier 2
/usr/bin/python3 scripts/analyze_tier1.py \
  --move-to-tier2 \
  --threshold 0.75
```

## 💡 Как выбрать веса?

### Если важен **консенсус** (согласованность источников):
```bash
--centrality-weight 0.8 --detail-weight 0.2
```
Найдёт новости которые похожи на большинство других.

### Если важна **детальность** (полнота информации):
```bash
--centrality-weight 0.3 --detail-weight 0.7
```
Найдёт самые подробные репортажи.

### Если важен **баланс**:
```bash
--centrality-weight 0.6 --detail-weight 0.4  # По умолчанию
```
Золотая середина между консенсусом и детальностью.

## 📈 Интерпретация результатов

### Centrality Score (Центральность)

- **0.0 - 0.3**: Вектор сильно отличается от других (outlier)
- **0.3 - 0.6**: Средняя схожесть
- **0.6 - 0.8**: Высокая схожесть (консенсус)
- **0.8 - 1.0**: Очень похож на все остальные

### Detail Score (Детальность)

- **0.0**: Самый короткий текст в коллекции
- **0.5**: Средняя длина
- **1.0**: Самый длинный текст в коллекции

### Combined Score (Общий балл)

- **< 0.5**: Низкий балл - не рекомендуется для Tier 2
- **0.5 - 0.7**: Средний - можно рассмотреть
- **0.7 - 0.85**: Хороший - рекомендуется для Tier 2
- **> 0.85**: Отличный - definitely для Tier 2

## 🔧 Troubleshooting

### Ошибка: "No vectors found in collection"

```bash
# Проверить что данные есть в Tier 1
docker exec -it milvus-standalone milvus-cli
> use veritatis_tier1_lake
> show rows
```

Если пусто - загрузите данные:
```bash
python veritatis/ingest_parsed_content.py
```

### Ошибка: Connection refused

```bash
# Запустить Milvus
docker-compose up -d

# Подождать запуска
sleep 30

# Проверить статус
docker-compose ps
```

### Скрипт работает медленно

```bash
# Используйте --limit для ускорения
/usr/bin/python3 scripts/analyze_tier1.py --limit 100

# Или анализируйте по батчам
/usr/bin/python3 scripts/analyze_tier1.py --limit 100 --offset 0
/usr/bin/python3 scripts/analyze_tier1.py --limit 100 --offset 100
```

## 🔄 Workflow интеграция

### Автоматический pipeline

```bash
#!/bin/bash
# pipeline.sh

# 1. Загрузить новые данные в Tier 1
python veritatis/ingest_parsed_content.py

# 2. Проанализировать и переместить лучшие в Tier 2
/usr/bin/python3 scripts/analyze_tier1.py \
  --move-to-tier2 \
  --threshold 0.7 \
  --limit 1000

# 3. Проверить количество в Tier 2
echo "Tier 2 count:"
# ... запрос к Milvus
```

### Cron job (ежедневный анализ)

```bash
# Добавить в crontab
0 2 * * * cd /home/mukhit/PycharmProjects/EarthquakesParser/veritatis && /usr/bin/python3 scripts/analyze_tier1.py --move-to-tier2 --threshold 0.75 >> /var/log/tier1_analysis.log 2>&1
```

## 📚 Связанные скрипты

- `veritatis/ingest_parsed_content.py` - загрузка данных в Tier 1
- `tests/demo_vector_consensus.py` - демонстрация алгоритма
- `tests/test_vector_consensus.py` - тесты

## 🎓 Дополнительная информация

См. полную документацию:
- `tests/README_VECTOR_CONSENSUS.md` - описание алгоритма
- `tests/QUICK_START.md` - быстрый старт
