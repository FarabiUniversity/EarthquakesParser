# HTMLSchemaJudge

Модуль для пакетной экстракции схем из HTML-страниц по списку ссылок из CSV-файла.

`earthquakes_parser/parser/html_schema_judge.py`

---

## Что делает

1. Читает ссылки из `data/links.csv`
2. Скачивает HTML каждой страницы через `HTMLDownloader` (requests)
3. Извлекает схему (CSS-селекторы) через GPT, используя `SchemaExtractor._build_prompt`
4. Сохраняет результаты в `data/schema1.json`, `data/schema2.json` и т.д. — каждый запуск создаёт новый файл с следующим номером

---

## Использование

Как модуль:

```python
from earthquakes_parser.parser.html_schema_judge import HTMLSchemaJudge

judge = HTMLSchemaJudge()
output_path = judge.run()  # -> "data/schema1.json"
```

Как скрипт:

```bash
 # экстракция
  python -m earthquakes_parser.parser.html_schema_judge --run

  judge = HTMLSchemaJudge()
  judge.run(compare_after=True)   # включено
  judge.run(compare_after=False)  # выключено (по умолчанию)

  # оценка
  python -m earthquakes_parser.parser.html_schema_judge --evaluate
```

С указанием другого CSV:

```python
judge = HTMLSchemaJudge(csv_path="data/instagram_links.csv")
judge.run()
```

Сравнение без экстракции (GPT нужен, экземпляр класса нужен):

```python
judge = HTMLSchemaJudge()
metrics = judge.compare_all()
```

Экстракция + сравнение подряд:

```python
judge = HTMLSchemaJudge()
judge.run(compare_after=True)  # compare запускается автоматически, если schema файлов >= 2
```

---

## Параметры `__init__`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `openai_base_url` | `http://192.168.8.22:9999/v1` | URL локального LLM-сервера |
| `openai_api_key` | `api-key` | Ключ API |
| `model` | `gpt-4` | Модель |
| `max_tokens` | `80000` | Лимит токенов для промпта (трункация HTML) |
| `csv_path` | `data/links.csv` | Путь к CSV с ссылками |

---

## Формат CSV (`links.csv`)

```
query,link
землетрясение,https://example.com/article
```

Обязательный столбец — `link`. Столбец `query` не используется при извлечении.

---

## Формат выходного файла (`data/schemaX.json`)

Массив объектов — по одному на каждую ссылку из CSV. Успешная экстракция:

```json
{
  "url": "https://example.com/article",
  "domain": "example.com",
  "title": "Заголовок страницы",
  "schema": {
    "domain": "example.com",
    "main_text_selectors": ["article.content p", ".post-body"],
    "date_selector": ".pub-date",
    "is_valid": true
  },
  "is_valid": true
}
```

Если экстракция не удалась:

```json
{
  "url": "https://other.com/page",
  "error": "extraction failed"
}
```

---

## Нумерация файлов

Каждый запуск `run()` автоматически определяет номер:

- `data/` пуста → `schema1.json`
- Уже есть `schema1.json` → создаётся `schema2.json`
- И т.д.

Для нового прогона просто подмените `links.csv` и запустите снова.

---

## Сравнение схем (`compare_all`)

`compare_all()` читает все `schema1.json … schemaN.json`, собирает метрики и печатает отчёт.
Для каждого shared-домена GPT заново скачивает HTML и оценивает точность извлечённых селекторов.
Результат сохраняется в `data/comparison.json`.

### Что считается

| Метрика | Описание |
|---|---|
| `per_run.success_rate` | Доля URL, для которых GPT вернул схему |
| `per_run.valid_rate` | Доля URL, которые GPT отметил как `is_valid: true` (earthquake-related) |
| `shared_domains` | Домены, которые появились в 2+ разных запусках |
| `unique_domains` | Домены, появившиеся только в одном запуске |
| `stability[].consistent` | `true`, если CSS-селекторы для домена одинаковы во всех запусках |
| `judgments[].valid` | GPT-валидация для stable-домнов: верны ли селекторы? |
| `judgments[].best_run` | GPT-выбор лучшего запуска для unstable-домнов |
| `judgments[].confidence` | Уверенность модели (0.0–1.0) |

### Как работает GPT-оценка

- **Stable-домен** (селекторы одинаковы во всех запусках): GPT получает HTML + схему и валидирует — правильно ли селекторы указывают на контент и дату. Возвращает `valid`, `confidence`, `reason`.
- **Unstable-домен** (селекторы различаются между запусками): GPT получает HTML + все варианты схем и выбирает лучший. Возвращает `best_run`, `confidence`, `reason`.

### Пример отчёта

```
============================================================
  Schema comparison   (3 runs)
============================================================

  File             Total   OK   OK%  Valid  Vld%
  ---------------- ----- ---- ----- ----- -----
  schema1.json        31   20   65%    15   48%
  schema2.json        25   18   72%    12   48%
  schema3.json        28   22   79%    16   57%

  Domains:  shared=3  unique=42  total=45
  Stability: 2/3 shared domains have identical selectors

  Domain                                 Runs Stable
  -------------------------------------- ---- ------
  rbc.ru                                    2    YES
  interfax.ru                               2    YES
  gismeteo.ru                               3     NO
      schema1.json       main_text = ['article.content p']
                         date      = .pub-date
      schema3.json       main_text = ['div.news-text p']
                         date      = .date-item

  GPT Judgments
  ------------------------------------------------------------
  rbc.ru                                 OK       conf=0.92
      -> Selectors correctly target the article body and date
  interfax.ru                            OK       conf=0.88
      -> main_text selector matches the news text block
  gismeteo.ru                            best=schema3.json    conf=0.81
      -> schema3 selectors align with current page layout
============================================================
```

- `Stable = YES` + `OK` — GPT подтвердил, что селекторы верны.
- `Stable = YES` + `INVALID` — селекторы одинаковы во всех запусках, но GPT считает их неверными (страница могла измениться).
- `Stable = NO` + `best=schemaX.json` — селекторы различались, GPT выбрал лучший запуск.

---

## Оценка моделей (`evaluate_models`)

Новая функция для сравнения качества разных моделей (Gemma, GPT OSS, Llama и др.) при извлечении схем из HTML.

### Использование

Как скрипт:

```bash
python -m earthquakes_parser.parser.html_schema_judge --evaluate
```

Как модуль:

```python
from earthquakes_parser.parser.html_schema_judge import HTMLSchemaJudge

judge = HTMLSchemaJudge()
results = judge.evaluate_models()
```

### Формат файлов моделей

Модуль ищет файлы вида `data/schema-<model>.json`, например:
- `data/schema-gemma.json`
- `data/schema-gpt_oss.json`
- `data/schema-llama.json`

Формат файлов такой же, как у `schemaX.json` — массив объектов с извлеченными схемами.

### Что оценивается

Для каждой модели вычисляются три метрики с взвешенной оценкой:

| Метрика | Вес | Описание |
|---|---|---|
| **Success Rate** | 40% | Процент успешно извлеченных схем (без ошибок `"error": "extraction failed"`) |
| **Precision** | 30% | Проверка селекторов на реальном HTML: извлекается ли контент (не пустой результат), найдена ли дата |
| **GPT Quality** | 30% | Оценка GPT по качеству селекторов по шкале 0-10 для каждой ссылки |

**Итоговая оценка** из 100 баллов:

```
Final Score = Success Rate × 40 + Precision × 30 + GPT Quality × 30
```

### Процесс оценки

1. Загружаются все файлы `schema-*.json` из папки `data/`
2. Для каждой модели:
   - Вычисляется Success Rate (сколько схем извлечено успешно)
   - Для каждой успешно извлеченной схемы (до 20 ссылок):
     - Скачивается HTML страницы
     - Применяются селекторы модели к HTML
     - Проверяется, извлекается ли контент (Precision)
     - GPT оценивает качество селекторов по шкале 0-10
   - Рассчитываются средние значения
   - Вычисляется итоговая оценка по формуле
3. Результаты сохраняются в `data/model_evaluation.json`
4. Печатается сводная таблица

### Пример отчета

```
======================================================================
  Model Evaluation: 3 models x 20 links
======================================================================

[gemma] Evaluating...
  [1/18] dknews.kz
  [2/18] tengrinews.kz
  ...
  -> Success: 18/20 (90.0%)
  -> Precision: 0.85
  -> GPT Quality: 0.78
  -> Final Score: 82.30/100

[gpt_oss] Evaluating...
  ...
  -> Final Score: 88.50/100

[llama] Evaluating...
  ...
  -> Final Score: 75.20/100

======================================================================
  Model Evaluation Summary
======================================================================

  Model           Success  Precision   GPT Qual    Score
  --------------- -------- ---------- ---------- --------
  gpt_oss            95.0%      92.0%      89.0%    88.50
  gemma              90.0%      85.0%      78.0%    82.30
  llama              80.0%      78.0%      71.0%    75.20

  Winner: gpt_oss with 88.50/100 points

======================================================================
```

### Формат выходного файла (`data/model_evaluation.json`)

```json
{
  "models": {
    "gemma": {
      "total_links": 20,
      "success_count": 18,
      "success_rate": 0.900,
      "precision_avg": 0.850,
      "gpt_quality_avg": 0.780,
      "final_score": 82.30,
      "details": [
        {
          "url": "https://example.com",
          "domain": "example.com",
          "precision": {
            "main_text_found": true,
            "main_text_length": 1250,
            "date_found": true,
            "precision_score": 1.0
          },
          "gpt_judgment": {
            "score": 8.5,
            "confidence": 0.9,
            "reason": "Selectors accurately target main content"
          }
        }
      ]
    }
  }
}
```

### Метрики подробнее

#### Success Rate (40%)
- Вычисляется как: `success_count / total_links`
- `success_count` = количество записей с полем `"schema"` (без ошибок)
- Показывает, насколько надежно модель извлекает схемы

#### Precision (30%)
- Применяет каждый селектор к реальному HTML
- Проверяет:
  - `main_text_selectors` → извлекается ли текст > 50 символов (вес 70%)
  - `date_selector` → находится ли дата (вес 30%)
- Precision Score = 0.7 (если текст найден) + 0.3 (если дата найдена)

#### GPT Quality (30%)
- GPT получает HTML + схему модели
- Оценивает по расширенным критериям:
  - **Main Text (70%)**: Точность, полнота, чистота селекторов
  - **Date (30%)**: Корректность date_selector
  - **Качество**: Precision, Recall, Robustness, Reusability
- Возвращает детальный ответ:
  - `score`: общая оценка 0-10
  - `main_text_quality`: оценка селекторов текста 0-10
  - `date_quality`: оценка селектора даты 0-10
  - `issues`: список найденных проблем
  - `suggestions`: рекомендации по улучшению
  - `reason`: детальное объяснение
  - `confidence`: уверенность 0-1
- Общая оценка нормализуется к 0-1 для расчета финального скора

---

## Зависимости внутри проекта

| Модуль | Роль |
|---|---|
| `SchemaExtractor` | Промпт, вызов GPT, трункация HTML, парсинг ответа |
| `HTMLDownloader` | Скачивание HTML (по умолчанию через `requests`) |
| `PageSchema` | Модель данных для схемы |
