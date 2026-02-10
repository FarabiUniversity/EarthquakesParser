# Model Evaluation Guide

Руководство по оценке качества моделей для извлечения HTML-схем.

## Быстрый старт

### Вариант 1: Командная строка

```bash
python -m earthquakes_parser.parser.html_schema_judge --evaluate
```

### Вариант 2: Python скрипт

```python
from earthquakes_parser.parser.html_schema_judge import HTMLSchemaJudge

judge = HTMLSchemaJudge()
results = judge.evaluate_models()
```

### Вариант 3: Пример из папки examples

```bash
python examples/evaluate_models_example.py
```

## Требования

1. **Файлы схем**: В папке `data/` должны быть файлы `schema-<model>.json`:
   - `schema-gemma.json`
   - `schema-gpt_oss.json`
   - `schema-llama.json`
   - и т.д.

2. **Файл ссылок**: `data/links.csv` с тестовыми ссылками (до 20 ссылок)

3. **LLM сервер**: Настроенный OpenAI-совместимый сервер для GPT Judge

## Формат входных данных

### schema-<model>.json

Массив объектов с извлеченными схемами:

```json
[
  {
    "url": "https://example.com/article",
    "domain": "example.com",
    "title": "Article Title",
    "schema": {
      "domain": "example.com",
      "main_text_selectors": [".article p", ".content"],
      "date_selector": ".publish-date",
      "is_valid": true
    },
    "is_valid": true
  },
  {
    "url": "https://other.com/page",
    "error": "extraction failed"
  }
]
```

### links.csv

CSV файл с колонкой `link`:

```csv
query,link,title
землетрясение,https://example.com/article,Article Title
```

## Система оценки

### Метрики (взвешенная оценка из 100 баллов)

| Метрика | Вес | Описание |
|---------|-----|----------|
| **Success Rate** | 40% | Процент успешно извлеченных схем |
| **Precision** | 30% | Качество селекторов на реальном HTML |
| **GPT Quality** | 30% | Экспертная оценка GPT по шкале 0-10 |

### Формула

```
Final Score = Success Rate × 40 + Precision × 30 + GPT Quality × 30
```

### Success Rate (40%)

Вычисляется как доля успешных извлечений:

```
Success Rate = (successful extractions / total links)
```

Успешное извлечение = запись с полем `"schema"` (без `"error"`).

### Precision (30%)

Для каждой схемы применяем селекторы к реальному HTML:

1. **Main text** (70%): Извлекается ли текст > 50 символов?
2. **Date** (30%): Находится ли дата публикации?

```
Precision = 0.7 (if text found) + 0.3 (if date found)
```

Средняя precision по всем ссылкам × 30 = баллы за precision.

### GPT Quality (30%)

GPT оценивает каждую схему по расширенным критериям:

**Main Text Selectors (70% от GPT оценки):**
- Точность: нацелены ли на фактический контент статьи?
- Полнота: захватывают ли ВСЕ параграфы, заголовки, цитаты?
- Чистота: избегают ли навигацию, сайдбары, рекламу, комментарии?
- Специфичность: достаточно конкретные, но не слишком хрупкие?
- Существование: элементы действительно присутствуют в DOM?

**Date Selector (30% от GPT оценки):**
- Корректность: указывает на дату публикации/обновления?
- Наличие: элемент присутствует в HTML?
- Точность: избегает захвата неправильных временных меток?

**Факторы качества:**
- **Precision**: сколько нерелевантного контента захвачено?
- **Recall**: сколько релевантного контента пропущено?
- **Robustness**: сломаются ли селекторы при незначительных изменениях HTML?
- **Reusability**: не слишком ли специфичны селекторы (хардкод ID)?

**Шкала оценок:**
- 9-10: Отлично — точные, полные, надежные селекторы
- 7-8: Хорошо — захватывает большую часть контента с незначительным шумом
- 5-6: Приемлемо — работает, но есть значительные проблемы
- 3-4: Плохо — пропускает контент или захватывает слишком много шума
- 1-2: Провал — селекторы не существуют или полностью неверны

**Выходные данные GPT:**
```json
{
  "score": 8.5,
  "confidence": 0.9,
  "reason": "Детальное объяснение с конкретными примерами",
  "main_text_quality": 9.0,
  "date_quality": 7.0,
  "issues": ["Проблема 1", "Проблема 2"],
  "suggestions": "Рекомендации по улучшению"
}
```

Общая оценка GPT (0-10) нормализуется к 0-1, затем × 30 = баллы за GPT Quality.

## Выходные данные

### Консольный вывод

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

### Файл model_evaluation.json

Сохраняется в `data/model_evaluation.json` с детальной информацией:

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
            "reason": "Selectors accurately target main content",
            "main_text_quality": 9.0,
            "date_quality": 8.0,
            "issues": [
              "Hardcoded article ID reduces reusability"
            ],
            "suggestions": "Use more generic selectors like .article-body instead of #article-123"
          }
        }
      ]
    }
  }
}
```

## Настройка LLM сервера

При инициализации можно указать параметры сервера:

```python
judge = HTMLSchemaJudge(
    openai_base_url="http://192.168.8.22:9999/v1",
    openai_api_key="api-key",  # pragma: allowlist secret
    model="gpt-4",
    max_tokens=1000,
)
```

## Интерпретация результатов

### High Score (> 80)
- Модель надежно извлекает схемы
- Селекторы точны и полные
- GPT подтверждает высокое качество
- Main text quality > 8.0, Date quality > 7.0
- Минимум issues

### Medium Score (60-80)
- Приемлемое качество
- Возможны пропуски или неточности
- Требуется анализ деталей
- Проверьте issues и suggestions для улучшения

### Low Score (< 60)
- Много ошибок извлечения
- Селекторы неточные или неполные
- Не рекомендуется для продакшена
- Необходим полный пересмотр подхода

### Анализ GPT Judgment полей

**main_text_quality vs date_quality:**
- Если main_text_quality низкое: проблемы с основным контентом
- Если date_quality низкое: проблемы с датой публикации
- Разница > 3 балла: несбалансированное качество

**Типичные issues:**
- "Hardcoded ID" → Селектор слишком специфичен
- "Misses nested content" → Не захватывает весь контент
- "Includes navigation" → Захватывает посторонние элементы
- "Selector doesn't exist" → Элемент отсутствует в DOM
- "Too broad" → Селектор слишком общий

**Использование suggestions:**
Поле suggestions содержит конкретные рекомендации GPT по улучшению селекторов. Используйте их для итеративного улучшения промптов моделей.

## Советы по улучшению моделей

1. **Низкий Success Rate**: Модель часто не может извлечь схему
   - Проверить промпты
   - Увеличить max_tokens
   - Использовать более мощную модель

2. **Низкий Precision**: Селекторы неточны
   - Добавить примеры в промпт
   - Улучшить инструкции по выбору селекторов
   - Проверить качество HTML в обучающих данных

3. **Низкий GPT Quality**: Селекторы захватывают лишнее
   - Уточнить, что избегать (навигация, реклама)
   - Добавить валидацию селекторов
   - Использовать более специфичные селекторы

## Дополнительная информация

Подробная документация: `docs/HTML_SCHEMA_JUDGE.md`

## Примеры

### Сравнить все модели

```python
from earthquakes_parser.parser.html_schema_judge import HTMLSchemaJudge

judge = HTMLSchemaJudge()
results = judge.evaluate_models()

# Найти лучшую модель
best_model = max(
    results['models'].items(),
    key=lambda x: x[1]['final_score']
)
print(f"Best: {best_model[0]} with {best_model[1]['final_score']:.2f}/100")
```

### Проанализировать детали одной модели

```python
import json

with open('data/model_evaluation_old.json', 'r') as f:
    data = json.load(f)

gemma_details = data['models']['gemma']['details']
for detail in gemma_details:
    if 'error' not in detail:
        print(f"{detail['domain']}: precision={detail['precision']['precision_score']}")
```

### Экспорт в CSV для анализа

```python
import pandas as pd
import json

with open('data/model_evaluation_old.json', 'r') as f:
    data = json.load(f)

rows = []
for model, metrics in data['models'].items():
    rows.append({
        'model': model,
        'success_rate': metrics['success_rate'],
        'precision': metrics['precision_avg'],
        'gpt_quality': metrics['gpt_quality_avg'],
        'final_score': metrics['final_score']
    })

df = pd.DataFrame(rows)
df.to_csv('data/model_comparison.csv', index=False)
print("Saved to model_comparison.csv")
```
