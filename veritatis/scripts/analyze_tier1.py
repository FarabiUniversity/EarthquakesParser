#!/usr/bin/env python3
"""
Анализ реальных векторов из Tier 1 коллекции.

Этот скрипт:
1. Подключается к Milvus
2. Загружает векторы из veritatis_tier1_lake
3. Находит самый релевантный/подробный вектор
4. Показывает топ-N результатов
5. Опционально перемещает лучшие векторы в Tier 2

Использование:
    python scripts/analyze_tier1.py
    python scripts/analyze_tier1.py --limit 100
    python scripts/analyze_tier1.py --move-to-tier2 --threshold 0.7
"""

# isort: skip_file
import argparse
import sys
from pathlib import Path

# Добавить veritatis в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from veritatis.vector_consensus import find_most_relevant_vector  # noqa: E402
from veritatis.vector_stores import MilvusRecordStore, ensure_connection  # noqa: E402


def print_separator(char="=", length=80):
    """Печать разделителя."""
    print(char * length)


def print_header(text):
    """Печать заголовка."""
    print_separator()
    print(f"  {text}")
    print_separator()
    print()


def analyze_tier1(
    limit=None,
    centrality_weight=0.6,
    detail_weight=0.4,
    top_n=10,
    move_to_tier2=False,
    threshold=0.7,
):
    """
    Анализ векторов из Tier 1.

    Args:
        limit: Максимум векторов для анализа (None = все)
        centrality_weight: Вес центральности (0-1)
        detail_weight: Вес детальности (0-1)
        top_n: Сколько топ результатов показать
        move_to_tier2: Перемещать ли лучшие в Tier 2
        threshold: Минимальный score для перемещения в Tier 2
    """
    print_header("🔍 АНАЛИЗ ВЕКТОРОВ ИЗ TIER 1")

    # Подключение к Milvus
    print("📡 Подключение к Milvus...")
    try:
        ensure_connection()
        print("✅ Подключено к Milvus\n")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print("\n💡 Убедитесь что Milvus запущен:")
        print("   docker-compose up -d")
        return 1

    # Анализ
    print("🧮 Параметры анализа:")
    print(f"   Центральность: {centrality_weight:.1%}")
    print(f"   Детальность: {detail_weight:.1%}")
    print(f"   Лимит векторов: {limit if limit else 'все'}")
    print()

    print("⏳ Загрузка и анализ векторов...")
    print("   (это может занять время для большого количества векторов)")
    print()

    try:
        best, all_ranked = find_most_relevant_vector(
            collection_name="veritatis_tier1_lake",
            centrality_weight=centrality_weight,
            detail_weight=detail_weight,
            limit=limit,
        )
    except ValueError as e:
        print(f"❌ Ошибка: {e}")
        print("\n💡 Возможно коллекция пуста или не существует")
        print("   Проверьте что данные загружены в Tier 1")
        return 1
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Результаты
    print(f"✅ Анализ завершён! Проанализировано векторов: {len(all_ranked)}\n")

    print_header("🏆 ЛУЧШИЙ ВЕКТОР")
    print(f"ID: {best.id}")
    print(f"Source: {best.source_url}")
    print("\n📊 Оценки:")
    print(f"   Центральность: {best.centrality_score:.3f} (схожесть с другими)")
    print(f"   Детальность:   {best.detail_score:.3f} (относительная длина)")
    print(f"   Общий балл:    {best.combined_score:.3f}")
    print("\n📝 Детали:")
    print(f"   Длина текста: {best.content_length} символов")
    print(f"   Credibility: {best.credibility_score:.2f}")
    print(f"   Timestamp: {best.ingested_timestamp}")
    print("\n📄 Содержание:")
    # Показать первые 300 символов
    content_preview = best.content[:300]
    if len(best.content) > 300:
        content_preview += "..."
    print(f"   {content_preview}")
    print()

    # Топ-N
    print_header(f"📋 ТОП-{min(top_n, len(all_ranked))} ВЕКТОРОВ")
    for i, vec in enumerate(all_ranked[:top_n], 1):
        marker = "🏆" if i == 1 else f"{i}."
        print(f"{marker} {vec.id}")
        print(
            f"    Балл: {vec.combined_score:.3f} "
            f"(центр: {vec.centrality_score:.3f}, "
            f"детал: {vec.detail_score:.3f})"
        )
        print(f"    Длина: {vec.content_length} символов")
        print(f"    Source: {vec.source_url}")

        # Короткий превью текста
        preview = vec.content[:100].replace("\n", " ")
        if len(vec.content) > 100:
            preview += "..."
        print(f"    Текст: {preview}")
        print()

    # Статистика
    print_header("📊 СТАТИСТИКА")
    scores = [v.combined_score for v in all_ranked]
    centralities = [v.centrality_score for v in all_ranked]
    details = [v.detail_score for v in all_ranked]
    lengths = [v.content_length for v in all_ranked]

    print("Общий балл:")
    print(f"   Среднее: {sum(scores) / len(scores):.3f}")
    print(f"   Минимум: {min(scores):.3f}")
    print(f"   Максимум: {max(scores):.3f}")
    print()
    print("Центральность:")
    print(f"   Среднее: {sum(centralities) / len(centralities):.3f}")
    print()
    print("Детальность:")
    print(f"   Среднее: {sum(details) / len(details):.3f}")
    print()
    print("Длина текста:")
    print(f"   Среднее: {sum(lengths) // len(lengths)} символов")
    print(f"   Минимум: {min(lengths)} символов")
    print(f"   Максимум: {max(lengths)} символов")
    print()

    # Перемещение в Tier 2
    if move_to_tier2:
        print_header(f"🔄 ПЕРЕМЕЩЕНИЕ В TIER 2 (порог: {threshold:.2f})")

        record_store = MilvusRecordStore()
        moved_count = 0
        skipped_count = 0

        for vec in all_ranked:
            if vec.combined_score >= threshold:
                try:
                    success = record_store.move_record(
                        collection_from="veritatis_tier1_lake",
                        collection_to="veritatis_tier2_arena",
                        record_id=vec.id,
                    )

                    if success:
                        print(f"✅ {vec.id} → Tier 2 (score={vec.combined_score:.3f})")
                        moved_count += 1
                    else:
                        print(f"⚠️  {vec.id} - не удалось переместить")
                        skipped_count += 1
                except Exception as e:
                    print(f"❌ {vec.id} - ошибка: {e}")
                    skipped_count += 1
            else:
                skipped_count += 1

        print()
        print("📈 Результаты перемещения:")
        print(f"   Перемещено в Tier 2: {moved_count}")
        print(f"   Пропущено (score < {threshold}): {skipped_count}")
        print()

    # Рекомендации
    print_header("💡 РЕКОМЕНДАЦИИ")

    avg_score = sum(scores) / len(scores)
    if avg_score < 0.5:
        print("⚠️  Средний балл низкий (<0.5)")
        print("   Возможно векторы сильно различаются между собой")
        print("   Рекомендация: увеличить вес детальности")
    elif avg_score > 0.8:
        print("✅ Средний балл высокий (>0.8)")
        print("   Векторы хорошо согласованы друг с другом")

    print()

    avg_cent = sum(centralities) / len(centralities)
    if avg_cent < 0.3:
        print("⚠️  Низкая средняя центральность (<0.3)")
        print("   Векторы сильно отличаются друг от друга")
        print("   Рекомендация: возможно нужна дополнительная фильтрация")

    print()
    print("🎯 Для перемещения лучших в Tier 2:")
    print(f"   python scripts/analyze_tier1.py --move-to-tier2 --threshold {threshold}")
    print()

    return 0


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(
        description="Анализ векторов из Tier 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Максимум векторов для анализа (по умолчанию: все)",
    )

    parser.add_argument(
        "--centrality-weight",
        type=float,
        default=0.6,
        help="Вес центральности 0-1 (по умолчанию: 0.6)",
    )

    parser.add_argument(
        "--detail-weight",
        type=float,
        default=0.4,
        help="Вес детальности 0-1 (по умолчанию: 0.4)",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Показать топ-N результатов (по умолчанию: 10)",
    )

    parser.add_argument(
        "--move-to-tier2",
        action="store_true",
        help="Переместить лучшие векторы в Tier 2",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Минимальный score для перемещения в Tier 2 (по умолчанию: 0.7)",
    )

    args = parser.parse_args()

    # Валидация весов
    if not (0 <= args.centrality_weight <= 1):
        print("❌ Ошибка: centrality-weight должен быть между 0 и 1")
        return 1

    if not (0 <= args.detail_weight <= 1):
        print("❌ Ошибка: detail-weight должен быть между 0 и 1")
        return 1

    if abs(args.centrality_weight + args.detail_weight - 1.0) > 1e-6:
        print("❌ Ошибка: сумма весов должна быть равна 1.0")
        return 1

    return analyze_tier1(
        limit=args.limit,
        centrality_weight=args.centrality_weight,
        detail_weight=args.detail_weight,
        top_n=args.top_n,
        move_to_tier2=args.move_to_tier2,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    sys.exit(main())
