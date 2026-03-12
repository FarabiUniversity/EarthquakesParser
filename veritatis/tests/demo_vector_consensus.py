#!/usr/bin/env python3
"""
Демонстрация работы vector_consensus модуля.

Этот скрипт показывает как найти самый релевантный вектор
из набора векторов путём сравнения их между собой.

Использование:
    python tests/demo_vector_consensus.py
"""

# isort: skip_file
import sys
from pathlib import Path

# Добавить veritatis в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
from veritatis.vector_consensus import (  # noqa: E402
    calculate_centrality_scores,
    calculate_detail_scores,
    compute_pairwise_similarities,
    find_best_vector_with_embeddings,
)


def print_section(title: str):
    """Печать заголовка секции."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_pairwise_similarities():
    """Демо: вычисление попарных схожестей."""
    print_section("1. Вычисление попарных схожестей (cosine similarity)")

    # Создаём 3 вектора (упрощённые 3D для наглядности)
    vec1 = np.array([1.0, 0.0, 0.0])  # По оси X
    vec2 = np.array([0.9, 0.436, 0.0])  # Близко к vec1
    vec3 = np.array([0.0, 1.0, 0.0])  # По оси Y (ортогонален vec1)

    embeddings = [vec1, vec2, vec3]

    print("Векторы:")
    print(f"  vec1: {vec1}")
    print(f"  vec2: {vec2} (похож на vec1)")
    print(f"  vec3: {vec3} (ортогонален vec1)")

    # Вычисляем матрицу схожестей
    similarity_matrix = compute_pairwise_similarities(embeddings)

    print("\nМатрица попарных схожестей:")
    print("          vec1   vec2   vec3")
    for i, vec_name in enumerate(["vec1", "vec2", "vec3"]):
        row = f"  {vec_name}  "
        for j in range(3):
            row += f" {similarity_matrix[i, j]:.3f} "
        print(row)

    print("\nИнтерпретация:")
    print(f"  - vec1 и vec2 очень похожи: {similarity_matrix[0, 1]:.3f}")
    print(f"  - vec1 и vec3 ортогональны: {similarity_matrix[0, 2]:.3f}")
    print(f"  - vec2 и vec3 средняя схожесть: {similarity_matrix[1, 2]:.3f}")


def demo_centrality_scores():
    """Демо: вычисление центральности."""
    print_section("2. Вычисление центральности (кто ближе ко всем)")

    # Создаём сценарий: vec2 в центре
    similarity_matrix = np.array(
        [
            [1.0, 0.9, 0.5],  # vec1: близок к vec2, далёк от vec3
            [0.9, 1.0, 0.6],  # vec2: средне близок ко всем (центральный!)
            [0.5, 0.6, 1.0],  # vec3: далёк от vec1, близок к vec2
        ]
    )

    print("Матрица схожестей:")
    print("          vec1   vec2   vec3")
    for i, vec_name in enumerate(["vec1", "vec2", "vec3"]):
        row = f"  {vec_name}  "
        for j in range(3):
            row += f" {similarity_matrix[i, j]:.3f} "
        print(row)

    centrality_scores = calculate_centrality_scores(similarity_matrix)

    print("\nЦентральность (средняя схожесть с другими):")
    for i, vec_name in enumerate(["vec1", "vec2", "vec3"]):
        print(f"  {vec_name}: {centrality_scores[i]:.3f}")

    best_idx = np.argmax(centrality_scores)
    print(
        f"\n✅ Самый центральный: vec{best_idx + 1} (score={centrality_scores[best_idx]:.3f})"  # noqa: E501
    )


def demo_detail_scores():
    """Демо: вычисление детальности."""
    print_section("3. Вычисление детальности (по длине текста)")

    contents = [
        "Землетрясение в Турции.",  # Короткий
        "Сильное землетрясение магнитудой 7.8 произошло в Турции.",  # Средний
        "Мощное землетрясение магнитудой 7.8 по шкале Рихтера произошло в Турции в понедельник утром, "  # noqa: E501
        "вызвав массовые разрушения в нескольких провинциях. Тысячи зданий обрушились, спасательные "  # noqa: E501
        "службы работают круглосуточно.",  # Длинный
    ]

    print("Тексты:")
    for i, content in enumerate(contents, 1):
        print(f"\n  Текст {i} ({len(content)} символов):")
        print(f"    {content[:80]}{'...' if len(content) > 80 else ''}")

    detail_scores = calculate_detail_scores(contents)

    print("\n\nОценки детальности (нормализованная длина):")
    for i, (content, score) in enumerate(zip(contents, detail_scores), 1):
        print(f"  Текст {i}: {score:.3f} ({len(content)} символов)")

    best_idx = np.argmax(detail_scores)
    print(
        f"\n✅ Самый подробный: Текст {best_idx + 1} (score={detail_scores[best_idx]:.3f})"  # noqa: E501
    )


def demo_combined_analysis():
    """Демо: полный анализ с реальными данными."""
    print_section("4. Полный анализ: поиск лучшего вектора")

    # Создаём тестовые векторы с разными характеристиками
    print("Создаём 4 тестовых вектора:\n")

    # vec1: центральный, средний по детальности
    vec1_emb = np.random.randn(384)
    vec1_emb = vec1_emb / np.linalg.norm(vec1_emb)

    # vec2: похож на vec1 (центральный), короткий
    vec2_emb = vec1_emb + np.random.randn(384) * 0.1
    vec2_emb = vec2_emb / np.linalg.norm(vec2_emb)

    # vec3: похож на vec1 (центральный), очень длинный
    vec3_emb = vec1_emb + np.random.randn(384) * 0.15
    vec3_emb = vec3_emb / np.linalg.norm(vec3_emb)

    # vec4: сильно отличается (outlier), короткий
    vec4_emb = np.random.randn(384)
    vec4_emb = vec4_emb / np.linalg.norm(vec4_emb)

    vectors = [
        {
            "id": "news_1",
            "content": "Землетрясение в Турции магнитудой 7.8 баллов.",
            "source_url": "http://news1.com/earthquake",
            "credibility_score": 0.8,
            "ingested_timestamp": 1000,
            "supabase_id": "s1",
            "embedding": vec1_emb.tolist(),
        },
        {
            "id": "news_2",
            "content": "Турция 7.8.",  # Короткий
            "source_url": "http://news2.com/quake",
            "credibility_score": 0.7,
            "ingested_timestamp": 2000,
            "supabase_id": "s2",
            "embedding": vec2_emb.tolist(),
        },
        {
            "id": "news_3",
            "content": (
                "Разрушительное землетрясение магнитудой 7.8 по шкале Рихтера произошло в Турции "  # noqa: E501
                "в понедельник утром по местному времени. Эпицентр находился в провинции Газиантеп. "  # noqa: E501
                "Подземные толчки ощущались также в соседних странах включая Сирию, Ливан и Кипр. "  # noqa: E501
                "Тысячи зданий обрушились, особенно пострадали старые постройки. Спасательные службы "  # noqa: E501
                "работают круглосуточно, извлекая людей из-под завалов. По предварительным данным "  # noqa: E501
                "погибли сотни человек, тысячи получили ранения."
            ),  # Очень длинный и детальный
            "source_url": "http://news3.com/detailed",
            "credibility_score": 0.9,
            "ingested_timestamp": 3000,
            "supabase_id": "s3",
            "embedding": vec3_emb.tolist(),
        },
        {
            "id": "weather_1",
            "content": "Погода в Анкаре: солнечно, 15 градусов.",  # Несвязанная тема
            "source_url": "http://weather.com/ankara",
            "credibility_score": 0.5,
            "ingested_timestamp": 4000,
            "supabase_id": "s4",
            "embedding": vec4_emb.tolist(),
        },
    ]

    print("  news_1: центральный, средняя детальность")
    print("  news_2: центральный, низкая детальность (короткий)")
    print("  news_3: центральный, высокая детальность (длинный)")
    print("  weather_1: outlier (несвязанная тема), низкая детальность")

    # Запускаем анализ с балансированными весами
    print("\n" + "-" * 80)
    print("Анализ с балансированными весами (60% центральность, 40% детальность):")
    print("-" * 80)

    best, all_ranked = find_best_vector_with_embeddings(
        vectors,
        centrality_weight=0.6,
        detail_weight=0.4,
    )

    print("\n📊 Результаты анализа:\n")
    for i, vec in enumerate(all_ranked, 1):
        marker = "🏆" if i == 1 else f"{i}."
        print(f"{marker} {vec.id}")
        print(f"    Центральность: {vec.centrality_score:.3f}")
        print(f"    Детальность:   {vec.detail_score:.3f}")
        print(f"    Общий балл:    {vec.combined_score:.3f}")
        print(f"    Длина текста:  {vec.content_length} символов")
        print()

    print(f"✅ Лучший вектор: {best.id}")
    reason = "центральный и подробный" if best.id == "news_3" else "баланс факторов"
    print(f"   Причина: {reason}")

    # Попробуем с акцентом на центральность
    print("\n" + "-" * 80)
    print("Анализ с акцентом на центральность (90% центральность, 10% детальность):")
    print("-" * 80)

    best_cent, _ = find_best_vector_with_embeddings(
        vectors,
        centrality_weight=0.9,
        detail_weight=0.1,
    )

    print(f"\n✅ Лучший (центральность): {best_cent.id}")

    # Попробуем с акцентом на детальность
    print("\n" + "-" * 80)
    print("Анализ с акцентом на детальность (20% центральность, 80% детальность):")
    print("-" * 80)

    best_detail, _ = find_best_vector_with_embeddings(
        vectors,
        centrality_weight=0.2,
        detail_weight=0.8,
    )

    print(f"\n✅ Лучший (детальность): {best_detail.id}")


def main():
    """Запуск всех демо."""
    print("\n" + "🔬" * 40)
    print("  ДЕМОНСТРАЦИЯ VECTOR CONSENSUS ANALYSIS")
    print("🔬" * 40)

    try:
        demo_pairwise_similarities()
        demo_centrality_scores()
        demo_detail_scores()
        demo_combined_analysis()

        print("\n" + "=" * 80)
        print("  ✅ ВСЕ ДЕМО ЗАВЕРШЕНЫ УСПЕШНО!")
        print("=" * 80)

        print("\n📚 Для использования в вашем коде:")
        print(
            """
from veritatis.vector_consensus import find_most_relevant_vector

# Найти лучший вектор из Tier 1
best, all_ranked = find_most_relevant_vector(
    collection_name="veritatis_tier1_lake",
    centrality_weight=0.6,
    detail_weight=0.4,
)

print(f"Лучший: {best.id}, score: {best.combined_score:.3f}")
        """
        )

        print("\n📖 Документация: tests/README_VECTOR_CONSENSUS.md")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
