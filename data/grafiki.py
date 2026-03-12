"""Визуализация результатов оценки моделей для извлечения HTML схем.

Модуль для создания графиков и отчетов по результатам оценки моделей.
Автор: Claude
"""
# mypy: disable-error-code="type-arg,no-any-return,index"

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate  # type: ignore[import-untyped]

# Настройка стиля графиков
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12


def load_data(filepath: str) -> dict:
    """Загрузка JSON данных."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_summary_metrics(data: dict) -> pd.DataFrame:
    """Извлечение сводных метрик по моделям."""
    models = data["models"]
    summary = []

    for model_name, model_data in models.items():
        summary.append(
            {
                "Model": model_name,
                "Total Links": model_data["total_links"],
                "Success Count": model_data["success_count"],
                "Success Rate (%)": model_data["success_rate"] * 100,
                "Extraction Acc (%)": model_data["extraction_accuracy_avg"] * 100,
                "GPT Quality (%)": model_data["gpt_quality_avg"] * 100,
                "Final Score": model_data["final_score"],
            }
        )

    return pd.DataFrame(summary).sort_values("Final Score", ascending=False)


def extract_detailed_metrics(data: dict) -> pd.DataFrame:
    """Извлечение детальных метрик по каждому URL."""
    models = data["models"]
    details = []

    for model_name, model_data in models.items():
        for item in model_data["details"]:
            gpt = item.get("gpt_judgment") or {}
            details.append(
                {
                    "Model": model_name,
                    "Domain": item["domain"],
                    "URL": item["url"],
                    "Main Text Found": item["extraction_accuracy"]["main_text_found"],
                    "Text Length": item["extraction_accuracy"]["main_text_length"],
                    "Date Found": item["extraction_accuracy"]["date_found"],
                    "Extraction Score": item["extraction_accuracy"][
                        "extraction_accuracy_score"
                    ],
                    "GPT Score": gpt.get("score", 0),
                    "GPT Confidence": gpt.get("confidence", 0),
                    "Main Text Quality": gpt.get("main_text_quality", 0),
                    "Date Quality": gpt.get("date_quality", 0),
                }
            )

    return pd.DataFrame(details)


def extract_error_metrics(data: dict) -> pd.DataFrame:
    """Извлечение метрик по ошибкам из error_breakdown.

    Returns:
        DataFrame с колонками: Model, Domain, URL, Category, Severity,
        Count, Penalty, Examples
    """
    models = data["models"]
    error_data = []

    for model_name, model_data in models.items():
        for item in model_data["details"]:
            if "gpt_judgment" not in item or item["gpt_judgment"] is None:
                continue

            judgment = item["gpt_judgment"]
            reason = judgment.get("reason", {})

            # Handle both old (string) and new (dict) reason format
            if isinstance(reason, dict):
                error_breakdown = reason.get("error_breakdown", [])

                for error in error_breakdown:
                    error_data.append(
                        {
                            "Model": model_name,
                            "Domain": item["domain"],
                            "URL": item["url"],
                            "Category": error.get("category", "Unknown"),
                            "Severity": error.get("severity", "unknown"),
                            "Count": error.get("count", 0),
                            "Penalty": error.get("penalty", 0.0),
                            "Examples": "; ".join(error.get("examples", [])),
                        }
                    )
            # Fallback: if no errors in breakdown, check if there are issues
            elif judgment.get("issues"):
                # For backward compatibility with old format
                error_data.append(
                    {
                        "Model": model_name,
                        "Domain": item["domain"],
                        "URL": item["url"],
                        "Category": "Unclassified",
                        "Severity": "medium",
                        "Count": len(judgment["issues"]),
                        "Penalty": 0.0,
                        "Examples": "; ".join(judgment["issues"][:3]),  # First 3 issues
                    }
                )

    return pd.DataFrame(error_data)


def print_summary_table(df: pd.DataFrame):
    """Вывод сводной таблицы."""
    print("\n" + "=" * 80)
    print("📊 СВОДКА ОЦЕНКИ МОДЕЛЕЙ")
    print("=" * 80 + "\n")

    table_data = df.copy()
    table_data["Success Rate (%)"] = table_data["Success Rate (%)"].apply(
        lambda x: f"{x:.1f}%"
    )
    table_data["Extraction Acc (%)"] = table_data["Extraction Acc (%)"].apply(
        lambda x: f"{x:.1f}%"
    )
    table_data["GPT Quality (%)"] = table_data["GPT Quality (%)"].apply(
        lambda x: f"{x:.1f}%"
    )
    table_data["Final Score"] = table_data["Final Score"].apply(lambda x: f"{x:.2f}")

    print(tabulate(table_data, headers="keys", tablefmt="fancy_grid", showindex=False))

    winner = df.loc[df["Final Score"].idxmax()]
    model_name = winner["Model"].upper()
    final_score = winner["Final Score"]
    print(f"\n🏆 Победитель: {model_name} с итоговым счётом {final_score:.2f}/100")


def print_detailed_table(df: pd.DataFrame):
    """Вывод детальной таблицы по доменам."""
    print("\n" + "=" * 80)
    print("📋 ДЕТАЛЬНАЯ ОЦЕНКА ПО ДОМЕНАМ")
    print("=" * 80 + "\n")

    for model in df["Model"].unique():
        model_df = df[df["Model"] == model].copy()
        print(f"\n--- {model.upper()} ---\n")

        display_df = model_df[
            [
                "Domain",
                "Text Length",
                "Extraction Score",
                "GPT Score",
                "Main Text Quality",
                "Date Quality",
            ]
        ].copy()
        display_df["Extraction Score"] = display_df["Extraction Score"].apply(
            lambda x: f"{x:.2f}"
        )
        display_df["GPT Score"] = display_df["GPT Score"].apply(lambda x: f"{x:.1f}")
        display_df["Main Text Quality"] = display_df["Main Text Quality"].apply(
            lambda x: f"{x:.1f}"
        )
        display_df["Date Quality"] = display_df["Date Quality"].apply(
            lambda x: f"{x:.1f}"
        )

        print(tabulate(display_df, headers="keys", tablefmt="simple", showindex=False))


def plot_summary_comparison(df: pd.DataFrame, output_dir: str = "."):
    """График сравнения основных метрик моделей."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Model Comparison by Key Metrics", fontsize=16, fontweight="bold"
    )

    models = df["Model"].tolist()
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"][: len(models)]

    # 1. Success Rate
    ax1 = axes[0, 0]
    bars1 = ax1.bar(
        models, df["Success Rate (%)"], color=colors, edgecolor="black", linewidth=1.2
    )
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Extraction Success Rate")
    ax1.set_ylim(0, 100)
    for bar, val in zip(bars1, df["Success Rate (%)"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Extraction Accuracy
    ax2 = axes[0, 1]
    bars2 = ax2.bar(
        models, df["Extraction Acc (%)"], color=colors, edgecolor="black", linewidth=1.2
    )
    ax2.set_ylabel("Extraction Accuracy (%)")
    ax2.set_title("Extraction Accuracy")
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars2, df["Extraction Acc (%)"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. GPT Quality
    ax3 = axes[1, 0]
    bars3 = ax3.bar(
        models, df["GPT Quality (%)"], color=colors, edgecolor="black", linewidth=1.2
    )
    ax3.set_ylabel("GPT Quality (%)")
    ax3.set_title("GPT Quality Score")
    ax3.set_ylim(0, 100)
    for bar, val in zip(bars3, df["GPT Quality (%)"]):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 4. Final Score
    ax4 = axes[1, 1]
    bars4 = ax4.bar(
        models, df["Final Score"], color=colors, edgecolor="black", linewidth=1.2
    )
    ax4.set_ylabel("Final Score")
    ax4.set_title("Final Score (out of 100)")
    ax4.set_ylim(0, 100)
    for bar, val in zip(bars4, df["Final Score"]):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Сохранено: {output_dir}/model_comparison_bars.png")


def plot_radar_chart(df: pd.DataFrame, output_dir: str = "."):
    """Радарная диаграмма для сравнения моделей."""
    categories = ["Success Rate", "Extraction Acc", "GPT Quality", "Final Score (norm)"]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Замкнуть круг

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]

    for idx, (_, row) in enumerate(df.iterrows()):
        values = [
            row["Success Rate (%)"],
            row["Extraction Acc (%)"],
            row["GPT Quality (%)"],
            row["Final Score"],  # Уже в шкале 0-100
        ]
        values += values[:1]  # Замкнуть

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=row["Model"].upper(),
            color=colors[idx % len(colors)],
        )
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Radar Chart: Model Comparison", fontsize=14, fontweight="bold", pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Сохранено: {output_dir}/model_radar_chart.png")


def plot_domain_heatmap(details_df: pd.DataFrame, output_dir: str = "."):
    """Тепловая карта качества по доменам."""
    # Pivot table для GPT Score
    pivot = details_df.pivot_table(
        values="GPT Score", index="Domain", columns="Model", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.4)))

    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=10)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([m.upper() for m in pivot.columns])
    ax.set_yticklabels(pivot.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Добавить значения в ячейки
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 4 or val > 7 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                )

    ax.set_title("GPT Score by Domain and Model", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="GPT Score (0-10)")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/domain_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Сохранено: {output_dir}/domain_heatmap.png")


def plot_quality_distribution(details_df: pd.DataFrame, output_dir: str = "."):
    """Распределение качества по моделям."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = details_df["Model"].unique()
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"][: len(models)]

    # 1. Box plot для GPT Score
    ax1 = axes[0]
    data_gpt = [
        details_df[details_df["Model"] == m]["GPT Score"].values for m in models
    ]
    bp1 = ax1.boxplot(data_gpt, labels=[m.upper() for m in models], patch_artist=True)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel("GPT Score")
    ax1.set_title("GPT Score Distribution by Model")
    ax1.set_ylim(0, 10)
    ax1.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="Average level")
    ax1.legend()

    # 2. Box plot для Main Text Quality
    ax2 = axes[1]
    data_text = [
        details_df[details_df["Model"] == m]["Main Text Quality"].values for m in models
    ]
    bp2 = ax2.boxplot(data_text, labels=[m.upper() for m in models], patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Main Text Quality")
    ax2.set_title("Text Extraction Quality Distribution")
    ax2.set_ylim(0, 10)
    ax2.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="Average level")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/quality_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Сохранено: {output_dir}/quality_distribution.png")


def plot_text_vs_date_quality(details_df: pd.DataFrame, output_dir: str = "."):
    """Scatter plot: качество текста vs качество даты."""
    fig, ax = plt.subplots(figsize=(10, 8))

    models = details_df["Model"].unique()
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"][: len(models)]

    for idx, model in enumerate(models):
        model_data = details_df[details_df["Model"] == model]
        ax.scatter(
            model_data["Main Text Quality"],
            model_data["Date Quality"],
            s=100,
            alpha=0.7,
            label=model.upper(),
            color=colors[idx],
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("Main Text Quality (0-10)")
    ax.set_ylabel("Date Quality (0-10)")
    ax.set_title("Text Extraction Quality vs Date Extraction Quality")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=5, color="gray", linestyle="--", alpha=0.3)
    ax.legend()

    # Add quadrant labels
    ax.text(7.5, 7.5, "Excellent", fontsize=12, ha="center", color="green", alpha=0.7)
    ax.text(
        2.5, 7.5, "Weak Text", fontsize=10, ha="center", color="orange", alpha=0.7
    )
    ax.text(
        7.5, 2.5, "Weak Date", fontsize=10, ha="center", color="orange", alpha=0.7
    )
    ax.text(2.5, 2.5, "Poor", fontsize=12, ha="center", color="red", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/text_vs_date_quality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Сохранено: {output_dir}/text_vs_date_quality.png")


def plot_stacked_metrics(df: pd.DataFrame, output_dir: str = "."):
    """Стековая диаграмма компонентов итогового счёта."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.6

    # Нормализуем метрики для визуализации вклада
    success_contrib = df["Success Rate (%)"] * 0.2  # 20% вес
    extract_contrib = df["Extraction Acc (%)"] * 0.4  # 40% вес
    gpt_contrib = df["GPT Quality (%)"] * 0.4  # 40% вес

    ax.bar(x, success_contrib, width, label="Success Rate (20%)", color="#3498db")
    ax.bar(
        x,
        extract_contrib,
        width,
        bottom=success_contrib,
        label="Extraction Acc (40%)",
        color="#2ecc71",
    )
    ax.bar(
        x,
        gpt_contrib,
        width,
        bottom=success_contrib + extract_contrib,
        label="GPT Quality (40%)",
        color="#e74c3c",
    )

    ax.set_ylabel("Weighted Score Components")
    ax.set_title("Final Score Decomposition by Components")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend(loc="upper right")

    # Добавить итоговый счёт сверху
    for i, (_, row) in enumerate(df.iterrows()):
        total = success_contrib.iloc[i] + extract_contrib.iloc[i] + gpt_contrib.iloc[i]
        ax.text(
            i,
            total + 2,
            f'{row["Final Score"]:.1f}',
            ha="center",
            fontweight="bold",
            fontsize=12,
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Сохранено: {output_dir}/score_decomposition.png")


def plot_error_analysis(error_df: pd.DataFrame, output_dir: str = "."):
    """График анализа ошибок по источникам (доменам).

    Создает:
    1. Bar chart топ-10 доменов с наибольшим числом ошибок
    2. Stacked bar chart распределения типов ошибок по топ-доменам
    3. Таблицу детализации
    """
    if error_df.empty:
        print("⚠️  Нет данных об ошибках для визуализации")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle(
        "Error Analysis by Source (Domains)", fontsize=16, fontweight="bold"
    )

    # 1. Топ-10 доменов по общему количеству ошибок
    ax1 = axes[0]
    domain_errors = (
        error_df.groupby("Domain")["Count"].sum().sort_values(ascending=False)
    )
    top_domains = domain_errors.head(10)

    colors_gradient = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_domains)))
    bars = ax1.barh(range(len(top_domains)), top_domains.values, color=colors_gradient)
    ax1.set_yticks(range(len(top_domains)))
    ax1.set_yticklabels(top_domains.index)
    ax1.set_xlabel("Number of Errors")
    ax1.set_title("Top 10 Domains with Most Errors")
    ax1.invert_yaxis()

    # Добавить значения на столбцах
    for _i, (bar, val) in enumerate(zip(bars, top_domains.values)):
        ax1.text(
            val + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{int(val)}",
            va="center",
            fontweight="bold",
        )

    # 2. Stacked bar chart: распределение типов ошибок по топ-доменам
    ax2 = axes[1]
    top_domain_names = top_domains.index.tolist()
    error_by_category = (
        error_df[error_df["Domain"].isin(top_domain_names)]
        .groupby(["Domain", "Category"])["Count"]
        .sum()
        .unstack(fill_value=0)
    )

    # Сортируем по тому же порядку что и в топ-10
    error_by_category = error_by_category.reindex(top_domain_names)

    category_colors = {
        "Missing Content": "#e74c3c",
        "Excessive Noise": "#f39c12",
        "Wrong Element": "#9b59b6",
        "Fragility": "#3498db",
        "Unclassified": "#95a5a6",
    }

    x_pos = np.arange(len(error_by_category))
    bottom = np.zeros(len(error_by_category))

    for category in error_by_category.columns:
        color = category_colors.get(category, "#95a5a6")
        values = error_by_category[category].values
        ax2.barh(x_pos, values, left=bottom, label=category, color=color)
        bottom += values

    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(error_by_category.index)
    ax2.set_xlabel("Number of Errors by Category")
    ax2.set_title("Error Type Distribution for Top Domains")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Сохранено: {output_dir}/error_analysis.png")

    # 3. Создать детальную таблицу ошибок
    _print_error_table(error_df, top_domain_names)


def _print_error_table(error_df: pd.DataFrame, top_domains: list):
    """Вывод таблицы ошибок по топ-доменам."""
    print("\n" + "=" * 100)
    print("📊 ДЕТАЛЬНАЯ СТАТИСТИКА ОШИБОК ПО ТОП-ДОМЕНАМ")
    print("=" * 100 + "\n")

    for domain in top_domains:
        domain_errors = error_df[error_df["Domain"] == domain]
        if domain_errors.empty:
            continue

        print(f"\n--- {domain} ---")

        # Группировка по категориям
        category_summary = (
            domain_errors.groupby(["Category", "Severity"])
            .agg({"Count": "sum", "Penalty": "sum", "Examples": lambda x: "; ".join(x)})
            .reset_index()
        )

        # Сортировка: сначала критичные, потом средние, потом мелкие
        severity_order = {"critical": 0, "medium": 1, "minor": 2, "unknown": 3}
        category_summary["severity_rank"] = category_summary["Severity"].map(
            severity_order
        )
        category_summary = category_summary.sort_values("severity_rank").drop(
            "severity_rank", axis=1
        )

        display_df = category_summary[
            ["Category", "Severity", "Count", "Penalty"]
        ].copy()
        display_df["Penalty"] = display_df["Penalty"].apply(lambda x: f"{x:.1f}")

        print(tabulate(display_df, headers="keys", tablefmt="simple", showindex=False))

        # Вывод примеров для каждой категории
        for _, row in category_summary.iterrows():
            examples = row["Examples"]
            if examples and examples != "":
                print(f"  Примеры ({row['Category']} - {row['Severity']}):")
                for example in examples.split("; ")[:2]:  # Показать первые 2 примера
                    if example.strip():
                        print(f"    • {example[:100]}...")

    print(f"\n{'=' * 100}\n")


def generate_html_report(
    summary_df: pd.DataFrame,
    details_df: pd.DataFrame,
    error_df: pd.DataFrame,
    output_dir: str = ".",
):
    """Генерация HTML отчёта."""
    # Генерируем строки таблицы для сводки
    summary_rows = []
    for _, row in summary_df.iterrows():
        score = row["Final Score"]
        score_class = (
            "metric-good"
            if score >= 75
            else ("metric-medium" if score >= 60 else "metric-bad")
        )
        summary_rows.append(
            f"""
            <tr>
                <td><strong>{row["Model"].upper()}</strong></td>
                <td>{row["Success Rate (%)"]:.1f}%</td>
                <td>{row["Extraction Acc (%)"]:.1f}%</td>
                <td>{row["GPT Quality (%)"]:.1f}%</td>
                <td class="{score_class}">{row["Final Score"]:.2f}</td>
            </tr>"""
        )

    # Генерируем строки таблиц для детальной статистики
    model_sections = []
    for model in details_df["Model"].unique():
        model_data = details_df[details_df["Model"] == model]
        detail_rows = []
        for _, row in model_data.iterrows():
            gpt_score = row["GPT Score"]
            gpt_class = (
                "metric-good"
                if gpt_score >= 7
                else ("metric-medium" if gpt_score >= 5 else "metric-bad")
            )
            detail_rows.append(
                f"""
            <tr>
                <td>{row["Domain"]}</td>
                <td>{row["Text Length"]}</td>
                <td>{row["Extraction Score"]:.2f}</td>
                <td class="{gpt_class}">{row["GPT Score"]:.1f}</td>
                <td>{row["Main Text Quality"]:.1f}</td>
                <td>{row["Date Quality"]:.1f}</td>
            </tr>"""
            )

        model_sections.append(
            f"""
        <h3>{model.upper()}</h3>
        <table>
            <tr>
                <th>Domain</th>
                <th>Text Length</th>
                <th>Extraction Score</th>
                <th>GPT Score</th>
                <th>Main Text Quality</th>
                <th>Date Quality</th>
            </tr>
            {''.join(detail_rows)}
        </table>
        """
        )

    # Генерируем секцию анализа ошибок
    error_section = ""
    if not error_df.empty:
        # Топ-10 доменов по ошибкам
        top_error_domains = (
            error_df.groupby("Domain")["Count"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        error_rows = []
        for domain, total_count in top_error_domains.items():
            domain_errors = error_df[error_df["Domain"] == domain]

            # Группировка по категориям
            category_breakdown = (
                domain_errors.groupby("Category")["Count"].sum().to_dict()
            )
            total_penalty = domain_errors["Penalty"].sum()

            # Форматируем категории
            categories_str = ", ".join(
                [f"{cat}: {count}" for cat, count in category_breakdown.items()]
            )

            error_rows.append(
                f"""
            <tr>
                <td><strong>{domain}</strong></td>
                <td>{int(total_count)}</td>
                <td class="metric-bad">{total_penalty:.1f}</td>
                <td style="font-size: 0.9em;">{categories_str}</td>
            </tr>"""
            )

        error_section = f"""
    <div class="summary-card">
        <h2>🔍 Error Analysis by Source</h2>
        <p>Top 10 domains with the highest number of selector errors</p>
        <table>
            <tr>
                <th>Domain</th>
                <th>Total Errors</th>
                <th>Penalty (points)</th>
                <th>Category Breakdown</th>
            </tr>
            {''.join(error_rows)}
        </table>

        <h3 style="margin-top: 30px;">Error Categories:</h3>
        <ul style="line-height: 1.8;">
            <li><strong>Missing Content</strong> (critical): important content was not extracted</li>
            <li><strong>Excessive Noise</strong> (medium): extra content extracted (ads, navigation)</li>
            <li><strong>Wrong Element</strong> (critical): selector points to wrong element</li>
            <li><strong>Fragility</strong> (minor): fragile selectors (nth-child, long chains)</li>
        </ul>
    </div>
    """

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2 {{
            color: #2c3e50;
        }}
        .summary-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .winner {{
            background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
            color: #2c3e50;
            padding: 15px 25px;
            border-radius: 10px;
            font-size: 1.2em;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-good {{ color: #27ae60; font-weight: bold; }}
        .metric-medium {{ color: #f39c12; font-weight: bold; }}
        .metric-bad {{ color: #e74c3c; font-weight: bold; }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-gallery img {{
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>📊 HTML Extraction Model Evaluation Report</h1>

    <div class="winner">
        🏆 Winner: <strong>{summary_df.iloc[0]['Model'].upper()}</strong>
        with final score <strong>{summary_df.iloc[0]['Final Score']:.2f}/100</strong>
    </div>

    <div class="summary-card">
        <h2>Summary Table</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Success Rate</th>
                <th>Extraction Acc</th>
                <th>GPT Quality</th>
                <th>Final Score</th>
            </tr>
            {''.join(summary_rows)}
        </table>
    </div>

    <div class="summary-card">
        <h2>Visualizations</h2>
        <div class="image-gallery">
            <img src="model_comparison_bars.png" alt="Model Comparison">
            <img src="model_radar_chart.png" alt="Radar Chart">
            <img src="domain_heatmap.png" alt="Domain Heatmap">
            <img src="quality_distribution.png" alt="Quality Distribution">
            <img src="text_vs_date_quality.png" alt="Text vs Date Quality">
            <img src="score_decomposition.png" alt="Score Decomposition">
            <img src="error_analysis.png" alt="Error Analysis">
        </div>
    </div>

    {error_section}

    <div class="summary-card">
        <h2>Model Statistics</h2>
        {''.join(model_sections)}
    </div>

    <footer style="text-align: center; margin-top: 40px; color: #7f8c8d;">
        <p>Report generated automatically</p>
    </footer>
</body>
</html>
"""

    with open(f"{output_dir}/report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Сохранено: {output_dir}/report.html")


def main(json_path: str, output_dir: str = "data/grafiki"):
    """Главная функция."""
    # Создать директорию если не существует
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("🔄 Загрузка данных...")
    data = load_data(json_path)

    print("📊 Извлечение метрик...")
    summary_df = extract_summary_metrics(data)
    details_df = extract_detailed_metrics(data)
    error_df = extract_error_metrics(data)

    # Вывод таблиц в консоль
    print_summary_table(summary_df)
    print_detailed_table(details_df)

    print("\n📈 Генерация графиков...")
    plot_summary_comparison(summary_df, output_dir)
    plot_radar_chart(summary_df, output_dir)
    plot_domain_heatmap(details_df, output_dir)
    plot_quality_distribution(details_df, output_dir)
    plot_text_vs_date_quality(details_df, output_dir)
    plot_stacked_metrics(summary_df, output_dir)

    print("\n🔍 Анализ ошибок...")
    plot_error_analysis(error_df, output_dir)

    print("\n📄 Генерация HTML отчёта...")
    generate_html_report(summary_df, details_df, error_df, output_dir)

    print("\n" + "=" * 80)
    print("✅ ГОТОВО! Все файлы сохранены в:", output_dir)
    print("=" * 80)

    return summary_df, details_df, error_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = "data/data/model_evaluation.json"

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "data/grafiki"

    main(json_path, output_dir)
