from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIRST_GAME_RESULTS_DIR, K_COURSES, ensure_directories


sns.set_theme(style="whitegrid")


def load_results(results_dir=FIRST_GAME_RESULTS_DIR):
    results_dir = Path(results_dir)

    assignment = pd.read_csv(results_dir / "centralized_assignment_matrix.csv", index_col=0)
    course_loads = pd.read_csv(results_dir / "centralized_course_loads.csv")
    student_loads = pd.read_csv(results_dir / "centralized_student_loads.csv")
    objective = pd.read_csv(results_dir / "centralized_objective_value.csv")
    open_courses = pd.read_csv(results_dir / "centralized_open_courses.csv")

    return assignment, course_loads, student_loads, objective, open_courses


def _ensure_plots_dir(results_dir):
    plots_dir = Path(results_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def plot_course_loads(course_loads, save_dir):
    df = course_loads.copy()
    df["status"] = df.apply(
        lambda row: "Открыт" if row["is_open"] == 1 else "Закрыт",
        axis=1
    )
    df["below_min_if_open"] = df.apply(
        lambda row: row["is_open"] == 1 and row["L_opt"] < row["min_quantity"],
        axis=1
    )

    df = df.sort_values(["is_open", "L_opt"], ascending=[False, False]).reset_index(drop=True)

    colors = []
    for _, row in df.iterrows():
        if row["is_open"] == 0:
            colors.append("#B0B0B0")
        elif row["below_min_if_open"]:
            colors.append("#DD8452")
        else:
            colors.append("#4C72B0")

    fig, ax = plt.subplots(figsize=(11, 6))

    y_pos = range(len(df))
    ax.barh(y_pos, df["capacity"], color="#D9D9D9", alpha=0.6, label="Вместимость")
    ax.barh(y_pos, df["L_opt"], color=colors, alpha=0.95, label="Фактическая загрузка")

    for i, row in df.iterrows():
        ax.text(
            row["L_opt"] + 0.2,
            i,
            f"{int(row['L_opt'])}/{int(row['capacity'])}",
            va="center",
            fontsize=9
        )
        ax.axvline(x=row["min_quantity"], color="#C44E52", linestyle="--", alpha=0.15)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["course_id"])
    ax.invert_yaxis()
    ax.set_xlabel("Количество студентов")
    ax.set_ylabel("Курсы")
    ax.set_title("Загрузка курсов: вместимость, фактический набор и порог открытия")

    from matplotlib.lines import Line2D
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="#D9D9D9", alpha=0.6, label="Вместимость"),
        plt.Rectangle((0, 0), 1, 1, color="#4C72B0", label="Открытый курс"),
        plt.Rectangle((0, 0), 1, 1, color="#B0B0B0", label="Закрытый курс"),
        Line2D([0], [0], color="#C44E52", lw=2, linestyle="--", label="Минимум для открытия"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_dir / "centralized_course_loads.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_assignment_heatmap(assignment, save_dir):
    rows = assignment.shape[0]

    if rows <= 40:
        plot_df = assignment.copy()
        figsize = (10, max(6, rows * 0.35))
        show_y = True
    else:
        plot_df = assignment.iloc[:40].copy()
        figsize = (10, 12)
        show_y = False

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_df,
        cmap=sns.color_palette(["#F2F2F2", "#2A9D8F"], as_cmap=True),
        cbar=True,
        linewidths=0.4,
        linecolor="white",
        vmin=0,
        vmax=1,
        ax=ax
    )

    ax.set_title(
        "Матрица назначений студентов на курсы"
        + (" (первые 40 студентов)" if rows > 40 else "")
    )
    ax.set_xlabel("Курсы")
    ax.set_ylabel("Студенты")

    if not show_y:
        ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(save_dir / "centralized_assignment_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_student_load_distribution(student_loads, save_dir):
    df = student_loads.copy()

    counts = (
        df["assigned_courses"]
        .value_counts()
        .reindex(range(K_COURSES + 1), fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#55A868", alpha=0.9)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.1,
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_xlabel("Сколько курсов получил студент")
    ax.set_ylabel("Число студентов")
    ax.set_title("Распределение студентов по числу назначенных курсов")

    plt.tight_layout()
    plt.savefig(save_dir / "centralized_student_load_distribution.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_unassigned_students(student_loads, save_dir):
    df = student_loads.copy().sort_values(
        ["unassigned_slots", "assigned_courses", "student_id"],
        ascending=[False, True, True]
    )

    top_n = min(25, len(df))
    df_top = df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(11, max(5, top_n * 0.35)))

    y_pos = range(len(df_top))
    ax.barh(y_pos, df_top["max_allowed_courses"], color="#E5E5E5", label="Максимум K", alpha=0.8)
    ax.barh(y_pos, df_top["assigned_courses"], color="#4C72B0", label="Назначено", alpha=0.95)

    for i, row in enumerate(df_top.itertuples()):
        ax.text(
            row.assigned_courses + 0.05,
            i,
            f"{row.assigned_courses}/{row.max_allowed_courses}",
            va="center",
            fontsize=9
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_top["student_id"])
    ax.invert_yaxis()
    ax.set_xlabel("Количество курсов")
    ax.set_ylabel("Студенты")
    ax.set_title("Студенты с наибольшим числом незакрытых слотов (top-25)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "centralized_unassigned_students.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_summary_dashboard(course_loads, student_loads, objective, save_dir):
    z_opt = float(objective.loc[0, "Z_opt"])
    total_assigned = int(objective.loc[0, "total_assigned"])
    total_possible = int(objective.loc[0, "total_possible"])
    opened_courses = int(course_loads["is_open"].sum())
    total_courses = int(len(course_loads))
    full_students = int((student_loads["assigned_courses"] == K_COURSES).sum())
    partial_students = int((student_loads["assigned_courses"] < K_COURSES).sum())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    metrics = [
        ("Z_opt", f"{z_opt:.2f}", "#4C72B0"),
        ("Назначено мест", f"{total_assigned}/{total_possible}", "#55A868"),
        ("Открыто курсов", f"{opened_courses}/{total_courses}", "#C44E52"),
        ("Студентов с K курсами", str(full_students), "#8172B2"),
        ("Студентов с недобором", str(partial_students), "#DD8452"),
        ("Средняя загрузка курса", f"{course_loads['L_opt'].mean():.2f}", "#937860"),
    ]

    for ax, (title, value, color) in zip(axes, metrics):
        ax.set_facecolor("#F8F9FA")
        ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=18, fontweight="bold", color=color)
        ax.text(0.5, 0.28, title, ha="center", va="center", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle("Итоговые показатели централизованной модели", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(save_dir / "centralized_summary_dashboard.png", dpi=220, bbox_inches="tight")
    plt.close()


def create_all_visualizations():
    ensure_directories()

    results_dir = Path(FIRST_GAME_RESULTS_DIR)
    plots_dir = _ensure_plots_dir(results_dir)

    assignment, course_loads, student_loads, objective, open_courses = load_results(results_dir)

    plot_course_loads(course_loads, plots_dir)
    plot_assignment_heatmap(assignment, plots_dir)
    plot_student_load_distribution(student_loads, plots_dir)
    plot_unassigned_students(student_loads, plots_dir)
    plot_summary_dashboard(course_loads, student_loads, objective, plots_dir)
    closed_courses = save_closed_courses_reports(course_loads, plots_dir)
    plot_closed_courses(closed_courses, plots_dir)

    print("Visualizations created successfully:")
    print(f" - {plots_dir / 'centralized_course_loads.png'}")
    print(f" - {plots_dir / 'centralized_assignment_heatmap.png'}")
    print(f" - {plots_dir / 'centralized_student_load_distribution.png'}")
    print(f" - {plots_dir / 'centralized_unassigned_students.png'}")
    print(f" - {plots_dir / 'centralized_summary_dashboard.png'}")

def build_closed_courses_table(course_loads):
    df = course_loads.copy()

    closed = df[df["is_open"] == 0].copy()
    if closed.empty:
        return pd.DataFrame(columns=[
            "course_id",
            "L_opt",
            "min_quantity",
            "capacity",
            "students_missing_to_open"
        ])

    closed["students_missing_to_open"] = (
        closed["min_quantity"] - closed["L_opt"]
    ).clip(lower=0)

    closed = closed[[
        "course_id",
        "L_opt",
        "min_quantity",
        "capacity",
        "students_missing_to_open"
    ]].sort_values(
        by=["students_missing_to_open", "course_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return closed

def save_closed_courses_reports(course_loads, save_dir):
    closed = build_closed_courses_table(course_loads)

    csv_path = save_dir / "centralized_closed_courses.csv"
    txt_path = save_dir / "centralized_closed_courses_log.txt"

    closed.to_csv(csv_path, index=False)

    lines = []
    lines.append("НЕОТКРЫВШИЕСЯ КУРСЫ\n")
    lines.append("=" * 60 + "\n")

    if closed.empty:
        lines.append("Все курсы открылись.\n")
    else:
        lines.append(f"Количество неоткрывшихся курсов: {len(closed)}\n\n")
        for _, row in closed.iterrows():
            lines.append(
                f"Курс {int(row['course_id'])}: "
                f"загрузка={int(row['L_opt'])}, "
                f"минимум_для_открытия={int(row['min_quantity'])}, "
                f"вместимость={int(row['capacity'])}, "
                f"не хватило студентов={int(row['students_missing_to_open'])}\n"
            )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return closed

def plot_closed_courses(closed_courses, save_dir):
    if closed_courses.empty:
        return

    df = closed_courses.copy().sort_values(
        "students_missing_to_open",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, max(4, 0.7 * len(df))))

    ax.barh(df["course_id"].astype(str), df["min_quantity"], color="#E6E6E6", label="Минимум для открытия")
    ax.barh(df["course_id"].astype(str), df["L_opt"], color="#C44E52", label="Фактическая загрузка")

    for i, row in df.reset_index(drop=True).iterrows():
        ax.text(
            row["L_opt"] + 0.2,
            i,
            f"{int(row['L_opt'])}/{int(row['min_quantity'])}",
            va="center",
            fontsize=9
        )

    ax.set_title("Курсы, которые не открылись")
    ax.set_xlabel("Количество студентов")
    ax.set_ylabel("Курс")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "centralized_closed_courses.png", dpi=220, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    create_all_visualizations()