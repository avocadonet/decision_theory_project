from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIRST_GAME_RESULTS_DIR, ensure_directories


def visualize_first_game(
    results_dir=FIRST_GAME_RESULTS_DIR
):
    ensure_directories()
    results_dir = Path(results_dir)

    student_loads_file = results_dir / "greedy_student_loads.csv"
    course_loads_file = results_dir / "greedy_course_loads.csv"
    assignment_matrix_file = results_dir / "greedy_assignment_matrix.csv"

    student_loads = pd.read_csv(student_loads_file)
    course_loads = pd.read_csv(course_loads_file)
    assignment_matrix = pd.read_csv(assignment_matrix_file)

    sns.set(style="whitegrid")

    # 1. Загрузка студентов
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=student_loads,
        x="student_id",
        y="assigned_courses_count",
        color="steelblue"
    )
    plt.title("Number of assigned courses per student")
    plt.xlabel("Student ID")
    plt.ylabel("Assigned courses count")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_student_loads.png", dpi=300)
    plt.close()

    # 2. Загрузка курсов
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=course_loads,
        x="course_id",
        y="assigned_students_count",
        color="darkorange"
    )

    for i, row in course_loads.iterrows():
        ax.text(
            i,
            row["assigned_students_count"] + 0.2,
            f'cap={row["capacity"]}',
            ha="center",
            fontsize=9
        )

    plt.title("Course loads")
    plt.xlabel("Course ID")
    plt.ylabel("Assigned students count")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_course_loads.png", dpi=300)
    plt.close()

    # 3. Матрица назначений
    matrix = assignment_matrix.drop(columns=["student_id"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        matrix,
        annot=True,
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="gray"
    )
    plt.title("Assignment matrix")
    plt.xlabel("Course ID")
    plt.ylabel("Student ID")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_assignment_matrix.png", dpi=300)
    plt.close()

    print("Visualization files created:")
    print(f" - {results_dir / 'plot_student_loads.png'}")
    print(f" - {results_dir / 'plot_course_loads.png'}")
    print(f" - {results_dir / 'plot_assignment_matrix.png'}")


if __name__ == "__main__":
    visualize_first_game()