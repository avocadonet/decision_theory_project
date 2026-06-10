from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import cvxpy as cp

from config import GENERATED_DIR, FIRST_GAME_RESULTS_DIR, K_COURSES, ensure_directories


def load_input_data(data_dir=GENERATED_DIR):
    data_dir = Path(data_dir)

    usefulness_course = pd.read_csv(data_dir / "usefulness_course.csv")
    quality_teaching = pd.read_csv(data_dir / "quality_teaching.csv")
    ease_passing_exam = pd.read_csv(data_dir / "ease_passing_exam.csv")
    student_laziness = pd.read_csv(data_dir / "student_laziness.csv")
    importance_certainty = pd.read_csv(data_dir / "importance_certainty.csv")
    exchange_rate_uncertainty = pd.read_csv(data_dir / "exchange_rate_uncertainty.csv")
    student_certainty_weight = pd.read_csv(data_dir / "student_certainty_weight.csv")
    capacity = pd.read_csv(data_dir / "capacity.csv")
    min_quantity = pd.read_csv(data_dir / "min_quantity.csv")

    return {
        "usefulness_course": usefulness_course,
        "quality_teaching": quality_teaching,
        "ease_passing_exam": ease_passing_exam,
        "student_laziness": student_laziness,
        "importance_certainty": importance_certainty,
        "exchange_rate_uncertainty": exchange_rate_uncertainty,
        "student_certainty_weight": student_certainty_weight,
        "capacity": capacity,
        "min_quantity": min_quantity,
    }


def build_utility_table(data):
    df = data["usefulness_course"].merge(
        data["quality_teaching"], on="course_id", how="left"
    )
    df = df.merge(data["ease_passing_exam"], on="course_id", how="left")
    df = df.merge(data["student_laziness"], on="student_id", how="left")
    df = df.merge(data["importance_certainty"], on="student_id", how="left")
    df = df.merge(data["exchange_rate_uncertainty"], on="course_id", how="left")
    df = df.merge(data["student_certainty_weight"], on="student_id", how="left")

    df["v_ij"] = (
        df["student_laziness"] * df["usefulness_course"] * df["quality_teaching"]
        + (1 - df["student_laziness"]) * df["usefulness_course"] * df["ease_passing_exam"]
        - df["importance_certainty"] * df["student_certainty_weight"] * df["exchange_rate_uncertainty"]
    )

    return df


def get_available_solver():
    installed = cp.installed_solvers()
    if "HIGHS" in installed:
        return cp.HIGHS
    if "GLPK_MI" in installed:
        return cp.GLPK_MI
    raise ValueError(
        "Не найден mixed-integer solver. Установи один из вариантов:\n"
        "pip install highspy\n"
        "или\n"
        "pip install cvxopt"
    )


def solve_centralized_optimization():
    ensure_directories()

    data = load_input_data()
    utility_df = build_utility_table(data)

    student_ids = sorted(utility_df["student_id"].unique())
    course_ids = sorted(utility_df["course_id"].unique())

    n = len(student_ids)
    m = len(course_ids)
    k = K_COURSES

    student_index = {sid: idx for idx, sid in enumerate(student_ids)}
    course_index = {cid: idx for idx, cid in enumerate(course_ids)}

    v = np.zeros((n, m))
    for _, row in utility_df.iterrows():
        i = student_index[row["student_id"]]
        j = course_index[row["course_id"]]
        v[i, j] = row["v_ij"]

    c_max = (
        data["capacity"]
        .sort_values("course_id")["capacity"]
        .to_numpy(dtype=int)
    )

    c_min = (
        data["min_quantity"]
        .sort_values("course_id")["min_quantity"]
        .to_numpy(dtype=int)
    )

    x = cp.Variable((n, m), boolean=True)
    y = cp.Variable(m, boolean=True)
    L = cp.Variable(m, integer=True)

    constraints = []

    # Студент может быть записан не более чем на K курсов
    for i in range(n):
        constraints.append(cp.sum(x[i, :]) <= k)

    # Связь загрузки с назначениями
    for j in range(m):
        constraints.append(L[j] == cp.sum(x[:, j]))
        constraints.append(L[j] >= 0)

    # Курс либо закрыт, либо открыт и набирает минимум/максимум
    for j in range(m):
        constraints.append(c_min[j] * y[j] <= L[j])
        constraints.append(L[j] <= c_max[j] * y[j])

    objective = cp.Maximize(cp.sum(cp.multiply(v, x)))
    problem = cp.Problem(objective, constraints)

    solver = get_available_solver()
    problem.solve(solver=solver, verbose=False)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed. Status: {problem.status}")

    x_opt = np.rint(x.value).astype(int)
    y_opt = np.rint(y.value).astype(int)
    l_opt = np.rint(L.value).astype(int)

    student_loads = x_opt.sum(axis=1).astype(int)

    results_dir = Path(FIRST_GAME_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    assignment_df = pd.DataFrame(
        x_opt,
        index=student_ids,
        columns=course_ids
    )
    assignment_df.index.name = "student_id"
    assignment_df.to_csv(results_dir / "centralized_assignment_matrix.csv")

    course_load_df = pd.DataFrame({
        "course_id": course_ids,
        "L_opt": l_opt,
        "capacity": c_max,
        "min_quantity": c_min,
        "is_open": y_opt,
    })
    course_load_df.to_csv(results_dir / "centralized_course_loads.csv", index=False)

    student_load_df = pd.DataFrame({
        "student_id": student_ids,
        "assigned_courses": student_loads,
        "max_allowed_courses": k,
        "unassigned_slots": k - student_loads,
    })
    student_load_df.to_csv(results_dir / "centralized_student_loads.csv", index=False)

    objective_df = pd.DataFrame([{
        "Z_opt": float(problem.value),
        "n_students": n,
        "m_courses": m,
        "k_courses": k,
        "solver": str(solver),
        "status": problem.status,
        "total_assigned": int(x_opt.sum()),
        "total_possible": int(n * k),
    }])
    objective_df.to_csv(results_dir / "centralized_objective_value.csv", index=False)

    opened_df = pd.DataFrame({
        "course_id": course_ids,
        "y_j": y_opt,
    })
    opened_df.to_csv(results_dir / "centralized_open_courses.csv", index=False)

    print("Centralized optimization completed.")
    print(f"Status: {problem.status}")
    print(f"Solver: {solver}")
    print(f"Z_opt: {problem.value:.4f}")
    print(f"Total assigned slots: {int(x_opt.sum())} / {n * k}")
    print(f"Opened courses: {int(y_opt.sum())} / {m}")

    return {
        "status": problem.status,
        "solver": str(solver),
        "Z_opt": float(problem.value),
        "x_opt": x_opt,
        "L_opt": l_opt,
        "y_opt": y_opt,
        "student_loads": student_loads,
    }


if __name__ == "__main__":
    solve_centralized_optimization()