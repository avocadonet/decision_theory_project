import sys
from pathlib import Path
import time as tm
import numpy as np
import pandas as pd
import cvxpy as cp

# Автоматическое добавление корня проекта в пути поиска модулей
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GENERATED_DIR, FIRST_GAME_RESULTS_DIR, ensure_directories


def get_available_solver():
    """Определяет оптимальный доступный MILP-солвер."""
    if "HIGHS" in cp.installed_solvers():
        return cp.HIGHS
    return cp.CBC


def load_input_data(data_dir=GENERATED_DIR):
    """Загружает исходные файлы и рассчитывает сводную матрицу полезности."""
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

    course_ids = capacity["course_id"].values
    student_ids = student_laziness["student_id"].values

    n = len(student_ids)
    m = len(course_ids)

    # Построение матрицы полезности v (N x M)
    v = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            u_base = usefulness_course.iloc[j]["usefulness"]
            q = quality_teaching.iloc[j]["quality"]
            e = ease_passing_exam.iloc[j]["ease"]
            l = student_laziness.iloc[i]["laziness"]
            w_cert = student_certainty_weight.iloc[i]["weight"]
            imp = importance_certainty.iloc[j]["importance"]
            unc = exchange_rate_uncertainty.iloc[j]["uncertainty"]

            v[i, j] = (u_base * q + e * (1 - l)) + w_cert * (imp * (1 - unc))

    capacities = capacity["capacity"].values
    min_quantities = min_quantity["min_quantity"].values

    return v, capacities, min_quantities, course_ids, student_ids


def run_optimization_core(n, m, k, v, capacities, min_quantities, solver): 
    """
    Строит модель смешанно-целочисленного программирования
    """
    # Объявление бинарных переменных решения
    x = cp.Variable((n, m), boolean=True)  # Студент i на курсе j
    y = cp.Variable(m, boolean=True)       # Открыт ли курс j вообще

    # Формирование системы жестких ограничений
    constraints = []
    constraints.append(cp.sum(x, axis=1) == k)                          # Каждый студент берет ровно K курсов
    constraints.append(cp.sum(x, axis=0) <= cp.multiply(capacities, y))      # Ограничение сверху (вместимость)
    constraints.append(cp.sum(x, axis=0) >= cp.multiply(min_quantities, y))  # Ограничение снизу (минимальный порог)

    # Настройка целевой функции (максимизация глобального благосостояния)
    objective = cp.Maximize(cp.sum(cp.multiply(v, x)))
    problem = cp.Problem(objective, constraints)

    # Замер времени точного решения задачи солвером
    start_time = tm.time()
    problem.solve(solver=solver, verbose=False)
    execution_time = tm.time() - start_time

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Критическая ошибка оптимизации. Статус: {problem.status}")

    # Округление результатов до строгих целых чисел (0 или 1)
    x_opt = np.rint(x.value).astype(int)
    y_opt = np.rint(y.value).astype(int)

    return problem.status, float(problem.value), x_opt, y_opt, execution_time


def process_and_save_results(results_dir, course_ids, student_ids, x_opt, y_opt, capacities, n, m, k, solver, status, obj_value, execution_time):
    """
    ПОДФУНКЦИЯ 2 (ВТОРИЧНАЯ): Постобработка и работа с файловой системой.
    Рассчитывает нагрузки, формирует отчетные таблицы CSV и выводит логи в терминал.
    """
    # Аналитический расчет фактических нагрузок на сущности
    course_loads = x_opt.sum(axis=0)
    student_loads = x_opt.sum(axis=1)

    # 1. Сохранение матрицы распределения студентов
    assignment_df = pd.DataFrame(x_opt, columns=course_ids)
    assignment_df.insert(0, "student_id", student_ids)
    assignment_df.to_csv(results_dir / "centralized_assignment_matrix.csv", index=False)

    # 2. Сохранение итоговой загруженности курсов
    course_load_df = pd.DataFrame({
        "course_id": course_ids,
        "assigned_students_count": course_loads,
        "capacity": capacities,
    })
    course_load_df.to_csv(results_dir / "centralized_course_loads.csv", index=False)

    # 3. Сохранение индивидуальных нагрузок студентов
    student_load_df = pd.DataFrame({
        "student_id": student_ids,
        "assigned_courses": student_loads,
        "max_allowed_courses": k,
        "unassigned_slots": k - student_loads,
    })
    student_load_df.to_csv(results_dir / "centralized_student_loads.csv", index=False)

    # 4. Сохранение сводной метрики эффективности (Z_opt и время)
    objective_df = pd.DataFrame([{
        "Z_opt": obj_value,
        "n_students": n,
        "m_courses": m,
        "k_courses": k,
        "solver": str(solver),
        "status": status,
        "total_assigned": int(x_opt.sum()),
        "total_possible": int(n * k),
        "execution_time_sec": execution_time
    }])
    objective_df.to_csv(results_dir / "centralized_objective_value.csv", index=False)

    # 5. Сохранение реестра открытых / закрытых курсов
    opened_df = pd.DataFrame({
        "course_id": course_ids,
        "y_j": y_opt,
    })
    opened_df.to_csv(results_dir / "centralized_open_courses.csv", index=False)

    # Вывод информационных логов работы в консоль
    print("Centralized optimization completed.")
    print(f"Status: {status}")
    print(f"Solver: {solver}")
    print(f"Z_opt: {obj_value:.4f}")
    print(f"Total assigned slots: {int(x_opt.sum())} / {int(n * k)}")
    print(f"Opened courses: {int(y_opt.sum())} / {m}")


def solve_centralized_optimization():
    """Основная точка входа. Координирует вызовы первичной и вторичной подфункций."""
    print("=== FIRST GAME START (CENTRALIZED OPTIMIZATION) ===")
    ensure_directories()

    # Считывание динамических параметров текущего шага эксперимента
    meta = pd.read_csv(GENERATED_DIR / "meta.csv")
    n = int(meta.loc[0, "n_students"])
    m = int(meta.loc[0, "m_courses"])
    k = int(meta.loc[0, "k_courses_per_student"])

    # Загрузка подготовленных матриц данных
    v, capacities, min_quantities, course_ids, student_ids = load_input_data()
    solver = get_available_solver()

    # ВЫЗОВ ПОДФУНКЦИИ 1: Расчет математического ядра
    status, obj_value, x_opt, y_opt, execution_time = run_optimization_core(
        n, m, k, v, capacities, min_quantities, solver
    )

    # ВЫЗОВ ПОДФУНКЦИИ 2: Запись отчетов и аналитика логов
    process_and_save_results(
        FIRST_GAME_RESULTS_DIR, course_ids, student_ids, x_opt, y_opt, 
        capacities, n, m, k, solver, status, obj_value, execution_time
    )


if __name__ == "__main__":
    solve_centralized_optimization()