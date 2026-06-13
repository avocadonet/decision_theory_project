import numpy as np
import pandas as pd
from itertools import combinations

def run_best_response_dynamics(utility_matrix, capacities, ratings, k_courses, mechanism="rating", max_iter=100):
    n_students, m_courses = utility_matrix.shape
    
    # Инициализация: случайно раздаем каждому студенту K курсов
    current_strategies = np.zeros((n_students, m_courses), dtype=int)
    for i in range(n_students):
        random_courses = np.random.choice(m_courses, k_courses, replace=False)
        current_strategies[i, random_courses] = 1

    # Заранее генерируем все возможные комбинации K курсов из M доступных
    all_course_combinations = list(combinations(range(m_courses), k_courses))
    
    for iteration in range(max_iter):
        strategy_changed = False
        
        # Считаем текущий спрос на курсы
        demand = current_strategies.sum(axis=0)
        
        for i in range(n_students):
            current_choice = current_strategies[i].copy()
            best_utility = -np.inf
            best_choice = current_choice
            
            # Временно убираем студента из общего спроса, чтобы он оценил картину "снаружи"
            demand -= current_choice
            
            # Перебираем все возможные комбинации курсов
            for combo in all_course_combinations:
                expected_utility = 0
                combo_strategy = np.zeros(m_courses, dtype=int)
                combo_strategy[list(combo)] = 1
                
                for j in combo:
                    current_course_demand = demand[j] + 1 # Спрос, если бы студент добавил себя сюда
                    cap = capacities[j]
                    
                    # Считаем вероятность зачисления
                    prob_admission = 0.0
                    if current_course_demand <= cap:
                        prob_admission = 1.0
                    else:
                        if mechanism == "random":
                            # При случайном зачислении шансы равны вместимость / спрос
                            prob_admission = cap / current_course_demand
                        elif mechanism == "rating":
                            # Смотрим на конкурентов, которые УЖЕ хотят на этот курс
                            competitors_mask = current_strategies[:, j] == 1
                            competitors_ratings = ratings[competitors_mask]
                            
                            # Сколько человек имеют рейтинг строго выше нашего студента?
                            higher_rating_count = np.sum(competitors_ratings > ratings[i])
                            
                            # Если мест больше, чем людей с высоким рейтингом, мы проходим
                            if higher_rating_count < cap:
                                prob_admission = 1.0
                            else:
                                prob_admission = 0.0

                    expected_utility += utility_matrix[i, j] * prob_admission
                
                # Если нашли комбинацию выгоднее — запоминаем
                if expected_utility > best_utility:
                    best_utility = expected_utility
                    best_choice = combo_strategy
            
            # Возвращаем студента в спрос с его (возможно новым) выбором
            demand += best_choice
            
            # Проверяем, поменял ли студент свое решение
            if not np.array_equal(current_choice, best_choice):
                current_strategies[i] = best_choice
                strategy_changed = True
                
        print(f"Итерация {iteration + 1}: Изменения в стратегиях = {strategy_changed}")
        
        # Если ни один студент не поменял выбор, равновесие найдено
        if not strategy_changed:
            print(f"Равновесие Нэша найдено за {iteration + 1} итераций!")
            break
            
    # Итоговое распределение
    final_demand = current_strategies.sum(axis=0)
    actual_loads = np.minimum(final_demand, capacities)
    
    return current_strategies, actual_loads