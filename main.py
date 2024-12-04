import numpy as np

# Початкові дані
x1_0, x2_0, x3_0 = 4, 1.75, 380
x1_delta, x2_delta, x3_delta = 1, 0.25, 20

factorial_plan = np.array([
    [-1, -1, -1],
    [+1, -1, -1],
    [-1, +1, -1],
    [+1, +1, -1],
    [-1, -1, +1],
    [+1, -1, +1],
    [-1, +1, +1],
    [+1, +1, +1],
])

y1 = np.array([-0.5673, -0.6888, -0.7090, -0.4870, -0.6790, -0.4875, -0.8650, 0.7890])
y2 = np.array([-0.6723, -0.5460, -0.5090, -0.4170, -0.7550, -0.8875, -1.0650, 0.1330])

# Пошук середнього
y_mean = (y1 + y2) / 2

x1 = x1_0 + x1_delta * factorial_plan[:, 0]
x2 = x2_0 + x2_delta * factorial_plan[:, 1]
x3 = x3_0 + x3_delta * factorial_plan[:, 2]

# Формуємо матрицю X
X = np.hstack([np.ones((len(factorial_plan), 1)), factorial_plan])

# Обчислення коефіцієнтів регресії
b = np.linalg.inv(X.T @ X) @ X.T @ y_mean

# Кількість паралельних експериментів (y)
m = 2
# Кількість експериментів
N = len(y_mean)
# Кількість коефіцієнтів регресії
l = len(b)

# Оцінка дисперсії відтворюваності
# Критерій Кохрена
S_u_arr = np.array([])
for i in range(N):
    S_u = (1 / (m - 1)) * ((y1[i] - y_mean[i])**2 + (y2[i] - y_mean[i])**2)
    S_u_arr = np.append(S_u_arr, S_u)

S_u_max = np.amax(S_u_arr)
G_p = S_u_max / np.sum(S_u_arr)
G_t = 0.7271

if G_p < G_t:
    print("Дисперсія є однорідною.")
else:
    print("Дисперсія НЕ є однорідною.")
    exit(0)

# Помилка досліду
S_0 = np.sum((y1 - y_mean)**2 + (y2 - y_mean)**2) / (N * (m - 1))

# Перевірка значущості коефіцієнтів за t-критерієм Стьюдента
S_bi = S_0 / N
student_t = 2.36  # Табличне значення
significant_factors = []

for i in range(l):
    t_ip = np.abs(b[i]) / np.sqrt(S_bi)
    if t_ip >= student_t:
        significant_factors.append(i)
    else:
        print(f"Коефіцієнт b{i} ({b[i]:.4f}) незначущий при t={t_ip:.4f}")

# Оновлюємо матрицю X для значущих коефіцієнтів
X_significant = X[:, significant_factors]

# Повторна регресія
b_significant = np.linalg.inv(X_significant.T @ X_significant) @ X_significant.T @ y_mean

# Передбачені значення y
y_pred_significant = X_significant @ b_significant

# Дисперсія адекватності
S_ad = (m / (N - len(b_significant))) * np.sum((y_mean - y_pred_significant)**2)

# Критерій Фішера
F_exp = S_ad / S_0
f_t = 9.15  # Табличне значення

# Вивід результатів
print("\nКоефіцієнти регресійної моделі після фільтрації:")
for i, coef in zip(significant_factors, b_significant):
    print(f"b{i}: {coef:.4f}")

print(f"\nРівняння регресії:")
equation = "y = " + " + ".join([f"({coef:.4f})*x{i}" if i != 0 else f"{coef:.4f}" 
                                for i, coef in zip(significant_factors, b_significant)])
print(equation)

print(f"\nКритерій Фішера F_exp: {F_exp:.4f}")
print(f"Табличне значення F_t: {f_t:.4f}")

if F_exp < f_t:
    print("Модель є адекватною.")
else:
    print("Модель НЕ є адекватною.")
