import numpy as np
from scipy.stats import f

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

X = np.hstack([np.ones((len(factorial_plan), 1)), factorial_plan])
# Пошук коефіцієнтів регресії
b = np.linalg.inv(X.T @ X) @ X.T @ y_mean

# Рівняння регресії
equation = f"y = {b[0]:.4f}"
for i in range(1, len(b)):
    equation += f" + ({b[i]:.4f})*x{i}"

# Передбачені значення y
y_pred = X @ b

m = 2
N = len(y_mean)
l = len(b)

S_u_arr = np.array([])
for i in range(N):
    S_u = (1/(m - 1)) * (y1[i]-y_mean[i])**2 + (y2[i]-y_mean[i])**2
    S_u_arr = np.append(S_u_arr, S_u)

S_u_max = np.amax(S_u_arr)

# Критерій Кохрена
# Для чисел ступенів свободи f1 = 1, f2 = 7
G_p = S_u_max/np.sum(S_u_arr)
G_t = 0.7271

if G_p < G_t:
    print("Дисперсія є однорідною.")
else:
    print("Дисперсія НЕ є однорідною.")
    exit(0)

# Помилка досліду
S_0 = np.sum((y1 - y_mean)**2 + (y2 - y_mean)**2) / (N * (m - 1))
# Дисперсія адекватності
S_ad = (m / (N - l)) * np.sum((y_mean - y_pred)**2)

F_exp = S_ad / S_0

# Табличне значення критерію Фішера
# Для чисел ступенів свободи f_1 = 4 f_2 = 15
f_t = 9.15

# Перевірка значущості коефіцієнтів
S_bi = S_0/N
student = 2.36
for i in range(l):
    t_ip = np.abs(b[i])/ S_bi
    if (t_ip < student):
        print(f"{i}-й коефіцієнт регресії при оцінці {t_ip} не є значущим")
        exit(0)
print("Всі коефіцієнти регресії є значущими\n")

print("Середнє значення y:", y_mean)
print("\nКоефіцієнти регресійної моделі:")
for i, coef in enumerate(b):
    print(f"b{i}: {coef:.4f}")
print("\nРівняння регресії:", equation)

print(f"\nТабличне значення критерію Кохрена: {G_p:.4f}")
print(f"Критерій Кохрена: {G_p:.4f}")
print(f"Дисперсія відтворюваності S_0: {S_0:.4f}")
print(f"Дисперсія адекватності S_ad: {S_ad:.4f}")
print(f"Значення критерію Фішера F_exp: {F_exp:.4f}")
print(f"Табличне значення критерію Фішера F_t: {f_t:.4f}")

if F_exp < f_t:
    print("Модель є адекватною.")
else:
    print("Модель НЕ є адекватною.")
