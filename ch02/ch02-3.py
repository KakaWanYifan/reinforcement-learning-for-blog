from sympy import symbols
from sympy import solve
from sympy import Eq
from sympy import maximum

v_hungry = symbols('v_hungry')
v_full = symbols('v_full')
q_hungry_eat = symbols('q_hungry_eat')
q_hungry_dont_eat = symbols('q_hungry_dont_eat')
q_full_eat = symbols('q_full_eat')
q_full_dont_eat = symbols('q_full_dont_eat')
alpha = 2.0/3.0
beta = 3.0/4.0
gamma = 4.0/5.0

ans = solve([Eq(maximum(q_hungry_eat, q_hungry_dont_eat), v_hungry),
             Eq(maximum(q_full_eat, q_full_dont_eat), v_full),
             Eq(-2 + gamma * v_hungry, q_hungry_dont_eat),
             Eq(alpha * (2 + gamma * v_full) + (1 - alpha) * (1 + gamma * v_hungry), q_hungry_eat),
             Eq(beta * (2 + gamma * v_full) + (1 - beta) * (1 + gamma * v_hungry), q_full_dont_eat),
             Eq(gamma * v_full, q_full_eat)],
            [v_hungry, v_full, q_hungry_eat, q_hungry_dont_eat, q_full_eat, q_full_dont_eat])

for k, v in ans.items():
    print(k)
    print(v)
