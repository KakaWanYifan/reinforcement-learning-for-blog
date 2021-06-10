from sympy import symbols
from sympy import solve
from sympy import Eq

v_hungry = symbols('v_hungry')
v_full = symbols('v_full')
q_hungry_eat = symbols('q_hungry_eat')
q_hungry_dont_eat = symbols('q_hungry_dont_eat')
q_full_eat = symbols('q_full_eat')
q_full_dont_eat = symbols('q_full_dont_eat')
alpha = symbols('alpha')
beta = symbols('beta')
gamma = symbols('gamma')
x = symbols('x')
y = symbols('y')

ans = solve([Eq((1 - x) * q_hungry_dont_eat + x * q_hungry_eat, v_hungry),
             Eq(y * q_full_dont_eat + (1 - y) * q_full_eat, v_full),
             Eq(-2 + gamma * v_hungry, q_hungry_dont_eat),
             Eq(alpha * (2 + gamma * v_full) + (1 - alpha) * (1 + gamma * v_hungry), q_hungry_eat),
             Eq(beta * (2 + gamma * v_full) + (1 - beta) * (1 + gamma * v_hungry), q_full_dont_eat),
             Eq(gamma * v_full, q_full_eat)],
            [v_hungry, v_full, q_hungry_eat, q_hungry_dont_eat, q_full_eat, q_full_dont_eat])

for k,v in ans.items():
    print(k)
    print(v)