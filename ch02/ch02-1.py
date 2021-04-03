from numpy.random import choice

samples = choice(a=['正', '反'], size=100, p=[0.5, 0.5])
print(samples)
