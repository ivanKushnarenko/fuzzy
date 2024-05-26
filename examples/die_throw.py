import numpy as np

import fuzzy_calculator as fc


calc = fc.FuzzyProbabilities(np.arange(1, 7), np.full((6,), 1/6))

# number grater than 2
event1 = np.array([0, 0, 1, 1, 1, 1])
prob1 = calc.probability(event1)
print(f"Probability to get a number greater than 2 is {prob1}")
bernoulli_prob1 = calc.bernoulli(event1, 2, 1)
print(f"Probability to get a number greater than 2 2 times out of 3 is {bernoulli_prob1}")


event2 = np.array([0, 0, 0.1, 0.3, 0.7, 1])
prob2 = calc.probability(event2)
print(f'Probability to get a BIG NUMBER is {prob2}')