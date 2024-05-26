import numpy as np

from fuzzy_calculator import FuzzyProbabilities


# Possible profit levels
universe = np.array([0, 5, 10, 15, 20])

# Probabilities to get given desired level of profit
probabilities = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

# Initialization of fuzzy probabilities calculator
fuzzy_prob = FuzzyProbabilities(universe, probabilities, normalize=True)

# Define different profit outcome
event_low = np.array([1.0, 0.5, 0.0, 0.0, 0.0])   # Low
event_medium = np.array([0.0, 0.5, 1.0, 0.5, 0.0])  # Medium
event_high = np.array([0.0, 0.0, 0.0, 0.5, 1.0])   # High

# Add desired events to calculator
event_low_index = fuzzy_prob.add_event(event_low)
event_medium_index = fuzzy_prob.add_event(event_medium)
event_high_index = fuzzy_prob.add_event(event_high)

# Find probabilities of gaining different profit levels
prob_low = fuzzy_prob.probability(event_low)
prob_medium = fuzzy_prob.probability(event_medium)
prob_high = fuzzy_prob.probability(event_high)

print(f"Low profit probability: {prob_low:.2f}")
print(f"Medium profit probability: {prob_medium:.2f}")
print(f"High profit probability: {prob_high:.2f}")

# Fuzzy probability of getting medium level profit
fuzzy_probs, alphas = fuzzy_prob.fuzzy_probability(event_medium)
print(f"Medium profit fuzzy probability: {fuzzy_probs}, μ = {alphas}")

# Define event of high or medium profit level
sum_event = fuzzy_prob.events_sum(event_medium_index, event_high_index)
print(f"Membership values of getting high or medium profit: {sum_event}")
