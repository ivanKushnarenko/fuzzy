import numpy as np


def universe_diverse(universe) -> bool:
    return len(universe) == len(np.unique(universe))


def probabilities_valid(probabilities) -> bool:
    return (np.all(probabilities >= 0)
            and np.all(probabilities <= 1)
            and not np.isclose(np.sum(probabilities), 0.0))


def probabilities_normalized(probabilities) -> bool:
    return np.isclose(np.sum(probabilities), 1.0)


def normalize_probabilities(probabilities):
    total = np.sum(probabilities)
    return probabilities / total
