import numpy as np
import skfuzzy as fuzz

from fuzzy_calculator import FuzzyProbabilities


def test_add_event():
    universe = np.array([0, 1, 2, 3, 4])
    probabilities = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    fuzzy_prob = FuzzyProbabilities(universe, probabilities)

    event = np.array([0.2, 0.4, 0.4, 0.0, 0.0])
    event_index = fuzzy_prob.add_event(event)

    assert event_index == 0
    assert np.allclose(fuzzy_prob._events[event_index], event)


def test_probability():
    universe = np.array([0, 1, 2, 3, 4])
    probabilities = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    fuzzy_prob = FuzzyProbabilities(universe, probabilities)

    event = np.array([0.2, 0.4, 0.4, 0.0, 0.0])
    fuzzy_prob.add_event(event)

    prob = fuzzy_prob.probability(event)
    expected_prob = np.dot(event, probabilities)

    assert np.isclose(prob, expected_prob)


def test_events_sum():
    universe = np.array([0, 1, 2, 3, 4])
    probabilities = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    fuzzy_prob = FuzzyProbabilities(universe, probabilities)

    event1 = np.array([0.2, 0.4, 0.4, 0.0, 0.0])
    event2 = np.array([0.1, 0.3, 0.2, 0.3, 0.1])
    fuzzy_prob.add_event(event1)
    fuzzy_prob.add_event(event2)

    sum_event = fuzzy_prob.events_sum(0, 1)
    expected_sum = fuzz.fuzzy_add(universe, event1, universe, event2)[1]

    assert np.allclose(sum_event, expected_sum)


def test_events_intersection():
    universe = np.array([0, 1, 2, 3, 4])
    probabilities = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    fuzzy_prob = FuzzyProbabilities(universe, probabilities)

    event1 = np.array([0.2, 0.4, 0.4, 0.0, 0.0])
    event2 = np.array([0.1, 0.3, 0.2, 0.3, 0.1])
    fuzzy_prob.add_event(event1)
    fuzzy_prob.add_event(event2)

    intersection_event = fuzzy_prob.events_intersection(0, 1)
    expected_intersection = fuzz.fuzzy_and(universe, event1, universe, event2)[1]

    assert np.allclose(intersection_event, expected_intersection)


if __name__ == '__main__':
    test_add_event()
    test_probability()
    test_events_sum()
    test_events_intersection()
    print('All tests passed!')
