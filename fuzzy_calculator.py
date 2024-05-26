import math

import numpy as np
import skfuzzy as fuzz

import utils


class FuzzyProbabilitiesCalculator:
    def __init__(self, universe, probabilities, *, normalize=False):
        """
        Initialize fuzzy probabilities' calculator.

        :param universe: 1d array, length N
            The universe, which represents all possible elementary events.
        :param probabilities: 1d array, length N
            The initial probabilities associated with each event in the universe.
        :param normalize: bool, optional (default=False)
            Flag to indicate whether to normalize probabilities to ensure their sum equals 1.
        """
        if not utils.universe_diverse(universe):
            raise ValueError("Every elementary event should be different")
        if not utils.probabilities_valid(probabilities):
            raise ValueError("All probabilities should be in [0,1] interval and their sum cannot be zero")
        if not utils.probabilities_normalized(probabilities):
            if not normalize:
                raise ValueError("Sum of all probabilities values should be equal to 1")
            else:
                self._probabilities = utils.normalize_probabilities(probabilities)
        else:
            self._probabilities = probabilities
        self._universe = universe
        self._events = []
        self._combined_events = []

    def _normalized_event(self, event):
        """
        Normalize the length of the event array to match the universe length.

        :param event: 1d array
            The event membership array to be normalized.
        :return: 1d array
            The normalized event array.
        """
        size_diff = len(self._universe) - len(event)
        if size_diff > 0:
            return np.pad(event, (0, size_diff), constant_values=0.0)
        elif size_diff < 0:
            return event[:size_diff]
        return event

    def add_event(self, membership) -> int:
        """
        Add a new fuzzy event to the collection of events.

        :param membership: 1d array
            The membership values of the fuzzy event.
        :return: int
            The index of the added event.

        Example:
        >>> fuzzy_prob = FuzzyProbabilities(np.array([0, 1, 2]), np.array([0.3, 0.4, 0.3]))
        >>> event = np.array([0.2, 0.5, 0.3])
        >>> event_index = fuzzy_prob.add_event(event)
        >>> print(event_index)  # Output: 0
        """
        normalized = self._normalized_event(membership)
        self._events.append(normalized)
        return len(self._events) - 1

    def events_sum(self, *events_indices):
        """
        Calculate the fuzzy sum of multiple events.

        :param events_indices: int
            The indices of events to be summed.
        :return: 1d array
            The membership values of the summed event.

        Example:
        >>> fuzzy_prob = FuzzyProbabilities(np.array([0, 1, 2]), np.array([0.3, 0.4, 0.3]))
        >>> event1 = np.array([0.2, 0.5, 0.3])
        >>> event2 = np.array([0.1, 0.6, 0.3])
        >>> fuzzy_prob.add_event(event1)
        >>> fuzzy_prob.add_event(event2)
        >>> sum_event = fuzzy_prob.events_sum(0, 1)
        >>> print(sum_event)  # Output: array with the summed membership values
        """
        if not events_indices:
            raise ValueError("At least one event index must be provided")

        selected_events = [self._events[idx] for idx in events_indices if idx < len(self._events)]

        if len(selected_events) == 1:
            return selected_events[0]

        result = selected_events[0]
        for event in selected_events[1:]:
            result = fuzz.fuzzy_add(self._universe, result, self._universe, event)[1]

        return result

    def events_intersection(self, *events_indices):
        """
        Calculate the fuzzy intersection of multiple events.

        :param events_indices: int
            The indices of events to be intersected.
        :return: 1d array
            The membership values of the intersected event.

        Example:
        >>> fuzzy_prob = FuzzyProbabilities(np.array([0, 1, 2]), np.array([0.3, 0.4, 0.3]))
        >>> event1 = np.array([0.2, 0.5, 0.3])
        >>> event2 = np.array([0.1, 0.6, 0.3])
        >>> fuzzy_prob.add_event(event1)
        >>> fuzzy_prob.add_event(event2)
        >>> intersection_event = fuzzy_prob.events_intersection(0, 1)
        >>> print(intersection_event)  # Output: array with the intersected membership values
        """
        if not events_indices:
            raise ValueError("At least one event index must be provided")

        selected_events = [self._events[idx] for idx in events_indices if idx < len(self._events)]

        if len(selected_events) == 1:
            return selected_events[0]

        result = selected_events[0]
        for event in selected_events[1:]:
            result = fuzz.fuzzy_and(self._universe, result, self._universe, event)[1]

        return result

    def probability(self, event):
        """
        Calculate the probability of a given fuzzy event.

        :param event: 1d array
            The membership values of the fuzzy event.
        :return: float
            The probability of the event.

        Example:
        >>> fuzzy_prob = FuzzyProbabilities(np.array([0, 1, 2]), np.array([0.3, 0.4, 0.3]))
        >>> event = np.array([0.2, 0.5, 0.3])
        >>> prob = fuzzy_prob.probability(event)
        >>> print(prob)  # Output: probability value
        """
        normalized = self._normalized_event(event)
        return np.dot(event, self._probabilities)

    def fuzzy_probability(self, event):
        """
        Calculate the fuzzy probability of a given fuzzy event.

        :param event: 1d array
            The membership values of the fuzzy event.
        :return: tuple
            A tuple containing an array of probabilities and an array of corresponding alpha levels.

        Example:
        >>> fuzzy_prob = FuzzyProbabilities(np.array([0, 1, 2]), np.array([0.3, 0.4, 0.3]))
        >>> event = np.array([0.2, 0.5, 0.3])
        >>> fuzzy_probs, alphas = fuzzy_prob.fuzzy_probability(event)
        >>> print(fuzzy_probs, alphas)  # Output: arrays of probabilities and alpha levels
        """
        normalized = self._normalized_event(event)
        alphas = np.trim_zeros(np.unique(normalized))
        probs = []
        for a in alphas:
            indices, = np.nonzero(normalized >= a)
            probs.append(np.sum(self._probabilities[indices]))
        return np.array(probs), alphas

    def bernoulli(self, event, success: int, failure: int):
        """
        Calculate the Bernoulli probability of a given fuzzy event.

        :param event: 1d array
            The membership values of the fuzzy event.
        :param success: int
            The number of successes.
        :param failure: int
            The number of failures.
        :return: float
            The Bernoulli probability of the event.

        Example:
        >>> fuzzy_prob = FuzzyProbabilities(np.array([0, 1, 2]), np.array([0.3, 0.4, 0.3]))
        >>> event = np.array([0.2, 0.5, 0.3])
        >>> bernoulli_prob = fuzzy_prob.bernoulli(event, 3, 2)
        >>> print(bernoulli_prob)  # Output: Bernoulli probability value
        """
        normalized = self._normalized_event(event)
        success_prob = self.probability(normalized)
        failure_prob = self.probability(fuzz.fuzzy_not(normalized))
        return math.comb(success + failure, success) * success_prob ** success * failure_prob ** failure
