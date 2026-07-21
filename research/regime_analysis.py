"""Extended regime transition analysis."""

class RegimeAnalysis:
    """Analyzes the transition matrix and state durations of the HMM."""
    
    def calculate_expected_duration(self, transition_matrix) -> list:
        """Uses markov mathematics to compute average expected time in a state."""
        # E[T] = 1 / (1 - P(state -> state))
        try:
            return [1 / (1 - transition_matrix[i][i]) for i in range(len(transition_matrix))]
        except Exception:
            return [0.0]
