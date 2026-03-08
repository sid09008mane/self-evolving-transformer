class StrategyController:
    """
    Controls the NAS search strategy across generations.
    Adjusts mutation rate, crossover rate, and elite fraction.
    """

    def __init__(self, total_generations):

        self.total_generations = total_generations

        self.mutation_rate = 0.5
        self.crossover_rate = 0.5
        self.elite_fraction = 0.3

    # ------------------------------------------------
    # Update strategy based on generation progress
    # ------------------------------------------------

    def update(self, generation):

        progress = generation / self.total_generations

        # Early stage → explore more
        if progress < 0.3:

            self.mutation_rate = 0.7
            self.crossover_rate = 0.3
            self.elite_fraction = 0.2

        # Middle stage → balanced search
        elif progress < 0.7:

            self.mutation_rate = 0.5
            self.crossover_rate = 0.5
            self.elite_fraction = 0.3

        # Late stage → exploit best architectures
        else:

            self.mutation_rate = 0.3
            self.crossover_rate = 0.7
            self.elite_fraction = 0.4

    # ------------------------------------------------
    # Return current search strategy
    # ------------------------------------------------

    def get_strategy(self):

        return {
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elite_fraction": self.elite_fraction
        }