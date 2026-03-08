import random
from copy import deepcopy

from nas.genome import random_genome
from nas.mutation_engine import MutationEngine


class ArchitectureGenerator:

    def __init__(self, population_size=10, elite_fraction=0.3):

        self.population_size = population_size
        self.elite_fraction = elite_fraction

        self.mutation_engine = MutationEngine()

    # ------------------------------------------------
    # Initialize Population
    # ------------------------------------------------

    def initialize_population(self):

        population = []

        for _ in range(self.population_size):

            genome = random_genome()

            population.append(genome)

        return population

    # ------------------------------------------------
    # Select Best Architectures
    # ------------------------------------------------

    def select_elite(self, population):

        population = sorted(
            population,
            key=lambda g: g.fitness if g.fitness is not None else -1,
            reverse=True
        )

        elite_count = max(1, int(self.population_size * self.elite_fraction))

        return population[:elite_count]

    # ------------------------------------------------
    # Crossover
    # ------------------------------------------------

    def crossover(self, parent1, parent2):

        child = deepcopy(parent1)

        if random.random() < 0.5:
            child.num_layers = parent2.num_layers

        if random.random() < 0.5:
            child.hidden_dim = parent2.hidden_dim

        if random.random() < 0.5:
            child.num_heads = parent2.num_heads

        if random.random() < 0.5:
            child.ff_dim = parent2.ff_dim

        if random.random() < 0.5:
            child.attention_type = parent2.attention_type

        if random.random() < 0.5:
            child.num_experts = parent2.num_experts

        child.fitness = None

        return child

    # ------------------------------------------------
    # Generate Next Population
    # ------------------------------------------------

    def generate_next_population(self, population):

        elite = self.select_elite(population)

        next_population = elite.copy()

        while len(next_population) < self.population_size:

            parent1 = random.choice(elite)
            parent2 = random.choice(elite)

            child = self.crossover(parent1, parent2)

            child = self.mutation_engine.mutate(child)

            next_population.append(child)

        return next_population