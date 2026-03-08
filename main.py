from nas.architecture_generator import ArchitectureGenerator
from nas.strategy_controller import StrategyController
from execution.trainer import Trainer


# ------------------------------------------------
# NAS Training Loop
# ------------------------------------------------

def run_nas():

    generations = 10
    population_size = 8

    print("\nStarting Self-Evolving Transformer NAS\n")

    generator = ArchitectureGenerator(population_size=population_size)

    strategy = StrategyController(total_generations=generations)

    trainer = Trainer()

    # initialize population
    population = generator.initialize_population()

    best_genome = None
    best_score = -1

    for generation in range(generations):

        print("\n==============================")
        print("Generation:", generation)
        print("==============================")

        # update NAS strategy
        strategy.update(generation)

        policy = strategy.get_strategy()

        print("Strategy:", policy)

        # evaluate architectures
        for genome in population:

            print("\nTraining genome:", genome)

            fitness = trainer.train_genome(genome)

            genome.fitness = fitness

            print("Fitness:", fitness)

            if fitness > best_score:

                best_score = fitness
                best_genome = genome

        print("\nBest Genome So Far:")
        print(best_genome)
        print("Best Score:", best_score)

        # evolve population
        population = generator.generate_next_population(population)

    print("\n==============================")
    print("NAS Search Complete")
    print("==============================")

    print("\nFinal Best Architecture:")
    print(best_genome)
    print("Fitness:", best_score)


# ------------------------------------------------
# Entry Point
# ------------------------------------------------

def main():

    run_nas()


if __name__ == "__main__":

    main()
