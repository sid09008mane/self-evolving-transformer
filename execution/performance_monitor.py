import json
import time
import os


class PerformanceMonitor:

    def __init__(self, log_dir="logs"):

        self.log_dir = log_dir

        os.makedirs(log_dir, exist_ok=True)

        self.history = []

        self.best_genome = None
        self.best_score = -1

        self.generation_start_time = None

    # ------------------------------------------------
    # Start generation timer
    # ------------------------------------------------

    def start_generation(self):

        self.generation_start_time = time.time()

    # ------------------------------------------------
    # Update fitness results
    # ------------------------------------------------

    def update(self, genome):

        if genome.fitness > self.best_score:

            self.best_score = genome.fitness
            self.best_genome = genome

    # ------------------------------------------------
    # Finish generation
    # ------------------------------------------------

    def end_generation(self, generation, population):

        generation_time = time.time() - self.generation_start_time

        avg_fitness = sum(g.fitness for g in population) / len(population)

        record = {
            "generation": generation,
            "avg_fitness": avg_fitness,
            "best_fitness": self.best_score,
            "generation_time": generation_time
        }

        self.history.append(record)

        print("\nGeneration Summary")
        print("-------------------")
        print("Generation:", generation)
        print("Average Fitness:", round(avg_fitness, 4))
        print("Best Fitness:", round(self.best_score, 4))
        print("Generation Time:", round(generation_time, 2), "seconds")

        self.save_logs()

    # ------------------------------------------------
    # Save logs
    # ------------------------------------------------

    def save_logs(self):

        history_path = os.path.join(self.log_dir, "nas_history.json")

        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=4)

        if self.best_genome is not None:

            best_path = os.path.join(self.log_dir, "best_architecture.json")

            with open(best_path, "w") as f:

                json.dump(self.best_genome.to_dict(), f, indent=4)

    # ------------------------------------------------
    # Return best genome
    # ------------------------------------------------

    def get_best(self):

        return self.best_genome, self.best_score
