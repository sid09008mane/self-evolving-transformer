class ArchitectureMemory:

    def __init__(self):

        self.memory = []

    def add(self, genome, score):

        self.memory.append((genome, score))

    def best(self):

        return sorted(self.memory, key=lambda x: x[1], reverse=True)[0]
