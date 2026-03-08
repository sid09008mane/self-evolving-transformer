import random 
from copy import deepcopy 
from nas.genome import Genome

class MutationEngine:

    def __init__(self):

        #Search Space 
        self.layers_choices = [2, 4, 6, 8]
        self.hidden_choices = [128, 256, 512]
        self.heads_choices = [4, 8]
        self.ff_choices = [256, 512, 1024]
        self.attention_choices = ["standard", "sparse"]
        self.experts_choices = [0, 2, 4]


    #---------------------------
    # Mutation Operations
    #---------------------------

    def mutate_layers(self, genome):

        new_genome = deepcopy(genome)
        new_genome.num_layers = random.choice(self.layers_choices)
        return new_genome

    def mutate_hidden(self, genome):

        new_genome = deepcopy(genome)
        new_genome.hidden_dim = random.choice(self.hidden_choices)
        return new_genome

    def mutate_heads(self, genome):

        new_genome = deepcopy(genome)
        new_genome.num_heads = random.choice(self.heads_choices)
        return new_genome

    def mutate_ff(self, genome):

        new_genome = deepcopy(genome)
        new_genome.ff_dim = random.choice(self.ff_choices)
        return new_genome

    def mutate_attention(self, genome):

        new_genome = deepcopy(genome)
        new_genome.attention_type = random.choice(self.attention_choices)
        return new_genome

    def mutate_experts(self, genome):

        new_genome = deepcopy(genome)
        new_genome.num_experts = random.choice(self.experts_choices)
        return new_genome


    #---------------------------
    # Main Mutation Function
    #---------------------------

    def mutate(self, genome):
        

        mutation_ops = [
            self.mutate_layers,
            self.mutate_hidden,
            self.mutate_heads,
            self.mutate_ff,
            self.mutate_attention,
            self.mutate_experts
        ]

        mutation = random.choice(mutation_ops)

        mutated_genome = mutation(genome)

        return mutated_genome
    