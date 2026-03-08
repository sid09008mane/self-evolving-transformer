import random

class Genome:
    def __init__(
        self,
        num_layers,
        hidden_dim,
        num_heads,
        ff_dim,
        attention_type,
        num_experts,
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.attention_type = attention_type
        self.num_experts = num_experts
        
        self.fitness = None

    def to_dict(self):


        return {
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "attention_type": self.attention_type,
            "num_experts": self.num_experts,
            "fitness": self.fitness,
        }

    def __str__(self):

        return (
            f"Genome("
            f"layers={self.num_layers}, "
            f"hidden={self.hidden_dim}, "
            f"heads={self.num_heads}, "
            f"ff={self.ff_dim}, "
            f"attention={self.attention_type}, "
            f"experts={self.num_experts}, "
            f"fitness={self.fitness}"
            f")"
        )

def random_genome():

    layers_choices = [2, 4, 6, 8]
    hidden_choices = [128, 256, 512]
    heads_choices = [4, 8]
    ff_choices = [256, 512, 1024]
    attention_choices = ["standard", "sparse"]
    experts_choices = [0, 2, 4]

    genome = Genome(
        num_layers=random.choice(layers_choices),
        hidden_dim=random.choice(hidden_choices),
        num_heads=random.choice(heads_choices),
        ff_dim=random.choice(ff_choices),
        attention_type=random.choice(attention_choices),
        num_experts=random.choice(experts_choices),
    )

    return genome