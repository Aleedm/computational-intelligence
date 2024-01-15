import torch
import torch.nn as nn
import random
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

class NeuralNetworkWithFixedOutput(nn.Module):
    def __init__(self, genome):
        super(NeuralNetworkWithFixedOutput, self).__init__()
        
        # Creazione dinamica degli strati intermedi in base al genoma
        self.layers = nn.ModuleList()
        for i in range(len(genome)):
            input_size, output_size, weights, bias = genome[i]
            layer = nn.Linear(input_size, output_size)
            
            # Impostazione dei pesi e del bias
            with torch.no_grad():
                layer.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
                layer.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
            
            self.layers.append(layer)

        # Aggiunta dell'ultimo strato fisso per l'output di 44 neuroni
        self.output_layer = nn.Linear(output_size, 44)

    def forward(self, x):
        # Forward pass attraverso ogni strato intermedio
        for layer in self.layers:
            x = torch.relu(layer(x))  # Utilizzo di ReLU come funzione di attivazione

        # Passaggio attraverso lo strato di output e applicazione della Softmax
        x = self.output_layer(x)
        return torch.softmax(x, dim=0)


def generate_random_genome(input_size=25, output_size=44, min_hidden_layers=2, max_hidden_layers=5):
    num_hidden_layers = random.randint(min_hidden_layers, max_hidden_layers)

    # Struttura del genoma: [(input_size, output_size, weights, bias), ...]
    genome = []
    
    # Dimensioni dello strato precedente, inizia con la dimensione dell'input
    previous_layer_size = input_size

    # Generazione casuale degli strati nascosti
    for _ in range(num_hidden_layers):
        current_layer_size = random.randint(1, 10)  # Numero casuale di neuroni nello strato

        # Pesi casuali e bias per lo strato corrente
        weights = [[random.uniform(-1, 1) for _ in range(previous_layer_size)] for _ in range(current_layer_size)]
        bias = [random.uniform(-1, 1) for _ in range(current_layer_size)]

        # Aggiungi lo strato al genoma
        genome.append((previous_layer_size, current_layer_size, weights, bias))

        # Aggiorna la dimensione dello strato precedente
        previous_layer_size = current_layer_size
        
    # Ultimo strato che collega all'output fisso di 44 neuroni
    weights = [[random.uniform(-1, 1) for _ in range(previous_layer_size)] for _ in range(output_size)]
    bias = [random.uniform(-1, 1) for _ in range(output_size)]
    genome.append((previous_layer_size, output_size, weights, bias))

    return genome

# Esempio di generazione di un genoma
random_genome = generate_random_genome()

#cast random genome to dataframe and print the shape
df = pd.DataFrame(random_genome)
df




# # Creazione della rete neurale
# net = NeuralNetworkWithFixedOutput(example_genome)

# # Test con un input casuale
# test_input = torch.rand(25)
# output = net(test_input)

# print(output) 
# print(output.shape)  # Output e la sua lunghezza (dovrebbe essere 44)

# # Inizializzazione di TensorBoard
# writer = SummaryWriter("runs/neural_network_visualization")

# # Aggiunta della rete a TensorBoard
# example_input = torch.rand(1, 3)  # Assicurati che questa dimensione corrisponda all'input della tua rete
# writer.add_graph(net, example_input)
# writer.close()
