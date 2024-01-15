import torch
import torch.nn as nn
import random
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

class model(nn.Module):
    def __init__(self, genome, output=44):
        super(model, self).__init__()
        
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
        self.output_layer = nn.Linear(output_size, output)

    def forward(self, x):
        # Forward pass attraverso ogni strato intermedio
        for layer in self.layers:
            x = torch.relu(layer(x))  # Utilizzo di ReLU come funzione di attivazione

        # Passaggio attraverso lo strato di output e applicazione della Softmax
        x = self.output_layer(x)
        return torch.softmax(x, dim=0)

