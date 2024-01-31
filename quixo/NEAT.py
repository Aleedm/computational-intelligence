import random
import torch
import torch.nn as nn
from utilities import convert_board_to_binary, get_possible_moves, move_to_index
from game_wrapper import GameWrapper as Game
from tqdm import tqdm
from game import Player, Move

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# The model always play with the X, if the model is the O, the board is rotated
class Model(nn.Module):
    def __init__(self, genome):
        super(Model, self).__init__()

        # Create dinamically the intermediate layers based on the genome
        self.layers = nn.ModuleList()
        for i in range(len(genome)):
            input_size, output_size, weights, bias = genome[i]
            layer = nn.Linear(input_size, output_size)

            # Set weights and bias
            with torch.no_grad():
                layer.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
                layer.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

            self.layers.append(layer)

    def forward(self, x):
        # Forward pass through each intermediate layer
        for layer in self.layers:
            # Apply ReLU as activation function
            x = torch.relu(layer(x))

        # Apply softmax to the output layer
        return torch.softmax(x, dim=0)


class Individual:
    def __init__(self, genome, fitness=0):
        # Genome structure: [(input_size, output_size, weights, bias), ...]
        self.genome = genome
        self.fitness = fitness
        self.model = Model(genome).to(device)
        self.model.eval()
    
    def choose_action(self, board, player):
        binary_board = convert_board_to_binary(board, player)[0]
        tensor_board = torch.tensor(binary_board.flatten(), dtype=torch.float32).to(device)
        
        # Get the probabilities for each action
        probabilities = self.model(tensor_board)
        
        # Get the possible moves
        possible_moves = get_possible_moves(board, player)
        possible_moves_idxs = [move_to_index(move) for move in possible_moves]
        
        # Get the probabilities for the possible moves
        possible_moves_probabilities = probabilities[possible_moves_idxs]

        # Choose the action with the highest probability
        action_idx = torch.argmax(possible_moves_probabilities).item()
        action = possible_moves[action_idx]
        return action
          
    def update_fitness(self, fitness):
        self.fitness = fitness
        
    def get_genome(self):
        return self.genome
    
    def get_fitness(self):
        return self.fitness
    
    def __str__(self):
        return f"Genome: {self.genome}\nFitness: {self.fitness}"
    
class Population:
    def __init__(self, population_size, mutation_rate, elitism_rate, tournament_size = 5, population=None, input_size=25, output_size=44, max_hidden_layers=10, max_neurons_per_layer=10, max_moves_per_game=100):
        # Initialization method for the Population class
        self.population_size = population_size  # Total number of individuals in the population
        self.mutation_rate = mutation_rate      # Probability of mutation
        self.elitism_rate = elitism_rate        # Proportion of the population to retain as elites
        self.input_size = input_size            # Size of the input layer in the neural network
        self.output_size = output_size          # Size of the output layer
        self.population = []                    # List to store individuals in the population
        self.generation = 0                     # Counter for generations
        self.best_individual = None             # Best individual in terms of fitness
        self.best_fitness = 0                   # Fitness score of the best individual
        self.fitness_history = []               # Records the fitness over generations
        self.max_hidden_layers = max_hidden_layers        # Maximum number of hidden layers in the neural network
        self.max_neurons_per_layer = max_neurons_per_layer# Maximum number of neurons in each hidden layer
        self.max_moves_per_game = max_moves_per_game      # Maximum number of moves allowed in a game
        self.tournament_size = tournament_size  # Number of individuals participating in each tournament

        # Initialize the population
        if population is not None:
            self.population = population
        else:
            self.create_initial_population()     # Method to create the initial population
  
    def create_initial_population(self):
        # Creates the initial population by generating individuals
        for _ in range(self.population_size):
            genome = self.create_genome()       # Generate a genome for each individual
            individual = Individual(genome)     # Create an individual with the generated genome
            self.population.append(individual)  # Add the individual to the population

    def create_genome(self):
        # Generates a random genome for an individual
        genome = []
        num_hidden_layers = random.randint(1, self.max_hidden_layers)  # Randomly determine the number of hidden layers

        # First layer that connects to the input of 25 neurons
        previous_layer_size = self.input_size

        # Create hidden layers with random sizes and weights
        for _ in range(num_hidden_layers):
            current_layer_size = random.randint(1, self.max_neurons_per_layer)  # Size of the current layer
            # Random weights and biases for the current layer
            weights = [[random.uniform(-1, 1) for _ in range(previous_layer_size)] for _ in range(current_layer_size)]
            bias = [random.uniform(-1, 1) for _ in range(current_layer_size)]
            # Add the layer to the genome
            genome.append((previous_layer_size, current_layer_size, weights, bias))
            previous_layer_size = current_layer_size

        # Add the output layer connecting to 44 neurons
        weights = [[random.uniform(-1, 1) for _ in range(previous_layer_size)] for _ in range(self.output_size)]
        bias = [random.uniform(-1, 1) for _ in range(self.output_size)]
        genome.append((previous_layer_size, self.output_size, weights, bias))

        return genome

    def simulate(self, num_generations, num_games_per_individual):
        # Simulates the evolution of the population over a number of generations
        for _ in tqdm(range(num_generations), desc="Generation"):
            self.run_generation(num_games_per_individual)  # Run each generation

    def run_generation(self, num_games_per_individual):
        # Executes the simulation for one generation
        for individual in self.population:
            fitness = self.run_games(individual, num_games_per_individual)  # Assess the fitness of each individual
            individual.update_fitness(fitness)  # Update individual's fitness

        # Sort the population based on fitness and update the best individual
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.population[0].fitness > self.best_fitness:
            self.best_individual = self.population[0]
            self.best_fitness = self.population[0].fitness
        self.fitness_history.append(self.best_fitness)  # Record fitness history

        self.next_generation()  # Create the next generation
        self.generation += 1    # Increment generation counter

    def run_games(self, individual, num_games_per_individual):
        # Simulates games for each individual to determine fitness
        fitness = 0
        for i in range(num_games_per_individual):
            # Randomly determine the starting player
            if i % 2 == 0:
                players = [individual, random.choice(self.population)]
                individual_start = 0  # Individual starts first
            else:
                players = [random.choice(self.population), individual]
                individual_start = 1  # Individual starts second

            game = Game()  # Initialize a new game
            n_moves = 0
            current_player = 0

            # Play the game until a winner is found or maximum moves are reached
            while game.check_winner() == -1 and n_moves < self.max_moves_per_game:
                move = players[current_player].choose_action(game.get_board(), current_player)
                game.move(move[0], move[1], current_player)
                current_player = (current_player + 1) % 2
                n_moves += 1

            winner = game.check_winner()
            # Increase fitness if the individual won
            if (winner == 0 and individual_start == 0) or (winner == 1 and individual_start == 1):
                fitness += 1

        return fitness

    def next_generation(self):
        # Generates the next generation of individuals
        next_generation = []

        # Retain the best individuals (elitism)
        num_elites = int(self.population_size * self.elitism_rate)
        next_generation.extend(self.population[:num_elites])
        for elite in next_generation:
            elite.update_fitness(0)

        # Generate the rest of the population through crossover and mutation
        while len(next_generation) < self.population_size:
            parent1 = self.select_individual()  # Select two parents
            parent2 = self.select_individual()
            child_genome = self.crossover(parent1.genome, parent2.genome)  # Crossover the parents
            self.mutate(child_genome, self.mutation_rate)  # Mutate the child
            child = Individual(child_genome)  # Create a new individual
            next_generation.append(child)  # Add the child to the next generation

        self.population = next_generation  # Update the population

    def select_individual(self):
        # Selects an individual for reproduction using tournament selection
        tournament = random.sample(self.population, self.tournament_size)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0]

    def crossover(self, G1, G2):
        # Combines two genomes (G1 and G2) to produce a new genome
        child_genome = []

        # Copy input and hidden layers from the parents
        remaining_layers_G1 = G1[1:-1]  # Excluding input and output layers
        remaining_layers_G2 = G2[1:-1]

        # Choose the first layer randomly from one of the two genomes
        first_layer = G1[0] if random.random() < 0.5 else G2[0]
        child_genome.append(first_layer)
        last_output_size = first_layer[1]

        # Decide the number of layers for the child genome
        min_layers = min(len(G1), len(G2)) - 2
        max_layers = min(max(len(remaining_layers_G1), len(remaining_layers_G2)), self.max_hidden_layers)
        layers = random.randint(min_layers, max_layers)

        # Randomly choose layers from parents and add them to the child genome
        for _ in range(layers):
            if (remaining_layers_G1 and random.random() < 0.5) or not remaining_layers_G2:
                chosen_layer = remaining_layers_G1.pop(0)
            else:
                chosen_layer = remaining_layers_G2.pop(0)

            # Adjust weights and biases if necessary
            input_size, output_size, weights, bias = chosen_layer
            if input_size != last_output_size:
                # Modify weights and biases to match the layer sizes
                weights = [[random.uniform(-1, 1) if j >= len(weights[0]) else weights[i][j] for j in range(last_output_size)] for i in range(output_size)]
                bias = bias[:output_size]

            child_genome.append((last_output_size, output_size, weights, bias))
            last_output_size = output_size

            if not remaining_layers_G1 and not remaining_layers_G2:
                break

        # Add the output layer
        child_genome.append((last_output_size, 44, [[random.uniform(-1, 1) for _ in range(last_output_size)] for _ in range(44)], [random.uniform(-1, 1) for _ in range(44)]))

        return child_genome

    def mutate(self, genome, mutation_rate=0.5):
        # Mutates the genome based on a given mutation rate
        # Mutate the number of layers
        if random.random() < mutation_rate:
            # Add or remove a layer
            if len(genome) == self.max_hidden_layers or random.random() < 0.5 and len(genome) > 2:
                # Remove a layer
                layer_to_remove = random.randint(1, len(genome) - 2)
                genome.pop(layer_to_remove)

                # Update the next layer
                if layer_to_remove < len(genome):
                    next_layer_input_size = genome[layer_to_remove - 1][1]
                    next_layer_output_size = genome[layer_to_remove][1]
                    next_layer_weights = [[random.uniform(-1, 1) for _ in range(next_layer_input_size)] for _ in range(next_layer_output_size)]
                    genome[layer_to_remove] = (next_layer_input_size, next_layer_output_size, next_layer_weights, genome[layer_to_remove][3])
            else:
                # Add a layer
                layer_to_add = random.randint(1, len(genome) - 1)
                input_size = genome[layer_to_add - 1][1] 
                output_size = random.randint(1, self.max_neurons_per_layer)
                weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]
                bias = [random.uniform(-1, 1) for _ in range(output_size)]
                genome.insert(layer_to_add, (input_size, output_size, weights, bias))
                
                # Update the next layer
                next_layer = layer_to_add + 1
                if next_layer < len(genome):
                    next_layer_weights = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(genome[next_layer][1])]
                    genome[next_layer] = (output_size, genome[next_layer][1], next_layer_weights, genome[next_layer][3])
        
        # Mutate the size of a random layer
        if len(genome) == 2:
            # Only the output layer can be mutated
            if random.random() < mutation_rate:
                new_size = random.randint(1, self.max_neurons_per_layer)
                intermediate_layer = 0  
                modified_weights = [[random.uniform(-1, 1) for _ in range(25)] for _ in range(new_size)]
                modified_bias = [random.uniform(-1, 1) for _ in range(new_size)]
                genome[intermediate_layer] = (25, new_size, modified_weights, modified_bias)
                # Modify the output layer
                output_layer_weights = [[random.uniform(-1, 1) for _ in range(new_size)] for _ in range(44)]
                genome[1] = (new_size, 44, output_layer_weights, genome[1][3])
        # If there are more than 2 layers, choose a random layer to mutate
        else:
            # Choose a random layer to mutate
            layer_to_modify = random.randint(1, len(genome) - 2)
            new_size = random.randint(1, self.max_neurons_per_layer)
            modified_weights = [[random.uniform(-1, 1) for _ in range(genome[layer_to_modify-1][1])] for _ in range(new_size)]
            modified_bias = [random.uniform(-1, 1) for _ in range(new_size)]
            genome[layer_to_modify] = (genome[layer_to_modify-1][1], new_size, modified_weights, modified_bias)

            # Modify the next layer
            next_layer = layer_to_modify + 1
            if next_layer < len(genome):
                next_layer_weights = [[random.uniform(-1, 1) for _ in range(new_size)] for _ in range(genome[next_layer][1])]
                genome[next_layer] = (new_size, genome[next_layer][1], next_layer_weights, genome[next_layer][3])

        # Mutate the weights and bias of each layer
        for layer in genome:
            for i in range(len(layer[2])): 
                for j in range(len(layer[2][i])):
                    if random.random() < mutation_rate: # Mutate the weight
                        layer[2][i][j] += random.uniform(-0.1, 0.1)
            for i in range(len(layer[3])): 
                if random.random() < mutation_rate: # Mutate the bias
                    layer[3][i] += random.uniform(-0.1, 0.1)

        return genome

    def get_best_individual(self):
        # Returns the best individual in the population
        return self.best_individual
    
    def get_best_fitness(self):
        # Returns the fitness of the best individual
        return self.best_fitness
    
    def get_fitness_history(self):
        # Returns the fitness history of the population over generations
        return self.fitness_history
    
class NeatPlayer(Player):
    def __init__(self, genome) -> None:
        super().__init__()
        self.individual = Individual(genome)

    def make_move(self, game: "Game") -> tuple[tuple[int, int], Move]:
        board = game.get_board()
        player = game.get_current_player()
        move = self.individual.choose_action(board, player)
        return move
        
 
 
 
 
 