import torch
from torch.utils.data import Dataset, DataLoader
import random


# Define a custom dataset class for Quixo game
class QuixoDataset(Dataset):
    def __init__(self):
        # Initialize lists to store game states and corresponding labels
        self.game_states = []  # List to store game states
        self.val_labels = []  # List to store value labels
        self.pol_labels = []  # List to store policy labels

    # Return the length of the dataset
    def __len__(self):
        return len(self.game_states)

    # Add new data to the dataset
    def add_data(self, new_game_states, new_val_labels, new_pol_labels):
        self.game_states.extend(new_game_states)  # Add new game states
        self.val_labels.extend(new_val_labels)  # Add new value labels
        self.pol_labels.extend(new_pol_labels)  # Add new policy labels

    # Add data and halve the existing dataset randomly
    def add_data_and_halve(self, new_game_states, new_val_labels, new_pol_labels):
        # Halve the existing data randomly
        indices = list(range(len(self.game_states)))
        random.shuffle(indices)
        keep_indices = set(indices[: len(indices) // 2])

        self.game_states = [self.game_states[i] for i in keep_indices]
        self.val_labels = [self.val_labels[i] for i in keep_indices]
        self.pol_labels = [self.pol_labels[i] for i in keep_indices]

        self.add_data(new_game_states, new_val_labels, new_pol_labels)

    # Add data and keep a fixed number of existing data randomly
    def add_data_and_keep_fixed(self, new_game_states, new_val_labels, new_pol_labels, fixed_num):
        # Keep a fixed number of existing data randomly
        if len(self.game_states) > fixed_num:
            indices = list(range(len(self.game_states)))
            random.shuffle(indices)
            keep_indices = set(indices[:fixed_num])

            self.game_states = [self.game_states[i] for i in keep_indices]
            self.val_labels = [self.val_labels[i] for i in keep_indices]
            self.pol_labels = [self.pol_labels[i] for i in keep_indices]

        self.add_data(new_game_states, new_val_labels, new_pol_labels)


    # Retrieve an item from the dataset by index
    def __getitem__(self, idx):
        # Access the game state and labels at the specified index
        state = self.game_states[idx]  # Get the game state
        val_label = self.val_labels[idx]  # Get the value label
        pol_label = self.pol_labels[idx]  # Get the policy label

        # Convert them to tensors before returning
        return (
            torch.tensor(state, dtype=torch.float),  # Game state tensor
            torch.tensor(val_label, dtype=torch.float),  # Value label tensor
            torch.tensor(pol_label, dtype=torch.float),  # Policy label tensor
        )


# Function to create a DataLoader for the Quixo dataset
def get_data_loader(dataset, batch_size, shuffle=True):
    # Create and return a DataLoader with the given dataset and batch size
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
