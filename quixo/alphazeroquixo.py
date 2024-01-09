import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Determine if CUDA is available and use it for training, else use CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# # Define the neural network class
# class QuixoNet(nn.Module):
#     def __init__(self):
#         super(QuixoNet, self).__init__()
#         # Define the convolutional layers with batch normalization
#         self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(512)

#         # Define fully connected layers for processing after convolutional layers
#         self.fc1 = nn.Linear(512 * 5 * 5, 1024)
#         self.bn5 = nn.BatchNorm1d(1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.dropout = nn.Dropout(0.3)

#         # Define output layers for value and policy predictions
#         self.fc_val = nn.Linear(512, 1)  # Value head
#         self.fc_pol = nn.Linear(512, 44)  # Policy head

#     def forward(self, x):
#         # Apply convolutional and batch normalization layers
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))

#         # Flatten the output for the fully connected layers
#         x = x.view(-1, 512 * 5 * 5)

#         # Apply fully connected layers with dropout
#         x = F.relu(self.bn5(self.fc1(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn6(self.fc2(x)))
#         x = self.dropout(x)

#         # Generate and return value and policy predictions
#         val = torch.tanh(self.fc_val(x))  # Value prediction
#         pol = F.log_softmax(self.fc_pol(x), dim=1)  # Policy prediction

#         return pol, val


# Definition of the residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Adding the residual value
        return F.relu(out)


# Modification of the QuixoNet class to include residual blocks
class QuixoNet(nn.Module):
    def __init__(self):
        super(QuixoNet, self).__init__()

        self.conv_initial = nn.Conv2d(2, 256, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(256)

        # Creating 5 residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # Defining the output layers
        self.fc_val = nn.Linear(256, 1)  # Value head
        self.fc_pol = nn.Linear(256, 44)  # Policy head

    def forward(self, x):
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        x = self.res_blocks(x)
        x = F.avg_pool2d(x, x.size()[2:])  # Global average pooling
        x = x.view(x.size(0), -1)  # Flattening

        # Calculating value and policy predictions
        val = torch.tanh(self.fc_val(x))  # Value prediction
        pol = F.log_softmax(self.fc_pol(x), dim=1)  # Policy prediction

        return pol, val


# Rest of the training code remains unchanged


def train(
    model,
    train_loader,
    optimizer=None,
    criterion_val=None,
    criterion_pol=None,
    epochs=32,
):
    # Initialize optimizer and loss functions if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if criterion_val is None:
        criterion_val = nn.MSELoss()
    if criterion_pol is None:
        criterion_pol = nn.CrossEntropyLoss()

    model.train()
    model = model.to(device)

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, val_labels, pol_labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                # Move data to the appropriate device
                inputs = inputs.to(device)
                val_labels = val_labels.to(device)
                pol_labels = pol_labels.to(device)

                # Forward pass, backpropagation, and optimization
                optimizer.zero_grad()
                pol_pred, val_pred = model(inputs)
                loss_val = criterion_val(val_pred.squeeze(), val_labels)
                loss_pol = criterion_pol(pol_pred, pol_labels)
                loss = loss_val + loss_pol
                loss.backward()
                optimizer.step()

                # Update running loss for progress display
                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))
