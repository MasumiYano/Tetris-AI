import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class ConvQNet(nn.Module):
    def __init__(self, additional_features_size=17, output_size=7):
        super().__init__()
        # Improved CNN architecture with batch norm and pooling
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate flattened CNN output size: 128 * 22 * 10 = 28,160
        cnn_output_size = 128 * 22 * 10
        
        # Improved fully connected layers with dropout
        self.fc1 = nn.Linear(cnn_output_size + additional_features_size, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, output_size)
        
    def forward(self, board_tensor, additional_features):
        # Improved CNN processing with batch norm
        x = F.relu(self.bn1(self.conv1(board_tensor)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten CNN output
        x = x.view(x.size(0), -1)
        
        # Combine with additional features
        if additional_features is not None:
            x = torch.cat([x, additional_features], dim=1)
        
        # Improved fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def save(self, file_name='conv_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class SimpleQNet(nn.Module):
    """Single-value state evaluator like the working implementation"""
    def __init__(self, input_size=4, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Single value output
        )
    
    def forward(self, x):
        return self.network(x)
    
    def save(self, file_name='simple_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.is_conv_model = isinstance(model, ConvQNet)

    def train_step(self, state, action, reward, next_state, done):
        if self.is_conv_model:
            # Handle ConvQNet with visual states
            # Check if this is a single experience or batch
            if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], np.ndarray):
                # Single experience: state is (board_tensor, features)
                board_tensor, features = state
                next_board_tensor, next_features = next_state
                
                board_tensor = torch.tensor(board_tensor, dtype=torch.float).unsqueeze(0)
                features = torch.tensor(features, dtype=torch.float).unsqueeze(0)
                next_board_tensor = torch.tensor(next_board_tensor, dtype=torch.float).unsqueeze(0)
                next_features = torch.tensor(next_features, dtype=torch.float).unsqueeze(0)
                action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
                reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
                done = (done,)
            else:
                # Batch of experiences: state is tuple of (board_tensor, features) tuples
                board_tensors, features_list = zip(*state)
                next_board_tensors, next_features_list = zip(*next_state)
                
                board_tensor = torch.tensor(np.array(board_tensors), dtype=torch.float)
                features = torch.tensor(np.array(features_list), dtype=torch.float)
                next_board_tensor = torch.tensor(np.array(next_board_tensors), dtype=torch.float)
                next_features = torch.tensor(np.array(next_features_list), dtype=torch.float)
                action = torch.tensor(action, dtype=torch.long)
                reward = torch.tensor(reward, dtype=torch.float)
                
            # Get predictions
            pred = self.model(board_tensor, features)
            target = pred.clone()
            
            # Calculate Q targets
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    if len(done) == 1:
                        # Single experience
                        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_board_tensor, next_features))
                    else:
                        # Batch experience - need to handle this more carefully
                        next_board_single = next_board_tensor[idx:idx+1]  # Keep batch dimension
                        next_features_single = next_features[idx:idx+1]   # Keep batch dimension
                        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_board_single, next_features_single))
                
                target[idx][torch.argmax(action[idx]).item()] = Q_new
                
        else:
            # Handle LinearQNet (original implementation)
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)

            if len(state.shape) == 1:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done, )

            pred = self.model(state)
            target = pred.clone()

            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

                target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Perform optimization step
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class SimpleQTrainer:
    """Trainer for single-value state evaluation"""
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, reward, next_state, done):
        # Handle both single and batch training
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Q-learning update for state values
        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * self.model(next_state[idx:idx+1]).item()
            target[idx] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
