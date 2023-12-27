import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


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


class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Turn parameters into tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )  # <-- tuple with one element. Looks like (0, 0, 0, 0, 1) where 0 means it's not game over, and 1 means game over.

        # 1. Predicted Q values with current state
        pred = self.model(state)  # <-- here state is neural network. pred is a tensor Q-values corresponding to each action.
        target = pred.clone()  # <-- here we cloned the pred because we want to use to calculate the loss for updating the model without affection pred.

        # 2. Q_new = reward + gamma * max(next_predicted_Q_value) <-- only do this if not done.
        # here idx represents state within each game.
        for idx in range(len(done)):  # <-- The done tensor indicates whether each state is terminal. (The end of an episode in the environment.)
            Q_new = reward[idx]  # <-- Setup the updated Q-value for state-action pair. Initially, just reward. If the game play is terminal, it ends here.
            if not done[idx]:  # <-- Checks if the current state is terminal.
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))  # <-- If the game play was not terminal, then we update the Q-value by doing this.

            # target[idx][torch.argmax(action).item()] indexing into the target tensor. Selects the Q-value corresponding to action taken at state idx.
            # Update the action by Q_new. Q_new based on reward received and the predicted future rewards (if tensor is non-terminal)
            # The purpose of it is to align the predicted Q-values with the observed rewards and the estimated future rewards.
            # THIS IS WHERE LEARNING HAPPENS!!!!!!!!!!
            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
