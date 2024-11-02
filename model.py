import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, file_name='model.pth'):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.file_name = os.path.join('./model', file_name)
        self.n_games = 0
        self.record = 0
        self.total_score = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        data = {
            'state_dict': self.state_dict(),
            'n_games': self.n_games,
            'record': self.record,
            'total_score': self.total_score
        }
        torch.save(data, self.file_name)

    def load(self):
        if os.path.exists(self.file_name):
            checkpoint = torch.load(self.file_name)
            self.load_state_dict(checkpoint['state_dict'])
            self.n_games = checkpoint['n_games']
            self.record = checkpoint['record']
            self.total_score = checkpoint['total_score']
            print('Model loaded')
            print('Game', self.n_games, 'Record:', self.record)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert lists of numpy arrays to numpy arrays
        state = np.array(state, dtype=np.float64)
        next_state = np.array(next_state, dtype=np.float64)

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

        # 1: predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx].item()  # Make Q_new a scalar
            if not done[idx]:
                Q_new = reward[idx].item() + self.gamma * \
                    torch.max(self.model(next_state[idx])).item()

            # Get the index for action
            action_index = action[idx].item() if action[idx].dim(
            ) == 0 else torch.argmax(action[idx]).item()
            target[idx][action_index] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
