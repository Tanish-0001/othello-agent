from othello import Othello
from game import Environment
from model import OthelloPlayer
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from collections import deque


class Agent:
    def __init__(self, gamma=0.999, epsilon=1.0, lr=0.001, max_memory=10_000, batch_size=32):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory)

        self.model = OthelloPlayer()
        self.target_model = OthelloPlayer()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = StepLR(optimizer=self.optimizer,step_size=5000,gamma=0.05)
        self.loss_func = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.snapshot = OthelloPlayer()
        self.snapshot.load_state_dict(self.model.state_dict())

    def action(self, state, legal_moves, version="new"):
        if version == "new":
            model = self.model

            if random.random() < self.epsilon and len(legal_moves) > 0:
                move_index = (random.choice(legal_moves))
                return 8 * move_index[0] + move_index[1]
        else:
            model = self.snapshot

        tensor_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = model(tensor_t).squeeze(0).cpu().detach().numpy()

        mask = np.full(q_values.shape, -np.inf)
        for r, c in legal_moves:
            idx = 8 * r + c
            mask[idx] = 0.0

        masked_q_values = q_values + mask
        return int(np.argmax(masked_q_values))

    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return -1.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_value = self.model(states_tensor)
        q_predicted = q_value.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        q_next = torch.zeros(self.batch_size, device=self.device)

        non_terminal_mask = (dones == 0)
        non_terminal_next_states = [s for s in next_states if s is not None]

        if non_terminal_next_states:
            non_terminal_next_states = torch.stack(non_terminal_next_states).to(self.device)
            q_next[non_terminal_mask] = self.target_model(non_terminal_next_states).detach().max(1)[0]

        q_target = rewards_tensor + (self.gamma * q_next * (1 - dones_tensor))

        loss = self.loss_func(q_target, q_predicted)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        return loss.item()

    def evaluate(self, episodes):
        test_env = Environment()
        rl_score = 0
        for ep in range(episodes):
            test_env.reset()
            test_game = test_env.game
            # print(ep)
            rl_player = 1 if ep % 2 == 0 else -1
            while not test_game.is_game_over():
                state = test_env.get_state()
                legal_moves = test_game.get_legal_moves()

                if len(legal_moves) == 0:
                    test_game.current_turn *= -1
                    legal_moves = test_game.get_legal_moves()

                if test_game.current_turn == rl_player:
                    action = self.action(state, legal_moves)
                else:
                    move_index = (random.choice(legal_moves))
                    action = 8 * move_index[0] + move_index[1]

                test_env.step(action=action)

            winner = test_game.get_winner_id()
            if winner == rl_player:
                rl_score += 1
            elif winner == 0:
                rl_score += 0.5
            else:
                pass
        print("RL Win rate: ", (rl_score / episodes) * 100, "%")
    
    def eval_against_snapshot(self):
        test_env = Environment()
        rl_score = 0
        for ep in range(10):
            test_env.reset()
            test_game = test_env.game
            # print(ep)
            rl_player = 1 if ep % 2 == 0 else -1
            while not test_game.is_game_over():
                state = test_env.get_state()
                legal_moves = test_game.get_legal_moves()

                if len(legal_moves) == 0:
                    test_game.current_turn *= -1
                    legal_moves = test_game.get_legal_moves()

                if test_game.current_turn == rl_player:
                    action = self.action(state, legal_moves)
                else:
                    action = self.action(state, legal_moves, "old")

                test_env.step(action=action)

            winner = test_game.get_winner_id()
            if winner == rl_player:
                rl_score += 1
            elif winner == 0:
                rl_score += 0.5
            else:
                pass
        print("RL Win rate against itself: ", (rl_score / 10) * 100, "%")

    def train(self):
        losses = []
        env = Environment()
        loss = 0.0
        for ep in range(1, 20_001):
            env.reset()

            num_random_moves = random.randint(0, int(ep / 500))
            env.play_random_moves(num_random_moves)

            if ep % 10 == 0:
                print(f"Episode: {ep}, Loss: {loss}")
            if ep % 50 == 0:
                self.epsilon = max(self.epsilon * 0.99, 0.08)
                self.evaluate(50)
                self.eval_against_snapshot()
            
            if ep % 2000:
                self.snapshot.load_state_dict(self.model.state_dict())

            game = env.game
            while not game.is_game_over():
                state = env.get_state()
                legal_moves = game.get_legal_moves()
                if len(legal_moves) == 0:
                    game.current_turn *= -1
                    continue

                action = self.action(state, legal_moves)

                next_state, reward, done = env.step(action=action)
                if done:
                    next_state = None

                self.memory.append((state, action, reward, next_state, done))

            loss = self.replay()

            if ep % 10 == 0:
                self.target_model.load_state_dict(self.model.state_dict())


agent = Agent(batch_size=128, lr=0.0001)
agent.train()
