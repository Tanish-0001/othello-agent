from othello import Othello
from game import Environment
from model import OthelloPlayer
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from collections import deque
import json


class Agent:
    def __init__(
            self, 
            gamma=0.99, 
            epsilon=0.9, 
            lr=0.001, 
            max_memory=10_000, 
            batch_size=32
        ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory)

        self.min_epsilon = 0.1

        self.model = OthelloPlayer()
        self.target_model = OthelloPlayer()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=10000, gamma=0.5)
        self.loss_func = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.snapshot = OthelloPlayer()
        self.snapshot.load_state_dict(self.model.state_dict())
        self.snapshot.to(self.device)
        self.snapshot.eval()

        self.logs = {"against_random_player": [], "against_snapshot": []}

    def action(self, state, legal_moves, use_epsilon_greedy=False, version="new"):
        if len(legal_moves) == 0:
            raise ValueError("No legal moves available to select an action.")

        if version == "new":
            model = self.model

            if use_epsilon_greedy and random.random() < self.epsilon:
                move_index = (random.choice(legal_moves))
                return 8 * move_index[0] + move_index[1]
        else:
            model = self.snapshot

        tensor_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = model(tensor_t).squeeze(0).cpu().detach().numpy()

        mask = state[2, :, :].reshape(64,).numpy()  # legal moves channel
        q_values[mask == 0] = -np.inf  # set illegal moves q score to -inf

        return int(np.argmax(q_values))

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

        non_terminal_mask = (dones_tensor == 0)
        non_terminal_next_states = [s for s in next_states if s is not None]

        if non_terminal_next_states:
            non_terminal_next_states = torch.stack(non_terminal_next_states).to(self.device)
            legal_move_mask = non_terminal_next_states[:, 2, :, :].reshape(-1, 64)  # last channel of each state is the legal moves for that state

            outputs = self.target_model(non_terminal_next_states)
            outputs[legal_move_mask == 0] = -torch.inf  # set illegal moves target q score to -inf

            q_next[non_terminal_mask] = outputs.detach().max(1)[0]

        q_target = rewards_tensor + (self.gamma * q_next * (1 - dones_tensor))

        loss = self.loss_func(q_target, q_predicted)

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def evaluate(self, num_trials=50):
        self.model.eval()
        test_env = Environment()
        rl_score = 0
        for ep in range(num_trials):
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

        self.logs["against_random_player"].append((rl_score) / num_trials)
        print("Win rate against random player: ", (rl_score / num_trials) * 100, "%")
    
    def eval_against_snapshot(self, num_trials=50):
        self.model.eval()
        test_env = Environment()
        rl_score = 0
        for ep in range(num_trials):
            test_env.reset()
            test_game = test_env.game
            rl_player = 1 if ep % 2 == 0 else -1  # alternate as white and black
            while not test_game.is_game_over():
                state = test_env.get_state()
                legal_moves = test_game.get_legal_moves()

                if len(legal_moves) == 0:
                    test_game.current_turn *= -1
                    legal_moves = test_game.get_legal_moves()

                if test_game.current_turn == rl_player:
                    action = self.action(state, legal_moves)
                else:
                    action = self.action(state, legal_moves, version="old")

                test_env.step(action=action)

            winner = test_game.get_winner_id()
            if winner == rl_player:
                rl_score += 1
            elif winner == 0:
                rl_score += 0.5
            else:
                pass

        self.logs["against_snapshot"].append((rl_score) / num_trials)
        print("Win rate against itself: ", (rl_score / num_trials) * 100, "%")

    def train(self, num_episodes=20000):
        print(f"Training on {self.device}.")

        losses = []
        env = Environment()
        loss = 0.0
        for ep in range(1, num_episodes + 1):
            env.reset()

            num_random_moves = random.randint(0, int(ep / 500))
            env.play_random_moves(num_random_moves)
                
            if ep % 50 == 0:
                self.epsilon = max(self.epsilon * 0.99, self.min_epsilon)
                print(f"Episode: {ep}, Loss: {loss}")
                self.evaluate(50)
                self.eval_against_snapshot(50)

            if ep % 5000 == 0:
                self.snapshot.load_state_dict(self.model.state_dict())

            game = env.game
            
            # Pending transitions for each player: {player_id: (state, action, reward)}
            pending = {1: None, -1: None}
            self.model.train()
            while not game.is_game_over():
                current_player = game.current_turn
                legal_moves = game.get_legal_moves()
                
                # Handle no legal moves - turn passes to opponent
                if len(legal_moves) == 0:
                    game.current_turn *= -1
                    continue

                state = env.get_state()
                
                # Complete the previous pending transition for this player
                # This works correctly even if this player just moved (consecutive turns)
                # because `state` is what they see now before their next action
                if pending[current_player] is not None:
                    prev_state, prev_action, prev_reward = pending[current_player]
                    self.memory.append((prev_state, prev_action, prev_reward, state, 0))

                action = self.action(state, legal_moves, use_epsilon_greedy=True)
                next_state, reward, done = env.step(action=action)

                # Store this player's transition as pending
                pending[current_player] = (state, action, reward)

            # Game is over - finalize pending transitions
            game_result = game.get_winner_id()
            for player_id in [1, -1]:
                if pending[player_id] is not None:
                    prev_state, prev_action, _ = pending[player_id]
                    # Final reward based on game outcome
                    if game_result == player_id:
                        final_reward = 1.0
                    elif game_result == 0:
                        final_reward = 0.0
                    else:
                        final_reward = -1.0
                    self.memory.append((prev_state, prev_action, final_reward, None, 1))

            loss = self.replay()
            losses.append(loss)

            if ep % 500 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
        
        torch.save(self.model.state_dict(), "othello_dqn_model.pt")
        with open("training_logs.json", "w") as f:
            json.dump(self.logs, f, indent=2)

try:
    agent = Agent(batch_size=256, lr=1e-4, max_memory=100_000)
    agent.train(num_episodes=50_000)
except KeyboardInterrupt:
    torch.save(agent.model.state_dict(), "othello_dqn_model_interrupted.pt")
    with open("training_logs.json", "w") as f:
            json.dump(agent.logs, f, indent=2)