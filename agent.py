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
import time


class SnapshotTracker:
    """
    Data structure to manage and evaluate against historical model snapshots.
    """

    def __init__(self, device):
        self.device = device
        self.snapshots = []  # List of {episode: int, model_state: dict}

    def save_snapshot(self, model, episode):
        snapshot_state = {
            'episode': episode,
            'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()}
        }
        self.snapshots.append(snapshot_state)
        print(f"Saved snapshot #{len(self.snapshots)} at episode {episode}")

    def get_snapshot_count(self):
        return len(self.snapshots)

    def load_snapshot_to_model(self, snapshot_model, snapshot_idx):
        if 0 <= snapshot_idx < len(self.snapshots):
            snapshot_model.load_state_dict(self.snapshots[snapshot_idx]['model_state'])
            return self.snapshots[snapshot_idx]['episode']
        return None


class Agent:
    def __init__(
            self,
            gamma=0.99,
            epsilon=0.5,
            lr=0.001,
            max_memory=10_000,
            batch_size=128,
            load_pretrained=True
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory)

        self.TRAIN_EVERY = 8  # Train every N moves

        self.min_epsilon = 0.1

        self.model = OthelloPlayer()

        if load_pretrained:
            self.model.load_state_dict(torch.load('best_supervised.pt',
                                                  weights_only=True))  # Load pre-trained weights from supervised learning

        self.target_model = OthelloPlayer()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Different learning rates for different parts of the model
        # Backbone (pre-trained via supervised learning) gets smaller LR
        backbone_params = list(self.model.conv_in.parameters()) + \
                          list(self.model.gn_in.parameters()) + \
                          list(self.model.res_blocks.parameters())

        param_groups = [
            {'params': backbone_params, 'lr': self.lr * 0.1},  # Smaller LR for pre-trained backbone
            {'params': self.model.value.parameters(), 'lr': self.lr},  # Value head
            {'params': self.model.advantage.parameters(), 'lr': self.lr},  # Advantage head
        ]

        self.optimizer = torch.optim.Adam(param_groups, lr=self.lr)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=75_000, gamma=0.5)

        self.loss_func = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.snapshot_tracker = SnapshotTracker(self.device)

        self.logs = {
            "against_random_player": [],
            "against_all_snapshots": []
        }

    def action(self, state, legal_moves, model, use_epsilon_greedy=False) -> int:
        if len(legal_moves) == 0:
            raise ValueError("No legal moves available to select an action.")

        if use_epsilon_greedy and random.random() < self.epsilon:
            move_index = (random.choice(legal_moves))
            return 8 * move_index[0] + move_index[1]

        tensor_t = state.unsqueeze(0).to(self.device)
        q_values = model(tensor_t).squeeze(0).cpu().detach().numpy()

        mask = state[2, :, :].reshape(64, ).numpy()  # legal moves channel
        q_values[mask == 0] = -np.inf  # set illegal moves q score to -inf

        return int(np.argmax(q_values))

    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return -1.0

        self.model.train()
        self.target_model.eval()

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.stack(states).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q values predicted by model for the actions taken in the current states
        q_value = self.model(states_tensor)
        q_predicted = q_value.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Q values predicted by model for the next states
        q_next = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            non_terminal_mask = (dones_tensor == 0)
            non_terminal_next_states = [s for s in next_states if s is not None]

            if non_terminal_next_states:
                non_terminal_next_states = torch.stack(non_terminal_next_states).to(self.device)
                legal_move_mask = non_terminal_next_states[:, 2, :, :].reshape(-1, 64)  # last channel of each state is the legal moves for that state

                # find the best actions according the online model
                online_q_next = self.model(non_terminal_next_states).detach()
                online_q_next[legal_move_mask == 0] = -torch.inf  # set illegal moves target q score to -inf

                best_actions = online_q_next.argmax(1).unsqueeze(1)

                # find the q values of those best actions according to the target model
                outputs = self.target_model(non_terminal_next_states).detach()
                q_next[non_terminal_mask] = outputs.gather(1, best_actions).squeeze(1)

        q_target = rewards_tensor + (self.gamma * q_next * (1 - dones_tensor))

        loss = self.loss_func(q_target, q_predicted)

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def evaluate(self, num_trials=2):
        """
        Evaluate against random player.
        num_trials=2: one as white, one as black
        """
        self.model.eval()
        test_env = Environment()
        rl_score = 0

        for ep in range(num_trials):
            test_env.reset()
            test_game = test_env.game
            rl_player = 1 if ep % 2 == 0 else -1

            while not test_game.is_game_over():
                state = test_env.get_state()
                legal_moves = test_game.get_legal_moves()

                if len(legal_moves) == 0:
                    test_game.current_turn *= -1
                    legal_moves = test_game.get_legal_moves()

                if test_game.current_turn == rl_player:
                    action = self.action(state, legal_moves, model=self.model, use_epsilon_greedy=False)
                else:
                    move_index = random.choice(legal_moves)
                    action = 8 * move_index[0] + move_index[1]

                test_env.step(action=action)

            winner = test_game.get_winner_id()
            if winner == rl_player:
                rl_score += 1
            elif winner == 0:
                rl_score += 0.5

        win_rate = (rl_score / num_trials)
        self.logs["against_random_player"].append(win_rate)
        print(f"Win rate against random player: {win_rate * 100:.1f}%")

    def eval_against_snapshot(self, current_episode, num_trials=2):
        """
        Evaluate current model against ALL previous snapshots.
        num_trials=2: one as white, one as black for each snapshot
        """
        self.model.eval()

        num_snapshots = self.snapshot_tracker.get_snapshot_count()
        if num_snapshots == 0:
            print("No snapshots available for evaluation.")
            return

        # Create a temporary model for loading snapshots
        snapshot_model = OthelloPlayer()
        snapshot_model.to(self.device)
        snapshot_model.eval()

        # Store results for all snapshots
        snapshot_results = {
            'current_episode': current_episode,
            'results': []
        }

        print(f"\n{'=' * 60}")
        print(f"Evaluating Episode {current_episode} against {num_snapshots} snapshots")
        print(f"{'=' * 60}")

        for snapshot_idx in range(num_snapshots):
            snapshot_episode = self.snapshot_tracker.load_snapshot_to_model(snapshot_model, snapshot_idx)

            test_env = Environment()
            rl_score = 0

            for ep in range(num_trials):
                test_env.reset()
                test_game = test_env.game
                rl_player = 1 if ep % 2 == 0 else -1

                while not test_game.is_game_over():
                    state = test_env.get_state()
                    legal_moves = test_game.get_legal_moves()

                    if len(legal_moves) == 0:
                        test_game.current_turn *= -1
                        legal_moves = test_game.get_legal_moves()

                    if test_game.current_turn == rl_player:
                        action = self.action(state, legal_moves, model=self.model, use_epsilon_greedy=False)
                    else:
                        action = self.action(state, legal_moves, model=snapshot_model, use_epsilon_greedy=False)

                    test_env.step(action=action)

                winner = test_game.get_winner_id()
                if winner == rl_player:
                    rl_score += 1
                elif winner == 0:
                    rl_score += 0.5

            win_rate = (rl_score / num_trials)
            snapshot_results['results'].append({
                'snapshot_episode': snapshot_episode,
                'win_rate': win_rate
            })

            print(f"  vs Snapshot #{snapshot_idx + 1} (Episode {snapshot_episode}): {win_rate * 100:.1f}%")

        self.logs["against_all_snapshots"].append(snapshot_results)
        print(f"{'=' * 60}\n")

    def train(self, num_episodes=20_000):
        print(f"Training on {self.device}.")

        env = Environment()
        loss = 0.0
        for ep in range(1, num_episodes + 1):
            env.reset()

            num_random_moves = random.randint(0, int(ep / 500))
            env.play_random_moves(num_random_moves)

            if ep % 5000 == 0:
                self.snapshot_tracker.save_snapshot(self.model, ep)

            if ep % 50 == 0:
                self.epsilon = max(self.epsilon * 0.99, self.min_epsilon)
                print(f"Episode: {ep}, Loss: {loss}")
                self.evaluate(50)
                self.eval_against_snapshot(current_episode=ep, num_trials=2)

            game = env.game

            # Pending transitions for each player: {player_id: (state, action, reward)}
            pending: dict[int, tuple[torch.Tensor, int, float]] = {1: None, -1: None}
            num_moves = 0
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

                action = self.action(state, legal_moves, model=self.model, use_epsilon_greedy=True)
                next_state, reward, done = env.step(action=action)
                num_moves += 1

                # Store this player's transition as pending
                pending[current_player] = (state, action, reward)

                if num_moves % self.TRAIN_EVERY == 0:
                    _ = self.replay()

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

            if ep % 500 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

        torch.save(self.model.state_dict(), "othello_dqn_model.pt")
        with open("training_logs.json", "w") as f:
            json.dump(self.logs, f, indent=2)


agent = Agent(batch_size=256, lr=1e-4, max_memory=100_000)
try:
    start = time.perf_counter()
    agent.train(num_episodes=50_000)
    print(f"Training took {time.perf_counter() - start} seconds.")
except KeyboardInterrupt:
    torch.save(agent.model.state_dict(), "othello_dqn_model_interrupted.pt")
    with open("training_logs.json", "w") as f:
        json.dump(agent.logs, f, indent=2)
