from othello import Othello
import numpy as np
import random
import torch


class Environment:
    def __init__(self):
        self.game = Othello()

    def get_state(self) -> torch.Tensor:
        """
        Retrieves the current state of the environment

        Returns:
            torch.Tensor: the current board where 1 is a black piece, -1 is white, and 0 is an empty square.
        """
        return self.game.get_input()

    def step(self, action: int) -> tuple[torch.Tensor, float, int]:
        """
        Takes a step in the environment.

        Parameters
        ----------
        action : int
            action = 8 * row + col, coordinates where the agent places a piece.

        Returns
        -------
        next_state : tuple of (torch.Tensor, int)
            The next state after taking the action. Includes the updated board and the current turn.
        reward : float
            The reward obtained from the action.
        done : int
            1 if the game is over after this action, otherwise 0.
        """

        r, c = divmod(action, 8)
        player = self.game.current_turn

        num_pieces_of_self_before = np.sum(self.game.board == player)
        success = self.game.take_turn(r, c)
        num_pieces_of_self_after = np.sum(self.game.board == player)

        num_pieces_overturned = num_pieces_of_self_after - num_pieces_of_self_before - 1

        if not success:
            return self.get_state(), -1, 0

        done = int(self.game.is_game_over())
        reward = self._reward(player, done, r, c, num_pieces_overturned)

        return self.get_state(), reward, done

    def _reward(self, player: int, done: int, r: int, c: int, num_pieces_overturned: int) -> float:
        """
        Returns the reward obtained from the action.

        Parameters
        ----------
        player : int
            Who made the action
        done : int
            True if the game is over after this action, otherwise False.
        r : int
        c : int
        num_pieces_overturned : int

        Returns
        -------
        reward : float
            The reward obtained from the action.
        """
        if done == 1:
            winner = self.game.get_winner_id()
            if winner == 0:
                return 0

            return 1 if winner == player else -1
        else:
            reward = 0
            # reward += num_pieces_overturned * 0.01  # number of pieces overturned by its move
            if (r == 0 and c == 0) or (r == 0 and c == 7) or (r == 7 and c == 0) or (r == 7 and c == 7):  # corner move
                reward += 0.2

            # TODO: reward based on the number of legal moves gained over the last move
            return reward

    def reset(self) -> None:
        self.game = Othello()

    def play_random_moves(self, num_moves: int) -> None:
        for _ in range(num_moves):
            legal_moves = self.game.get_legal_moves()

            if len(legal_moves) > 0:
                move = random.choice(legal_moves)
                self.game.take_turn(move[0], move[1])
            else:
                break
