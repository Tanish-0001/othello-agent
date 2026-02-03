import tkinter as tk
import numpy as np
import torch


class Othello:
    def __init__(self):
        self.board = np.array([[0 for _ in range(8)] for _ in range(8)],
                              dtype=int)  # Let 1 be 'B', -1 be 'W' and 0 be empty
        self.board[3][3] = self.board[4][4] = -1
        self.board[3][4] = self.board[4][3] = 1
        self.current_turn = 1
        self.legal_moves = []

    def is_valid_move(self, row, col):
        if self.board[row][col] != 0 or not (0 <= row < 8 and 0 <= col < 8): return False

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dy, dx in directions:
            nrow, ncol = row + dy, col + dx
            valid = False

            while 0 <= nrow < 8 and 0 <= ncol < 8:
                if self.board[nrow][ncol] == -self.current_turn:
                    valid = True
                elif self.board[nrow][ncol] == self.current_turn:
                    if valid:
                        return True
                    else:
                        break
                else:
                    break
                nrow += dy
                ncol += dx
        return False

    def get_legal_moves(self):
        self.legal_moves = []
        for row in range(8):
            for col in range(8):
                if self.is_valid_move(row, col): self.legal_moves.append((row, col))
        return self.legal_moves

    def flip_pieces(self, row, col):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dy, dx in directions:
            nrow, ncol = row + dy, col + dx
            flip = []

            while 0 <= nrow < 8 and 0 <= ncol < 8:
                if self.board[nrow][ncol] == -self.current_turn:
                    flip.append((nrow, ncol))
                elif self.board[nrow][ncol] == self.current_turn:
                    for r, c in flip: self.board[r][c] = self.current_turn
                    break
                else:
                    break
                nrow += dy
                ncol += dx

    def take_turn(self, row, col) -> bool:
        if len(self.get_legal_moves()) == 0:
            self.current_turn *= -1
            return True
        if not self.is_valid_move(row, col):
            return False

        self.board[row][col] = self.current_turn
        self.flip_pieces(row, col)
        self.current_turn *= -1
        return True

    def is_game_over(self) -> bool:
        if len(self.get_legal_moves()) == 0:
            self.current_turn *= -1
            if len(self.get_legal_moves()) == 0:
                self.current_turn *= -1
                return True
            self.current_turn *= -1

        return False

    def get_winner_id(self) -> int:
        if not self.is_game_over(): return 0
        return 1 if np.sum(self.board) > 0 else 0 if np.sum(self.board) == 0 else -1

    def get_input(self) -> torch.Tensor:
        # Canonical representation:
        # Channel 0: My pieces (always 1s for current player's pieces)
        # Channel 1: Opponent pieces (always 1s for opponent's pieces)
        # Channel 2: Legal moves
        
        my_pieces = (self.board == self.current_turn).astype(np.float32)
        opp_pieces = (self.board == -self.current_turn).astype(np.float32)

        legal_moves_tensor = torch.zeros((8, 8), dtype=torch.float32)
        for row, col in self.get_legal_moves(): 
            legal_moves_tensor[row][col] = 1.0

        input_tensor = torch.stack([
            torch.tensor(my_pieces),
            torch.tensor(opp_pieces),
            legal_moves_tensor
        ])
        return input_tensor

    def display_board(self):
        root = tk.Tk()
        root.title("Othello")
        label = tk.Label(root, text="Current Turn: " + ("Black" if self.current_turn == 1 else "White"),
                         font=("Arial", 16))
        label.pack()
        canvas = tk.Canvas(root, width=480, height=480)
        canvas.pack()
        legal_moves = self.get_legal_moves()

        for i in range(8):
            for j in range(8):
                x, y = j * 60, i * 60
                canvas.create_rectangle(x, y, x + 60, y + 60, fill="green")
                if self.board[i][j] == 1:
                    canvas.create_oval(x + 5, y + 5, x + 55, y + 55, fill="black", outline="")
                elif self.board[i][j] == -1:
                    canvas.create_oval(x + 5, y + 5, x + 55, y + 55, fill="white", outline="")
                if (i, j) in legal_moves: canvas.create_oval(x + 20, y + 20, x + 40, y + 40, fill="yellow", outline="")

        root.mainloop()


if __name__ == "__main__":
    new_game = Othello()
