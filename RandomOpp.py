import torch
import numpy as np
from game import Environment
from model import OthelloPlayer
import random

def evaluate_against_random(model, num_games=100, verbose=False, supervised=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    env = Environment()
    stats = {"wins": 0, "losses": 0, "draws": 0}

    print(f"Starting evaluation of {num_games} games on {device}...")

    for i in range(num_games):
        env.reset()
        game = env.game
        
        # Randomly decide if AI is Black (1) or White (-1) for each game
        ai_player = 1 if i % 2 == 0 else -1
        
        while not game.is_game_over():
            legal_moves = game.get_legal_moves()
            
            if not legal_moves:
                game.current_turn *= -1
                continue

            if game.current_turn == ai_player:
                # AI's Turn
                state = env.get_state().unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state, supervised=supervised).squeeze(0).cpu().numpy()
                
                # Pick the best legal move
                best_q = -np.inf
                best_move_idx = None
                for r, c in legal_moves:
                    idx = r * 8 + c
                    if q_values[idx] > best_q:
                        best_q = q_values[idx]
                        best_move_idx = idx
                
                env.step(best_move_idx)
            else:
                # Random Player's Turn
                move = random.choice(legal_moves)
                env.step(move[0] * 8 + move[1])

        # Report Game Results
        winner = game.get_winner_id()
        if winner == ai_player:
            stats["wins"] += 1
        elif winner == 0:
            stats["draws"] += 1
        else:
            stats["losses"] += 1

        if (i + 1) % 300 == 0:
            print(f"Games played: {i+1}/{num_games}...")

    # Final Report
    win_rate = (stats["wins"] / num_games) * 100

    if verbose:
        print("\n" + "="*30)
        print("EVALUATION REPORT")
        print("="*30)
        print(f"Model: {model_path}")
        print(f"Total Games: {num_games}")
        print(f"Wins:   {stats['wins']}")
        print(f"Losses: {stats['losses']}")
        print(f"Draws:  {stats['draws']}")
        print(f"Win Rate: {win_rate:.2f}%")
        print("="*30)

    return win_rate


if __name__ == "__main__":
    model_path = "othello_supervised.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OthelloPlayer()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Ensure you have a trained model file.")
        
    evaluate_against_random(model, num_games=100, verbose=True)
