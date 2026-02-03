import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from game import Environment
from model import OthelloPlayer


def visualize_q_values(model_path="othello_dqn_model.pt"):
    """
    Visualize the Q-values of the trained Othello agent for the starting position.
    
    Args:
        model_path (str): Path to the saved model file
    """
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OthelloPlayer()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    # Get the starting position
    env = Environment()

    env.play_random_moves(1)

    state = env.get_state()  # This returns a tensor with shape (3, 8, 8)
    legal_moves = env.game.get_legal_moves()
    
    # Get Q-values from the model
    with torch.no_grad():
        state_tensor = state.unsqueeze(0).to(device)  # Add batch dimension
        q_values = model(state_tensor).squeeze(0).cpu().numpy()  # Shape: (64,)
    
    # Reshape Q-values to 8x8 board
    q_board = q_values.reshape(8, 8)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Board state with legal moves
    ax1.set_xlim(-0.5, 7.5)
    ax1.set_ylim(-0.5, 7.5)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.set_title("Starting Position\n(Yellow dots = Legal moves)", fontsize=14, fontweight='bold')
    ax1.set_xticks(range(8))
    ax1.set_yticks(range(8))
    ax1.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax1.set_yticklabels(range(1, 9))
    ax1.grid(True, linewidth=2)
    
    # Draw the board
    board = env.game.board
    for i in range(8):
        for j in range(8):
            # Draw green background
            rect = plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9, facecolor='darkgreen', edgecolor='black', linewidth=1)
            ax1.add_patch(rect)
            
            # Draw pieces
            if board[i][j] == 1:  # Black
                circle = Circle((j, i), 0.35, color='black', zorder=3)
                ax1.add_patch(circle)
            elif board[i][j] == -1:  # White
                circle = Circle((j, i), 0.35, color='white', edgecolor='black', linewidth=1.5, zorder=3)
                ax1.add_patch(circle)
            
            # Draw legal moves
            if (i, j) in legal_moves:
                circle = Circle((j, i), 0.15, color='yellow', zorder=4)
                ax1.add_patch(circle)
    
    # Right plot: Q-values heatmap
    # Mask illegal moves
    q_board_masked = q_board.copy()
    mask = np.ones((8, 8), dtype=bool)
    for r, c in legal_moves:
        mask[r, c] = False
    q_board_masked[mask] = np.nan
    
    # Create heatmap
    im = ax2.imshow(q_board_masked, cmap='RdYlGn', aspect='equal')
    ax2.set_title("Q-Values for Legal Moves\n(Higher = Better)", fontsize=14, fontweight='bold')
    ax2.set_xticks(range(8))
    ax2.set_yticks(range(8))
    ax2.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax2.set_yticklabels(range(1, 9))
    
    # Add Q-values as text on the heatmap
    for i in range(8):
        for j in range(8):
            if (i, j) in legal_moves:
                text = ax2.text(j, i, f'{q_board[i, j]:.3f}',
                               ha="center", va="center", color="black", 
                               fontsize=10, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Q-Value', fontsize=12)
    
    # Find the best move
    best_q_value = -np.inf
    best_move = None
    for r, c in legal_moves:
        idx = r * 8 + c
        if q_values[idx] > best_q_value:
            best_q_value = q_values[idx]
            best_move = (r, c)
    
    if best_move:
        # Highlight best move on both plots
        r, c = best_move
        ax1.add_patch(Circle((c, r), 0.15, color='red', zorder=5, linewidth=2, fill=False))
        ax1.text(c, r-0.65, '★ BEST', ha='center', color='red', fontsize=10, fontweight='bold')
        
        ax2.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                    fill=False, edgecolor='red', linewidth=3, zorder=5))
        
        print(f"\nBest move: {chr(97+c)}{r+1} with Q-value: {best_q_value:.4f}")
    
    # Print statistics
    print(f"\nQ-value statistics for legal moves:")
    legal_q_values = [q_board[r, c] for r, c in legal_moves]
    print(f"  Min:  {min(legal_q_values):.4f}")
    print(f"  Max:  {max(legal_q_values):.4f}")
    print(f"  Mean: {np.mean(legal_q_values):.4f}")
    print(f"  Std:  {np.std(legal_q_values):.4f}")
    
    plt.tight_layout()
    plt.savefig('q_values_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as 'q_values_visualization.png'")
    plt.show()


if __name__ == "__main__":
    visualize_q_values()
