import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from game import Environment
from model import OthelloPlayer
import time
import argparse
import shutil


def draw_board(ax, board, legal_moves, current_turn, move_number, last_move=None):
    """Draw the Othello board with pieces and legal moves."""
    ax.clear()
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    turn_name = "Black" if current_turn == 1 else "White"
    title = f"Move {move_number} - {turn_name}'s Turn\n(Yellow dots = Legal moves)"
    if last_move:
        r, c = last_move
        title = f"Move {move_number} - {turn_name}'s Turn\nLast move: {chr(97+c)}{r+1}"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticklabels(range(1, 9))
    ax.grid(True, linewidth=2)
    
    for i in range(8):
        for j in range(8):
            # Draw green background
            rect = plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9, facecolor='darkgreen', edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Highlight last move
            if last_move and (i, j) == last_move:
                rect = plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9, facecolor='lightblue', edgecolor='black', linewidth=1)
                ax.add_patch(rect)
            
            # Draw pieces
            if board[i][j] == 1:  # Black
                circle = Circle((j, i), 0.35, color='black', zorder=3)
                ax.add_patch(circle)
            elif board[i][j] == -1:  # White
                circle = Circle((j, i), 0.35, color='white', edgecolor='black', linewidth=1.5, zorder=3)
                ax.add_patch(circle)
            
            # Draw legal moves
            if (i, j) in legal_moves:
                circle = Circle((j, i), 0.15, color='yellow', zorder=4)
                ax.add_patch(circle)


def draw_q_values(ax, q_board, legal_moves, best_move, vmin, vmax):
    """Draw the Q-values heatmap."""
    ax.clear()
    
    # Mask illegal moves
    q_board_masked = q_board.copy()
    mask = np.ones((8, 8), dtype=bool)
    for r, c in legal_moves:
        mask[r, c] = False
    q_board_masked[mask] = np.nan
    
    # Create heatmap
    im = ax.imshow(q_board_masked, cmap='RdYlGn', aspect='equal', vmin=vmin, vmax=vmax)
    ax.set_title("Q-Values for Legal Moves\n(Higher = Better)", fontsize=14, fontweight='bold')
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticklabels(range(1, 9))
    
    # Add Q-values as text
    for i in range(8):
        for j in range(8):
            if (i, j) in legal_moves:
                ax.text(j, i, f'{q_board[i, j]:.3f}',
                        ha="center", va="center", color="black", 
                        fontsize=10, fontweight='bold')
    
    # Highlight best move
    if best_move:
        r, c = best_move
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                                    fill=False, edgecolor='red', linewidth=3, zorder=5))
    
    return im


def get_model_action(model, state, legal_moves, device):
    """Get the best action from the model."""
    with torch.no_grad():
        state_tensor = state.unsqueeze(0).to(device)
        q_values = model(state_tensor).squeeze(0).cpu().numpy()
    
    # Find best legal move
    best_q_value = -np.inf
    best_move = None
    for r, c in legal_moves:
        idx = r * 8 + c
        if q_values[idx] > best_q_value:
            best_q_value = q_values[idx]
            best_move = (r, c)
    
    return best_move, q_values, best_q_value


def create_video(fig, ax1, ax2, frames_data, vmin, vmax, output_path, pause_duration):
    """Create a video file from the frames data."""
    cbar = None
    
    def update_frame(frame_idx):
        nonlocal cbar
        frame = frames_data[frame_idx]
        
        if frame.get('is_final'):
            # Final frame
            draw_board(ax1, frame['board'], frame['legal_moves'], 
                      frame['current_turn'], frame['move_number'], frame['last_move'])
            ax2.clear()
            winner = frame['winner']
            ax2.text(0.5, 0.5, f"Game Over!\n\nWinner: {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}", 
                     ha='center', va='center', fontsize=20, fontweight='bold',
                     transform=ax2.transAxes)
            ax2.axis('off')
        else:
            draw_board(ax1, frame['board'], frame['legal_moves'], 
                      frame['current_turn'], frame['move_number'], frame['last_move'])
            im = draw_q_values(ax2, frame['q_board'], frame['legal_moves'], 
                             frame['best_move'], vmin, vmax)
            
            if cbar is None:
                cbar = plt.colorbar(im, ax=ax2)
                cbar.set_label('Q-Value', fontsize=12)
        
        plt.tight_layout()
        return ax1, ax2
    
    # Create animation
    fps = 1.0 / pause_duration if pause_duration > 0 else 1
    anim = FuncAnimation(fig, update_frame, frames=len(frames_data), 
                        interval=pause_duration * 1000, repeat=False)
    
    # Determine writer based on file extension and availability
    if output_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
    else:
        # Check if FFmpeg is available
        if shutil.which('ffmpeg') is not None:
            writer = FFMpegWriter(fps=fps, bitrate=1800)
        else:
            print("FFmpeg not found, saving as GIF instead...")
            output_path = output_path.rsplit('.', 1)[0] + '.gif'
            writer = PillowWriter(fps=fps)
    
    anim.save(output_path, writer=writer)
    return output_path


def play_interactive(fig, ax1, ax2, frames_data, vmin, vmax, pause_duration):
    """Play the visualization interactively."""
    plt.ion()
    cbar = None
    
    try:
        for frame_idx, frame in enumerate(frames_data):
            if frame.get('is_final'):
                # Final frame
                draw_board(ax1, frame['board'], frame['legal_moves'], 
                          frame['current_turn'], frame['move_number'], frame['last_move'])
                ax2.clear()
                winner = frame['winner']
                ax2.text(0.5, 0.5, f"Game Over!\n\nWinner: {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}", 
                         ha='center', va='center', fontsize=20, fontweight='bold',
                         transform=ax2.transAxes)
                ax2.axis('off')
            else:
                draw_board(ax1, frame['board'], frame['legal_moves'], 
                          frame['current_turn'], frame['move_number'], frame['last_move'])
                im = draw_q_values(ax2, frame['q_board'], frame['legal_moves'], 
                                 frame['best_move'], vmin, vmax)
                
                if cbar is None:
                    cbar = plt.colorbar(im, ax=ax2)
                    cbar.set_label('Q-Value', fontsize=12)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(pause_duration)
        
        plt.ioff()
        plt.show()
        
    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user.")
        plt.ioff()
        plt.show()


def visualize_q_values(model_path="othello_dqn_model.pt", max_moves=60, pause_duration=2.0, 
                       start_random_moves=0, save_video=None):
    """
    Visualize the Q-values as the trained Othello agent plays.
    
    Args:
        model_path (str): Path to the saved model file
        max_moves (int): Maximum number of moves to visualize
        pause_duration (float): Seconds to pause between moves
        start_random_moves (int): Number of random moves to play before visualization
        save_video (str): Path to save video file (e.g., 'output.mp4' or 'output.gif'). 
                         If None, displays interactively.
    """
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OthelloPlayer()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    # Initialize environment
    env = Environment()
    
    # Play some random moves if requested
    if start_random_moves > 0:
        print(f"Playing {start_random_moves} random moves to start...")
        env.play_random_moves(start_random_moves)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Track Q-value range for consistent colorbar
    all_q_values = []
    
    # Store frames for video creation
    frames_data = []
    
    # Prepare game data
    print(f"\nPreparing game data...")
    move_count = 0
    last_move = None
    
    # Collect all game states first
    while move_count < max_moves and not env.game.is_game_over():
        state = env.get_state()
        legal_moves = env.game.get_legal_moves()
        
        if len(legal_moves) == 0:
            print(f"No legal moves available for player {env.game.current_turn}, switching turn...")
            env.game.current_turn *= -1
            continue
        
        # Get Q-values and best move
        best_move, q_values, best_q_value = get_model_action(model, state, legal_moves, device)
        q_board = q_values.reshape(8, 8)
        
        # Track Q-values for scaling
        legal_q_values = [q_board[r, c] for r, c in legal_moves]
        all_q_values.extend(legal_q_values)
        
        # Store frame data
        frames_data.append({
            'board': env.game.board.copy(),
            'legal_moves': legal_moves.copy(),
            'current_turn': env.game.current_turn,
            'move_number': move_count + 1,
            'last_move': last_move,
            'q_board': q_board.copy(),
            'best_move': best_move,
            'best_q_value': best_q_value,
            'legal_q_values': legal_q_values
        })
        
        # Print move info
        if best_move:
            r, c = best_move
            print(f"Move {move_count + 1}: {chr(97+c)}{r+1} (Q-value: {best_q_value:.4f})")
        
        # Make the move
        if best_move:
            r, c = best_move
            action = r * 8 + c
            env.step(action)
            last_move = (r, c)
            move_count += 1
        else:
            break
    
    # Add final frame if game is over
    if env.game.is_game_over():
        print("\nGame Over!")
        winner = env.game.get_winner_id()
        if winner == 1:
            print("Black wins!")
        elif winner == -1:
            print("White wins!")
        else:
            print("It's a draw!")
        
        frames_data.append({
            'board': env.game.board.copy(),
            'legal_moves': [],
            'current_turn': env.game.current_turn,
            'move_number': move_count,
            'last_move': last_move,
            'q_board': None,
            'best_move': None,
            'winner': winner,
            'is_final': True
        })
    
    # Determine colorbar range from all collected data
    if len(all_q_values) > 0:
        vmin = min(all_q_values)
        vmax = max(all_q_values)
    else:
        vmin, vmax = -1, 1
    
    # Now render: either to video or interactive display
    if save_video:
        print(f"\nCreating video: {save_video}")
        actual_output = create_video(fig, ax1, ax2, frames_data, vmin, vmax, save_video, pause_duration)
        plt.close(fig)
        print(f"Video saved successfully to: {actual_output}")
    else:
        print(f"\nPlaying visualization...")
        play_interactive(fig, ax1, ax2, frames_data, vmin, vmax, pause_duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Othello agent Q-values during gameplay')
    parser.add_argument('--model', type=str, default='othello_dqn_model.pt',
                       help='Path to the model file (default: othello_dqn_model.pt)')
    parser.add_argument('--max-moves', type=int, default=60,
                       help='Maximum number of moves to visualize (default: 60)')
    parser.add_argument('--pause', type=float, default=2.0,
                       help='Seconds to pause between moves (default: 2.0)')
    parser.add_argument('--start-random', type=int, default=0,
                       help='Number of random moves before visualization starts (default: 0)')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save to video file instead of showing interactively (e.g., output.mp4 or output.gif)')
    
    args = parser.parse_args()
    
    visualize_q_values(
        model_path=args.model,
        max_moves=args.max_moves,
        pause_duration=args.pause,
        start_random_moves=args.start_random,
        save_video=args.save_video
    )
