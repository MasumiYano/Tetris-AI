import matplotlib.pyplot as plt
import numpy as np
from game.game_main import TetrisAI
from .agent import create_board_tensor, block_type

def visualize_board_tensor(game, save_path=None):
    """Visualize the 4-channel board tensor to debug visual state extraction"""
    board_tensor = create_board_tensor(game)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 16))
    
    channel_names = [
        "Channel 0: Board State",
        "Channel 1: Current Piece", 
        "Channel 2: Ghost Piece",
        "Channel 3: Boundaries"
    ]
    
    for i in range(4):
        row = i // 2
        col = i % 2
        
        im = axes[row, col].imshow(board_tensor[i], cmap='viridis', aspect='auto')
        axes[row, col].set_title(channel_names[i])
        axes[row, col].set_xlabel('X Position')
        axes[row, col].set_ylabel('Y Position (top to bottom)')
        plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visual state saved to {save_path}")
    else:
        plt.show()

def debug_state_extraction():
    """Run a quick debug session to check state extraction"""
    print("Starting debug session...")
    
    # Create game instance
    game = TetrisAI()
    
    # Let the game run for a few steps to get some pieces on the board
    for i in range(10):
        action = [0, 0, 0, 0, 0, 1, 0]  # Hard drop action
        reward, done, score = game.play_step(action)
        if done:
            break
    
    # Get visual state
    board_tensor = create_board_tensor(game)
    
    # Print shapes and basic info
    print(f"Board tensor shape: {board_tensor.shape}")
    print(f"Current piece type: {type(game.active_block).__name__}")
    print(f"Current piece position: ({game.active_block.grid_x}, {game.active_block.grid_y})")
    print(f"Current piece rotation: {game.active_block.rotation_index}")
    
    # Check next piece and hold piece
    next_piece = block_type(game.next_blocks.next_blocks[0])
    hold_piece = block_type(game.hold_box.block)
    
    print(f"Next piece one-hot: {next_piece}")
    print(f"Hold piece one-hot: {hold_piece}")
    
    # Visualize the tensor
    visualize_board_tensor(game, "debug_visual_state.png")
    
    # Check for any obvious issues
    print("\nDebugging checks:")
    print(f"Board channel min/max: {board_tensor[0].min():.3f}/{board_tensor[0].max():.3f}")
    print(f"Current piece channel sum: {board_tensor[1].sum():.0f}")
    print(f"Ghost piece channel sum: {board_tensor[2].sum():.0f}")
    print(f"Boundary channel sum: {board_tensor[3].sum():.0f}")
    
    return game, board_tensor

if __name__ == "__main__":
    debug_state_extraction()
