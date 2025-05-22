import torch
from agent import Agent
from game.game_main import TetrisAI
import time

def load_model(agent, model_path):
    """Load a saved model"""
    try:
        agent.model.load_state_dict(torch.load(model_path))
        agent.model.eval()  # Set to evaluation mode
        print(f"✅ Model loaded from: {model_path}")
        return True
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def play_game_with_model(model_path='./model/simple_model.pth', show_game=True, num_games=1):
    """Play Tetris using a saved model"""
    
    # Create agent (make sure it matches the saved model type)
    agent = Agent(use_cnn=False, use_simple=True)  # For simple_model.pth
    
    # Load the saved model
    if not load_model(agent, model_path):
        return
    
    # Disable exploration - we want to see the trained behavior
    agent.epsilon = 0
    print("🎮 Playing with trained model (no exploration)")
    
    game = TetrisAI()
    all_scores = []
    
    for game_num in range(num_games):
        print(f"\n🎯 Game {game_num + 1}/{num_games}")
        
        steps = 0
        lines_cleared_total = 0
        
        while not game.game_over:
            # Get current state
            state = agent.get_state(game)
            
            # Get action from trained model (no randomness)
            action = agent.get_action(state, game=game)
            
            # Track lines before action
            lines_before = game.score_board.lines_cleared
            
            # Execute action
            _, done, score = game.play_step(action)
            
            # Track lines after action
            lines_after = game.score_board.lines_cleared
            lines_cleared_this_step = lines_after - lines_before
            lines_cleared_total += lines_cleared_this_step
            
            steps += 1
            
            # Print progress every 50 steps or when lines are cleared
            if steps % 50 == 0 or lines_cleared_this_step > 0:
                print(f"  Step {steps}: Score={score}, Total Lines={lines_cleared_total}")
                if lines_cleared_this_step > 0:
                    print(f"    🎉 Cleared {lines_cleared_this_step} lines!")
            
            if done:
                final_score = score
                all_scores.append(final_score)
                print(f"🏁 Game {game_num + 1} finished:")
                print(f"   Final Score: {final_score}")
                print(f"   Lines Cleared: {lines_cleared_total}")
                print(f"   Steps Taken: {steps}")
                game.reset()
                break
            
            # Add small delay if showing the game
            if show_game:
                time.sleep(0.01)  # Adjust speed here
    
    # Print summary
    if len(all_scores) > 1:
        print(f"\n📊 Summary of {num_games} games:")
        print(f"   Average Score: {sum(all_scores)/len(all_scores):.1f}")
        print(f"   Best Score: {max(all_scores)}")
        print(f"   Worst Score: {min(all_scores)}")
    
    return all_scores

def test_model_performance(model_path='./model/simple_model.pth', num_games=10):
    """Test model performance over multiple games"""
    print(f"🧪 Testing model performance over {num_games} games...")
    
    agent = Agent(use_cnn=False, use_simple=True)
    if not load_model(agent, model_path):
        return
    
    agent.epsilon = 0  # No exploration
    game = TetrisAI()
    
    scores = []
    lines_cleared_list = []
    
    for i in range(num_games):
        total_lines = 0
        
        while not game.game_over:
            state = agent.get_state(game)
            action = agent.get_action(state, game=game)
            
            lines_before = game.score_board.lines_cleared
            _, done, score = game.play_step(action)
            lines_after = game.score_board.lines_cleared
            total_lines += (lines_after - lines_before)
            
            if done:
                scores.append(score)
                lines_cleared_list.append(total_lines)
                game.reset()
                break
    
    # Results
    avg_score = sum(scores) / len(scores)
    avg_lines = sum(lines_cleared_list) / len(lines_cleared_list)
    
    print(f"\n📈 Performance Results:")
    print(f"   Games Played: {len(scores)}")
    print(f"   Average Score: {avg_score:.1f}")
    print(f"   Average Lines Cleared: {avg_lines:.1f}")
    print(f"   Best Score: {max(scores)}")
    print(f"   Best Lines: {max(lines_cleared_list)}")
    
    return {
        'scores': scores,
        'lines': lines_cleared_list,
        'avg_score': avg_score,
        'avg_lines': avg_lines
    }

def compare_trained_vs_random():
    """Compare trained model vs random actions"""
    print("🆚 Comparing Trained Model vs Random Actions")
    
    # Test trained model
    print("\n1️⃣ Testing Trained Model:")
    trained_results = test_model_performance('./model/simple_model.pth', num_games=5)
    
    # Test random actions
    print("\n2️⃣ Testing Random Actions:")
    agent = Agent(use_cnn=False, use_simple=True)
    agent.epsilon = 1000  # Force random actions
    game = TetrisAI()
    
    random_scores = []
    for i in range(5):
        while not game.game_over:
            state = agent.get_state(game)
            action = agent.get_action(state, game=game)  # Will be random
            _, done, score = game.play_step(action)
            if done:
                random_scores.append(score)
                game.reset()
                break
    
    random_avg = sum(random_scores) / len(random_scores)
    
    print(f"\n🏆 Comparison Results:")
    print(f"   Trained Model Avg: {trained_results['avg_score']:.1f}")
    print(f"   Random Actions Avg: {random_avg:.1f}")
    print(f"   Improvement: {(trained_results['avg_score']/random_avg - 1)*100:.1f}%")

def load_and_continue_training(model_path='./model/simple_model.pth'):
    """Load a saved model and continue training"""
    from agent import train
    
    print("🔄 Loading model and continuing training...")
    
    # This would require modifying the train() function to accept a pre-loaded agent
    # For now, let's show how to set it up
    agent = Agent(use_cnn=False, use_simple=True)
    if load_model(agent, model_path):
        print("✅ Model loaded successfully!")
        print("💡 To continue training, modify the train() function to accept this agent")
        print("    Or manually run the training loop with this loaded agent")
        return agent
    return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "play":
            # Play one game visually
            play_game_with_model(show_game=True, num_games=1)
            
        elif command == "test":
            # Test performance over multiple games
            test_model_performance(num_games=10)
            
        elif command == "compare":
            # Compare trained vs random
            compare_trained_vs_random()
            
        elif command == "load":
            # Just load and show it works
            agent = Agent(use_cnn=False, use_simple=True)
            load_model(agent, './model/simple_model.pth')
            
        else:
            print("Usage: python play_saved_model.py [play|test|compare|load]")
            print("  play: Play one game with the model")
            print("  test: Test performance over 10 games")  
            print("  compare: Compare trained model vs random actions")
            print("  load: Just load the model and verify it works")
    else:
        # Default: play one game
        print("🎮 Playing one game with your trained model...")
        play_game_with_model(show_game=True, num_games=1)
