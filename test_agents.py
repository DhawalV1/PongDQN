#!/usr/bin/env python3
"""
Quick test script to verify agent implementation works correctly.
This tests basic functionality without full training.
"""

import torch
import numpy as np
from pettingzoo_pong_wrapper import PettingZooPongWrapper
from gymnasium import spaces

# Import your agent (modify the entry number)
entry_number = "_"  # CHANGE THIS to your entry number
try:
    from importlib import import_module
    agent_module = import_module(f"{entry_number}_agent")
    StudentAgent = agent_module.StudentAgent
    print(f"✅ Successfully imported StudentAgent from {entry_number}_agent.py")
except ImportError as e:
    print(f"❌ Error importing agent: {e}")
    print(f"Make sure {entry_number}_agent.py exists and is in the same directory")
    exit(1)


def test_agent_initialization():
    """Test that agent can be initialized"""
    print("\n" + "="*60)
    print("TEST 1: Agent Initialization")
    print("="*60)
    
    try:
        action_space = spaces.Discrete(6)
        agent = StudentAgent(agent_id=0, action_space=action_space)
        print("✅ Agent initialized successfully")
        print(f"   - Device: {agent.device}")
        print(f"   - Epsilon: {agent.epsilon:.3f}")
        print(f"   - Replay buffer size: {agent.memory._maxsize}")
        return agent
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None


def test_action_selection(agent):
    """Test that agent can select actions"""
    print("\n" + "="*60)
    print("TEST 2: Action Selection")
    print("="*60)
    
    try:
        # Create dummy observation (RGB image)
        dummy_obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        
        # Test action selection
        action = agent.act(dummy_obs)
        
        if isinstance(action, int) and 0 <= action < 6:
            print(f"✅ Action selection works")
            print(f"   - Action: {action} (valid range: 0-5)")
        else:
            print(f"❌ Invalid action: {action} (expected int in range 0-5)")
            return False
        
        # Test multiple actions
        actions = [agent.act(dummy_obs) for _ in range(10)]
        print(f"   - Sample actions: {actions[:5]}")
        
        return True
    except Exception as e:
        print(f"❌ Action selection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_update(agent):
    """Test that agent can perform updates"""
    print("\n" + "="*60)
    print("TEST 3: Update Mechanism")
    print("="*60)
    
    try:
        # Create dummy transition
        obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        action = 2
        reward = 1.0
        next_obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        done = False
        
        # Perform update
        agent.update(obs, action, reward, next_obs, done)
        
        print(f"✅ Update mechanism works")
        print(f"   - Replay buffer size: {len(agent.memory)}")
        
        # Add more transitions
        for i in range(50):
            agent.update(obs, action, reward, next_obs, done)
        
        print(f"   - After 50 transitions: {len(agent.memory)} in buffer")
        
        return True
    except Exception as e:
        print(f"❌ Update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load(agent):
    """Test that agent can save and load models"""
    print("\n" + "="*60)
    print("TEST 4: Save and Load Model")
    print("="*60)
    
    try:
        # Save model
        test_path = "test_model.pt"
        agent.save_model(test_path)
        print(f"✅ Model saved successfully")
        
        # Create new agent
        action_space = spaces.Discrete(6)
        new_agent = StudentAgent(agent_id=0, action_space=action_space)
        
        # Load model
        model_data = torch.load(test_path, map_location='cpu')
        new_agent.load_model(model_data)
        print(f"✅ Model loaded successfully")
        
        # Cleanup
        import os
        os.remove(test_path)
        print(f"   - Test file cleaned up")
        
        return True
    except Exception as e:
        print(f"❌ Save/Load failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_environment(agent):
    """Test agent in actual environment"""
    print("\n" + "="*60)
    print("TEST 5: Environment Interaction")
    print("="*60)
    
    try:
        # Create environment
        env = PettingZooPongWrapper()
        print("✅ Environment created")
        
        # Run short episode
        obs, _ = env.reset()
        agent.frame_processor.reset()
        
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < 100:
            action = agent.act(obs[0])
            next_obs, rewards, done, _, _ = env.step([action, 0])
            
            agent.update(obs[0], action, rewards[0], next_obs[0], done)
            
            total_reward += rewards[0]
            obs = next_obs
            steps += 1
        
        env.close()
        
        print(f"✅ Environment interaction works")
        print(f"   - Steps: {steps}")
        print(f"   - Total reward: {total_reward}")
        print(f"   - Replay buffer: {len(agent.memory)} transitions")
        
        return True
    except Exception as e:
        print(f"❌ Environment interaction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frame_processing(agent):
    """Test frame processing pipeline"""
    print("\n" + "="*60)
    print("TEST 6: Frame Processing")
    print("="*60)
    
    try:
        # Create RGB observation
        rgb_obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        
        # Process frame
        agent.frame_processor.reset()
        processed = agent.frame_processor.get_stacked_frames(rgb_obs)
        
        expected_shape = (4, 84, 84)  # 4 stacked grayscale frames
        
        if processed.shape == expected_shape:
            print(f"✅ Frame processing works")
            print(f"   - Input shape: {rgb_obs.shape}")
            print(f"   - Output shape: {processed.shape}")
            print(f"   - Value range: [{processed.min()}, {processed.max()}]")
        else:
            print(f"❌ Incorrect output shape: {processed.shape}")
            print(f"   - Expected: {expected_shape}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Frame processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("AGENT IMPLEMENTATION TEST SUITE")
    print("="*60)
    print(f"Testing entry number: {entry_number}")
    
    results = []
    
    # Test 1: Initialization
    agent = test_agent_initialization()
    results.append(("Initialization", agent is not None))
    
    if agent is None:
        print("\n❌ Cannot proceed with other tests - initialization failed")
        return
    
    # Test 2: Action Selection
    results.append(("Action Selection", test_action_selection(agent)))
    
    # Test 3: Frame Processing
    results.append(("Frame Processing", test_frame_processing(agent)))
    
    # Test 4: Update Mechanism
    results.append(("Update Mechanism", test_update(agent)))
    
    # Test 5: Save/Load
    results.append(("Save/Load Model", test_save_load(agent)))
    
    # Test 6: Environment Interaction
    results.append(("Environment Interaction", test_with_environment(agent)))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! Your agent is ready for training.")
        print(f"\nNext steps:")
        print(f"1. Run training: python {entry_number}_train.py")
        print(f"2. Validate submission: python validate_student_submission.py ./{entry_number}_ail821_assignment2")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before training.")


if __name__ == "__main__":
    main()