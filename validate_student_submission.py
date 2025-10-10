#!/usr/bin/env python3
"""
Student Submission Validation Script

This script helps students validate their assignment submission before submitting.
It checks:
- File structure and naming
- Agent class implementation
- Model compatibility
- Basic functionality

Usage:
    python validate_student_submission.py <path_to_your_submission_directory>
"""

import os
import sys
import torch
import importlib.util
from pathlib import Path

def validate_submission(submission_path):
    """
    Validate student submission for completeness and correctness.

    Args:
        submission_path: Path to submission directory

    Returns:
        bool: True if validation passes, False otherwise
    """
    print("Validating Student Submission")
    print("=" * 40)
    print(f"Submission path: {submission_path}")
    print()

    # Check if path exists
    if not os.path.exists(submission_path):
        print(f"Error: Submission path '{submission_path}' does not exist!")
        return False

    # Find entry number from directory name
    dir_name = os.path.basename(submission_path)
    if not dir_name.endswith('_ail821_assignment2'):
        print(f"Warning: Directory name should end with '_ail821_assignment2', got: {dir_name}")

    # Extract entry number (everything before _ail821_assignment2)
    entry_number = dir_name.replace('_ail821_assignment2', '')

    print(f"Entry number: {entry_number}")
    print()

    # STEP 1: Check file structure
    print("1. Checking file structure...")
    required_files = [
        f"{entry_number}_agent.py",
        f"{entry_number}_model.pt",
        f"{entry_number}_train.py"
    ]

    missing_files = []
    found_files = []

    for file in required_files:
        file_path = os.path.join(submission_path, file)
        if os.path.exists(file_path):
            found_files.append(file)
            print(f"  Found: {file}")
        else:
            missing_files.append(file)
            print(f"  Missing: {file}")

    if missing_files:
        print(f"\nError: Missing required files: {missing_files}")
        return False

    print("  All required files present!")
    print()

    # STEP 2: Check agent implementation
    print("2. Checking agent implementation...")
    agent_file = os.path.join(submission_path, f"{entry_number}_agent.py")

    try:
        # Load the agent module
        spec = importlib.util.spec_from_file_location("student_agent", agent_file)
        student_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(student_module)

        # Check if StudentAgent class exists
        if not hasattr(student_module, 'StudentAgent'):
            print("  Error: StudentAgent class not found!")
            return False

        print("  StudentAgent class found!")

        # Check required methods
        required_methods = ['__init__', 'act', 'update', 'save_model', 'load_model']
        agent_class = student_module.StudentAgent

        for method in required_methods:
            if not hasattr(agent_class, method):
                print(f"  Error: Missing required method '{method}'")
                return False
            else:
                print(f"  Found method: {method}")

        print("  All required methods present!")
        print()

    except Exception as e:
        print(f"  Error loading agent: {e}")
        return False

    # STEP 3: Check model file
    print("3. Checking model file...")
    model_file = os.path.join(submission_path, f"{entry_number}_model.pt")

    try:
        model_data = torch.load(model_file, map_location=torch.device('cpu'))
        print("  Model file loaded successfully!")

        if isinstance(model_data, dict):
            print("  Model is a state dictionary (correct format)")
        elif hasattr(model_data, 'state_dict'):
            print("  Model is a PyTorch module (correct format)")
        else:
            print("  Warning: Unexpected model format")

        print("  Model validation passed!")
        print()

    except Exception as e:
        print(f"  Error loading model: {e}")
        return False

    # STEP 4: Test basic functionality
    print("4. Testing basic functionality...")

    try:
        from pettingzoo_pong_wrapper import PettingZooPongWrapper
        from gymnasium import spaces

        # Create environment
        env = PettingZooPongWrapper()
        action_space = spaces.Discrete(6)

        # Create student agent
        StudentAgent = student_module.StudentAgent
        agent = StudentAgent(agent_id=0, action_space=action_space)

        # Test with environment
        obs, _ = env.reset()

        # Test action selection
        action = agent.act(obs[0])

        if 0 <= action < 6:
            print(f"  Action selection works! Action: {action}")
        else:
            print(f"  Error: Invalid action {action} (should be 0-5)")
            return False

        # Test model loading
        agent.load_model(model_data)
        print("  Model loading works!")

        env.close()
        print("  Basic functionality test passed!")
        print()

    except Exception as e:
        print(f"  Error in functionality test: {e}")
        return False

    # STEP 5: Final validation
    print("5. Final validation...")
    print("  All checks passed!")
    print()

    print("=" * 40)
    print("VALIDATION SUCCESSFUL!")
    print("=" * 40)
    print(f"Entry number: {entry_number}")
    print("Your submission is ready for evaluation!")
    print()
    print("Summary:")
    for file in found_files:
        print(f"  - {file}")
    print()
    print("Next steps:")
    print("1. Make sure your training script demonstrates good performance")
    print("2. Test your agent against various opponents")
    print("3. Submit when you're satisfied with the performance")

    return True

def main():
    """Main validation function."""
    if len(sys.argv) != 2:
        print("Usage: python validate_student_submission.py <path_to_submission_directory>")
        print("Example: python validate_student_submission.py ./2023CS12345_ail821_assignment2")
        sys.exit(1)

    submission_path = sys.argv[1]

    success = validate_submission(submission_path)

    if success:
        print("\nYour submission is valid and ready for evaluation!")
        sys.exit(0)
    else:
        print("\nYour submission has issues that need to be fixed.")
        sys.exit(1)

if __name__ == "__main__":
    main()