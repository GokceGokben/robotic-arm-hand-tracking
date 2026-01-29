#!/usr/bin/env python3
"""
Quick test script for kinematics validation
Run this to verify your kinematics implementation is working
"""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from kinematics import create_default_robot
import numpy as np

def test_forward_kinematics():
    """Test forward kinematics"""
    print("=" * 60)
    print("Testing Forward Kinematics")
    print("=" * 60)
    
    robot = create_default_robot()
    
    # Test case 1: All zeros (home position)
    test_angles = np.array([0, 0, 0, 0, 0, 0])
    T, _ = robot.forward_kinematics(test_angles)
    pos, orient = robot.get_position_orientation(T)
    
    print(f"\nTest 1 - Home Position:")
    print(f"  Joint angles: {test_angles}")
    print(f"  End-effector position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    print(f"  End-effector orientation (quat): [{orient[0]:.3f}, {orient[1]:.3f}, {orient[2]:.3f}, {orient[3]:.3f}]")
    
    # Test case 2: Some joint angles
    test_angles = np.array([0.5, 0.3, -0.2, 0, 0, 0])
    T, _ = robot.forward_kinematics(test_angles)
    pos, orient = robot.get_position_orientation(T)
    
    print(f"\nTest 2 - Custom Position:")
    print(f"  Joint angles: {test_angles}")
    print(f"  End-effector position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    return True

def test_analytical_ik():
    """Test analytical inverse kinematics"""
    print("\n" + "=" * 60)
    print("Testing Analytical Inverse Kinematics")
    print("=" * 60)
    
    robot = create_default_robot()
    
    # Target position (reachable)
    target_pos = np.array([0.3, 0.2, 0.3])
    
    print(f"\nTarget position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
    # Solve IK
    solution = robot.inverse_kinematics_analytical(target_pos)
    
    if solution is not None:
        print(f"IK solution found: {solution}")
        
        # Verify solution
        T, _ = robot.forward_kinematics(solution)
        achieved_pos, _ = robot.get_position_orientation(T)
        
        error = np.linalg.norm(target_pos - achieved_pos)
        print(f"Achieved position: [{achieved_pos[0]:.3f}, {achieved_pos[1]:.3f}, {achieved_pos[2]:.3f}]")
        print(f"Position error: {error:.6f} meters")
        
        if error < 0.01:
            print("✓ Analytical IK PASSED")
            return True
        else:
            print("✗ Analytical IK error too large")
            return False
    else:
        print("✗ Analytical IK failed to find solution")
        return False

def test_numerical_ik():
    """Test numerical inverse kinematics"""
    print("\n" + "=" * 60)
    print("Testing Numerical Inverse Kinematics (Jacobian)")
    print("=" * 60)
    
    robot = create_default_robot()
    
    # Target position
    target_pos = np.array([0.35, 0.15, 0.35])
    
    print(f"\nTarget position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
    # Solve IK
    solution = robot.inverse_kinematics_numerical(target_pos, method='jacobian')
    
    if solution is not None:
        print(f"IK solution found: {solution}")
        
        # Verify solution
        T, _ = robot.forward_kinematics(solution)
        achieved_pos, _ = robot.get_position_orientation(T)
        
        error = np.linalg.norm(target_pos - achieved_pos)
        print(f"Achieved position: [{achieved_pos[0]:.3f}, {achieved_pos[1]:.3f}, {achieved_pos[2]:.3f}]")
        print(f"Position error: {error:.6f} meters")
        
        if error < 0.01:
            print("✓ Numerical IK PASSED")
            return True
        else:
            print("✗ Numerical IK error too large")
            return False
    else:
        print("✗ Numerical IK failed to find solution")
        return False

def test_workspace_limits():
    """Test workspace boundary checking"""
    print("\n" + "=" * 60)
    print("Testing Workspace Limits")
    print("=" * 60)
    
    robot = create_default_robot()
    
    # Test unreachable position (too far)
    unreachable_pos = np.array([1.0, 0.0, 0.5])
    print(f"\nTesting unreachable position: {unreachable_pos}")
    
    solution = robot.inverse_kinematics_analytical(unreachable_pos)
    if solution is None:
        print("✓ Correctly rejected unreachable position")
        return True
    else:
        print("✗ Should have rejected unreachable position")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("KINEMATICS VALIDATION TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Run tests
    try:
        results.append(("Forward Kinematics", test_forward_kinematics()))
    except Exception as e:
        print(f"✗ Forward Kinematics FAILED with error: {e}")
        results.append(("Forward Kinematics", False))
    
    try:
        results.append(("Analytical IK", test_analytical_ik()))
    except Exception as e:
        print(f"✗ Analytical IK FAILED with error: {e}")
        results.append(("Analytical IK", False))
    
    try:
        results.append(("Numerical IK", test_numerical_ik()))
    except Exception as e:
        print(f"✗ Numerical IK FAILED with error: {e}")
        results.append(("Numerical IK", False))
    
    try:
        results.append(("Workspace Limits", test_workspace_limits()))
    except Exception as e:
        print(f"✗ Workspace Limits FAILED with error: {e}")
        results.append(("Workspace Limits", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Your kinematics implementation is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
