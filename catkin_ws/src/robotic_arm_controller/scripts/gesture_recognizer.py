#!/usr/bin/env python3
"""
Gesture Recognizer using MediaPipe hand landmarks
Detects various hand gestures for robot control
"""

import numpy as np
from typing import Optional, Dict
import time
import collections


class GestureRecognizer:
    """
    Recognizes hand gestures from MediaPipe landmarks
    """
    
    def __init__(self, confidence_threshold: float = 0.7, debounce_time: float = 0.5):
        """
        Args:
            confidence_threshold: Minimum confidence for gesture detection
            debounce_time: Minimum time between gesture detections (seconds)
        """
        self.confidence_threshold = confidence_threshold
        self.debounce_time = debounce_time
        self.last_gesture_time = {}
        
        # History buffer for smoothing (removes flickering)
        self.history = collections.deque(maxlen=10)
        
        # Landmark indices (MediaPipe hand landmarks)
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        self.THUMB_IP = 3
        self.INDEX_PIP = 6
        self.MIDDLE_PIP = 10
        self.RING_PIP = 14
        self.PINKY_PIP = 18
    
    def recognize(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Recognize gestures from hand landmarks
        
        Args:
            landmarks: Nx3 array of hand landmarks (21 points for MediaPipe)
            
        Returns:
            Dictionary of {gesture_name: confidence}
        """
        if landmarks is None or len(landmarks) != 21:
            return {}
        
        gestures = {}
        
        # Check each gesture
        pinch_conf = self._detect_pinch(landmarks)
        if pinch_conf > self.confidence_threshold:
            gestures['pinch'] = pinch_conf
        
        fist_conf = self._detect_fist(landmarks)
        if fist_conf > self.confidence_threshold:
            gestures['fist'] = fist_conf
        
        open_conf = self._detect_open_palm(landmarks)
        if open_conf > self.confidence_threshold:
            gestures['open'] = open_conf
        
        peace_conf = self._detect_peace(landmarks)
        if peace_conf > self.confidence_threshold:
            gestures['peace'] = peace_conf
        
        thumbs_up_conf = self._detect_thumbs_up(landmarks)
        if thumbs_up_conf > self.confidence_threshold:
            gestures['thumbs_up'] = thumbs_up_conf
        
        return gestures
    
    def get_best_gesture(self, landmarks: np.ndarray) -> Optional[tuple]:
        """
        Get the gesture with highest confidence
        
        Returns:
            Tuple of (gesture_name, confidence) or None
        """
        gestures = self.recognize(landmarks)
        
        if not gestures:
            self.history.append(None)
            return None
        
        # Get immediate winner
        current_winner = max(gestures.items(), key=lambda x: x[1])
        
        # Add to history
        self.history.append(current_winner)
        
        # SMOOTHING: Vote on history
        # Only change gesture if consistent
        valid_history = [g for g in self.history if g is not None]
        if not valid_history:
            return None
            
        # Count occurrences
        from collections import Counter
        counts = Counter([g[0] for g in valid_history])
        most_common = counts.most_common(1)[0]
        
        # Win if present in > 60% of history
        if most_common[1] >= 6:
            # Find the full tuple (name, conf) for this winner
            for g in valid_history:
                if g[0] == most_common[0]:
                    return g
                    
        return None
    
    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(p1 - p2)
    
    def _detect_pinch(self, landmarks: np.ndarray) -> float:
        """
        Detect pinch gesture (thumb and index finger touching)
        """
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        
        # Distance between thumb and index
        dist = self._distance(thumb_tip, index_tip)
        
        # Normalize by hand size (distance from wrist to middle finger)
        wrist = landmarks[self.WRIST]
        middle_tip = landmarks[self.MIDDLE_TIP]
        hand_size = self._distance(wrist, middle_tip)
        
        normalized_dist = dist / (hand_size + 1e-6)
        
        # Pinch if distance is small (< 0.1 of hand size)
        if normalized_dist < 0.1:
            confidence = 1.0 - (normalized_dist / 0.1)
            return min(confidence, 1.0)
        
        return 0.0
    
    def _detect_fist(self, landmarks: np.ndarray) -> float:
        """
        Detect fist gesture (all fingers closed)
        """
        wrist = landmarks[self.WRIST]
        
        # Check if all fingertips are close to palm
        fingertips = [
            landmarks[self.INDEX_TIP],
            landmarks[self.MIDDLE_TIP],
            landmarks[self.RING_TIP],
            landmarks[self.PINKY_TIP]
        ]
        
        # Palm center (approximate)
        palm_center = landmarks[0]
        
        # Calculate average distance of fingertips to palm
        distances = [self._distance(tip, palm_center) for tip in fingertips]
        avg_dist = np.mean(distances)
        
        # Hand size
        hand_size = self._distance(wrist, landmarks[self.MIDDLE_PIP])
        normalized_dist = avg_dist / (hand_size + 1e-6)
        
        # Hand size
        hand_size = self._distance(wrist, landmarks[self.MIDDLE_PIP])
        normalized_dist = avg_dist / (hand_size + 1e-6)
        
        # Fist if fingertips are close to palm (< 1.1 of hand size - EXTREMELY RELAXED)
        # We really want to favor closing the gripper.
        if normalized_dist < 1.1:
            return 1.0
        
        return 0.0
    
    def _detect_open_palm(self, landmarks: np.ndarray) -> float:
        """
        Detect open palm gesture (all fingers extended)
        """
        wrist = landmarks[self.WRIST]
        
        # Check if all fingertips are far from palm
        fingertips = [
            landmarks[self.THUMB_TIP],
            landmarks[self.INDEX_TIP],
            landmarks[self.MIDDLE_TIP],
            landmarks[self.RING_TIP],
            landmarks[self.PINKY_TIP]
        ]
        
        palm_center = landmarks[0]
        
        # Calculate distances
        distances = [self._distance(tip, palm_center) for tip in fingertips]
        avg_dist = np.mean(distances)
        
        # Hand size
        hand_size = self._distance(wrist, landmarks[self.MIDDLE_TIP])
        normalized_dist = avg_dist / (hand_size + 1e-6)
        
        # Open palm if fingertips are far from palm (> 0.7 of hand size - RELAXED from 0.8)
        if normalized_dist > 0.7:
            confidence = min((normalized_dist - 0.7) / 0.2, 1.0)
            return confidence
        
        return 0.0
    
    def _detect_peace(self, landmarks: np.ndarray) -> float:
        """
        Detect peace sign (index and middle fingers extended, others closed)
        """
        wrist = landmarks[self.WRIST]
        palm_center = landmarks[0]
        
        # Index and middle should be extended
        index_tip = landmarks[self.INDEX_TIP]
        middle_tip = landmarks[self.MIDDLE_TIP]
        
        index_dist = self._distance(index_tip, palm_center)
        middle_dist = self._distance(middle_tip, palm_center)
        
        # Ring and pinky should be closed
        ring_tip = landmarks[self.RING_TIP]
        pinky_tip = landmarks[self.PINKY_TIP]
        
        ring_dist = self._distance(ring_tip, palm_center)
        pinky_dist = self._distance(pinky_tip, palm_center)
        
        # Hand size
        hand_size = self._distance(wrist, landmarks[self.MIDDLE_TIP])
        
        # Normalize
        index_norm = index_dist / (hand_size + 1e-6)
        middle_norm = middle_dist / (hand_size + 1e-6)
        ring_norm = ring_dist / (hand_size + 1e-6)
        pinky_norm = pinky_dist / (hand_size + 1e-6)
        
        # Peace: index and middle extended (> 0.7), ring and pinky closed (< 0.5)
        if index_norm > 0.7 and middle_norm > 0.7 and ring_norm < 0.5 and pinky_norm < 0.5:
            confidence = min(index_norm, middle_norm) * (1.0 - max(ring_norm, pinky_norm))
            return min(confidence, 1.0)
        
        return 0.0
    
    def _detect_thumbs_up(self, landmarks: np.ndarray) -> float:
        """
        Detect thumbs up gesture (thumb extended upward, others closed)
        """
        wrist = landmarks[self.WRIST]
        thumb_tip = landmarks[self.THUMB_TIP]
        
        # Thumb should be above wrist (y-coordinate smaller in image space)
        # and extended
        thumb_height = wrist[1] - thumb_tip[1]  # Positive if thumb is above
        
        # Other fingers should be closed
        palm_center = landmarks[0]
        fingertips = [
            landmarks[self.INDEX_TIP],
            landmarks[self.MIDDLE_TIP],
            landmarks[self.RING_TIP],
            landmarks[self.PINKY_TIP]
        ]
        
        distances = [self._distance(tip, palm_center) for tip in fingertips]
        avg_dist = np.mean(distances)
        
        # Hand size
        hand_size = self._distance(wrist, landmarks[self.MIDDLE_TIP])
        
        # Normalize
        thumb_height_norm = thumb_height / (hand_size + 1e-6)
        fingers_dist_norm = avg_dist / (hand_size + 1e-6)
        
        # Thumbs up: thumb extended upward (> 0.5) and other fingers closed (< 0.5)
        if thumb_height_norm > 0.5 and fingers_dist_norm < 0.5:
            confidence = thumb_height_norm * (1.0 - fingers_dist_norm)
            return min(confidence, 1.0)
        
        return 0.0


if __name__ == "__main__":
    # Test with sample landmarks
    print("=== Testing Gesture Recognizer ===")
    
    recognizer = GestureRecognizer()
    
    # Create sample landmarks (21 points)
    # This is a simplified test - real landmarks come from MediaPipe
    landmarks = np.random.rand(21, 3)
    
    gestures = recognizer.recognize(landmarks)
    print(f"Detected gestures: {gestures}")
    
    best = recognizer.get_best_gesture(landmarks)
    if best:
        print(f"Best gesture: {best[0]} with confidence {best[1]:.2f}")
    else:
        print("No gesture detected")
