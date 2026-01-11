#!/usr/bin/env python3
"""
Unit tests for TRAFFIX-AI tracking features.
Tests filtering, duplicate removal, and tracking functionality.
"""

import sys
import unittest
from unittest.mock import Mock, MagicMock
import numpy as np


# Mock imports for testing
class MockModule:
    def __getattr__(self, name):
        return Mock()

sys.modules['cv2'] = MockModule()
sys.modules['torch'] = MockModule()
sys.modules['ultralytics'] = MockModule()

# Now import the module
import traffixai


class TestDetectionFiltering(unittest.TestCase):
    """Test detection filtering functionality."""
    
    def test_detection_area_calculation(self):
        """Test that detection area is calculated correctly."""
        box = np.array([10, 10, 50, 30])  # 40x20 box
        detection = traffixai.Detection(box=box, class_id=1, confidence=0.8)
        
        self.assertEqual(detection.area, 800.0)
    
    def test_detection_center_calculation(self):
        """Test that detection center is calculated correctly."""
        box = np.array([10, 10, 50, 30])
        detection = traffixai.Detection(box=box, class_id=1, confidence=0.8)
        
        cx, cy = detection.center
        self.assertEqual(cx, 30)
        self.assertEqual(cy, 20)
    
    def test_detection_aspect_ratio(self):
        """Test aspect ratio calculation."""
        box = np.array([10, 10, 50, 30])  # 40x20 = 2.0 aspect ratio
        detection = traffixai.Detection(box=box, class_id=1, confidence=0.8)
        
        self.assertAlmostEqual(detection.aspect_ratio, 2.0)
    
    def test_detection_filtering_min_area(self):
        """Test that detections with area < 400px² are filtered."""
        # Too small: 15x15 = 225px²
        box = np.array([10, 10, 25, 25])
        detection = traffixai.Detection(box=box, class_id=1, confidence=0.8)
        
        self.assertFalse(detection.is_valid())
    
    def test_detection_filtering_max_area(self):
        """Test that detections with area > 150000px² are filtered."""
        # Too large: 500x500 = 250000px²
        box = np.array([10, 10, 510, 510])
        detection = traffixai.Detection(box=box, class_id=1, confidence=0.8)
        
        self.assertFalse(detection.is_valid())
    
    def test_detection_filtering_aspect_ratio_min(self):
        """Test that detections with aspect ratio < 0.3 are filtered."""
        # Too narrow: 10x100 = 0.1 aspect ratio
        box = np.array([10, 10, 20, 110])
        detection = traffixai.Detection(box=box, class_id=1, confidence=0.8)
        
        self.assertFalse(detection.is_valid())
    
    def test_detection_filtering_aspect_ratio_max(self):
        """Test that detections with aspect ratio > 3.5 are filtered."""
        # Too wide: 100x10 = 10.0 aspect ratio
        box = np.array([10, 10, 110, 20])
        detection = traffixai.Detection(box=box, class_id=1, confidence=0.8)
        
        self.assertFalse(detection.is_valid())
    
    def test_detection_filtering_confidence(self):
        """Test that detections with confidence < 0.35 are filtered."""
        box = np.array([10, 10, 50, 50])  # Valid size
        detection = traffixai.Detection(box=box, class_id=1, confidence=0.3)
        
        self.assertFalse(detection.is_valid())
    
    def test_valid_detection(self):
        """Test that valid detections pass filtering."""
        box = np.array([10, 10, 50, 50])  # 40x40, aspect ratio 1.0
        detection = traffixai.Detection(box=box, class_id=1, confidence=0.8)
        
        self.assertTrue(detection.is_valid())


class TestDuplicateDetection(unittest.TestCase):
    """Test duplicate detection removal."""
    
    def test_iou_calculation(self):
        """Test IoU calculation between two detections."""
        det1 = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.8)
        det2 = traffixai.Detection(box=np.array([30, 30, 70, 70]), class_id=1, confidence=0.7)
        
        iou = det1.iou(det2)
        # Overlapping area: 20x20 = 400
        # Union: 40*40 + 40*40 - 400 = 2800
        # IoU = 400/2800 = 0.142857
        self.assertAlmostEqual(iou, 400.0/2800.0, places=5)
    
    def test_center_distance(self):
        """Test center distance calculation."""
        det1 = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.8)
        det2 = traffixai.Detection(box=np.array([10, 10, 90, 90]), class_id=1, confidence=0.7)
        
        # Centers: (30, 30) and (50, 50)
        # Distance: sqrt((50-30)^2 + (50-30)^2) = sqrt(800) ≈ 28.28
        distance = det1.center_distance(det2)
        self.assertAlmostEqual(distance, np.sqrt(800), places=2)
    
    def test_duplicate_removal_by_iou(self):
        """Test that overlapping detections are removed."""
        # Two highly overlapping detections
        det1 = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.9)
        det2 = traffixai.Detection(box=np.array([15, 15, 55, 55]), class_id=1, confidence=0.7)
        
        detections = [det1, det2]
        result = traffixai.remove_duplicate_detections(detections)
        
        # Should keep only the higher confidence detection
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, 0.9)
    
    def test_duplicate_removal_by_center_distance(self):
        """Test that detections with close centers are removed."""
        # Two detections with centers within 50px
        det1 = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.9)
        det2 = traffixai.Detection(box=np.array([30, 30, 70, 70]), class_id=1, confidence=0.7)
        
        detections = [det1, det2]
        result = traffixai.remove_duplicate_detections(detections)
        
        # Centers are (30, 30) and (50, 50), distance = sqrt(800) ≈ 28.28 < 50
        # Should keep only one
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, 0.9)
    
    def test_no_duplicates(self):
        """Test that non-overlapping detections are kept."""
        det1 = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.9)
        det2 = traffixai.Detection(box=np.array([100, 100, 140, 140]), class_id=1, confidence=0.7)
        
        detections = [det1, det2]
        result = traffixai.remove_duplicate_detections(detections)
        
        # Should keep both
        self.assertEqual(len(result), 2)


class TestTrackedObject(unittest.TestCase):
    """Test TrackedObject functionality."""
    
    def setUp(self):
        """Reset track ID counter before each test."""
        traffixai.TrackedObject._next_id = 1
    
    def test_track_creation(self):
        """Test that tracked object is created with unique ID."""
        det = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.8)
        lane = traffixai.Lane.NORTH
        
        track = traffixai.TrackedObject(det, lane, frame_num=1)
        
        self.assertEqual(track.track_id, 1)
        self.assertFalse(track.confirmed)
        self.assertEqual(track.consecutive_detections, 1)
        self.assertEqual(track.current_lane, lane)
    
    def test_track_confirmation(self):
        """Test that track is confirmed after 3 consecutive detections."""
        det = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.8)
        lane = traffixai.Lane.NORTH
        
        track = traffixai.TrackedObject(det, lane, frame_num=1)
        
        # Update twice more
        track.update(det, lane, frame_num=2)
        self.assertFalse(track.confirmed)
        
        track.update(det, lane, frame_num=3)
        self.assertTrue(track.confirmed)
    
    def test_position_smoothing(self):
        """Test that position is smoothed across updates."""
        det1 = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.8)
        det2 = traffixai.Detection(box=np.array([100, 100, 140, 140]), class_id=1, confidence=0.8)
        lane = traffixai.Lane.NORTH
        
        track = traffixai.TrackedObject(det1, lane, frame_num=1)
        initial_center = track.smoothed_center
        
        # Update with detection at different position
        track.update(det2, lane, frame_num=2)
        
        # Smoothed center should be between initial and new position
        # Alpha = 0.7, so new_center = 0.7 * 120 + 0.3 * 30 = 84 + 9 = 93
        expected_cx = 0.7 * 120 + 0.3 * 30
        expected_cy = 0.7 * 120 + 0.3 * 30
        
        self.assertAlmostEqual(track.smoothed_center[0], expected_cx)
        self.assertAlmostEqual(track.smoothed_center[1], expected_cy)
    
    def test_confidence_averaging(self):
        """Test that confidence is averaged over recent frames."""
        det1 = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.8)
        det2 = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.6)
        lane = traffixai.Lane.NORTH
        
        track = traffixai.TrackedObject(det1, lane, frame_num=1)
        track.update(det2, lane, frame_num=2)
        
        avg = track.avg_confidence
        expected = (0.8 + 0.6) / 2
        self.assertAlmostEqual(avg, expected)
    
    def test_track_removal_after_misses(self):
        """Test that track is marked for removal after 15 missed frames."""
        det = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.8)
        lane = traffixai.Lane.NORTH
        
        track = traffixai.TrackedObject(det, lane, frame_num=1)
        
        # Mark as missed 15 times
        for _ in range(15):
            track.mark_missed()
        
        self.assertTrue(track.should_remove())


class TestObjectTracker(unittest.TestCase):
    """Test ObjectTracker functionality."""
    
    def setUp(self):
        """Reset track ID counter before each test."""
        traffixai.TrackedObject._next_id = 1
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = traffixai.ObjectTracker()
        
        self.assertEqual(len(tracker.tracks), 0)
        self.assertEqual(tracker.frame_num, 0)
        self.assertEqual(tracker.total_tracked, 0)
    
    def test_new_track_creation(self):
        """Test that new tracks are created for detections."""
        tracker = traffixai.ObjectTracker()
        
        det = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.8)
        lane = traffixai.Lane.NORTH
        
        tracker.update([det], [lane])
        
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.total_tracked, 1)
    
    def test_track_matching(self):
        """Test that detections are matched to existing tracks."""
        tracker = traffixai.ObjectTracker()
        
        # Create initial track
        det1 = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.8)
        lane = traffixai.Lane.NORTH
        tracker.update([det1], [lane])
        
        track_id = tracker.tracks[0].track_id
        
        # Update with nearby detection
        det2 = traffixai.Detection(box=np.array([15, 15, 55, 55]), class_id=1, confidence=0.9)
        tracker.update([det2], [lane])
        
        # Should still have only one track
        self.assertEqual(len(tracker.tracks), 1)
        # Track ID should be the same
        self.assertEqual(tracker.tracks[0].track_id, track_id)
    
    def test_clear_tracks(self):
        """Test that clear_all_tracks removes all tracks."""
        tracker = traffixai.ObjectTracker()
        
        det = traffixai.Detection(box=np.array([10, 10, 50, 50]), class_id=1, confidence=0.8)
        lane = traffixai.Lane.NORTH
        tracker.update([det], [lane])
        
        self.assertEqual(len(tracker.tracks), 1)
        
        tracker.clear_all_tracks()
        
        self.assertEqual(len(tracker.tracks), 0)


if __name__ == '__main__':
    unittest.main()
