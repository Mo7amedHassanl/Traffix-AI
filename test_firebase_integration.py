#!/usr/bin/env python3
"""
Test Firebase integration functionality.
"""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import json

# Mock imports for testing
class MockModule:
    def __getattr__(self, name):
        return Mock()

sys.modules['cv2'] = MockModule()
sys.modules['torch'] = MockModule()
sys.modules['ultralytics'] = MockModule()

# Now import the module
import traffixai


class TestFirebaseConfig(unittest.TestCase):
    """Test Firebase configuration."""
    
    def test_firebase_config_exists(self):
        """Test that Firebase config is defined."""
        self.assertIsNotNone(traffixai.FIREBASE_CONFIG)
        self.assertEqual(traffixai.FIREBASE_CONFIG.DATABASE_URL, 
                        "https://traffix-ai-d02a3-default-rtdb.firebaseio.com")
        self.assertEqual(traffixai.FIREBASE_CONFIG.UPDATE_INTERVAL, 0.5)
        self.assertTrue(traffixai.FIREBASE_CONFIG.ENABLED)


class TestTrafficData(unittest.TestCase):
    """Test TrafficData dataclass."""
    
    def test_traffic_data_creation(self):
        """Test that TrafficData can be created with all fields."""
        lanes_data = {
            "north": {"count": 2, "has_ambulance": False},
            "south": {"count": 1, "has_ambulance": True},
            "east": {"count": 0, "has_ambulance": False},
            "west": {"count": 0, "has_ambulance": False},
            "junction": {"count": 0, "has_ambulance": False}
        }
        
        data = traffixai.TrafficData(
            timestamp=1705067890,
            total_vehicles=3,
            cars=2,
            ambulances=1,
            emergency_active=True,
            lanes=lanes_data,
            fps=25.0,
            camera_online=True
        )
        
        self.assertEqual(data.timestamp, 1705067890)
        self.assertEqual(data.total_vehicles, 3)
        self.assertEqual(data.cars, 2)
        self.assertEqual(data.ambulances, 1)
        self.assertTrue(data.emergency_active)
        self.assertEqual(data.lanes["south"]["count"], 1)
        self.assertTrue(data.lanes["south"]["has_ambulance"])
        self.assertEqual(data.fps, 25.0)
        self.assertTrue(data.camera_online)


class TestFirebaseUploader(unittest.TestCase):
    """Test FirebaseUploader functionality."""
    
    def setUp(self):
        """Create a test config with Firebase disabled."""
        self.test_config = traffixai.FirebaseConfig(
            DATABASE_URL="https://test.firebaseio.com",
            UPDATE_INTERVAL=0.5,
            ENABLED=False  # Disabled to avoid background threads in tests
        )
    
    def test_uploader_initialization(self):
        """Test that uploader initializes correctly."""
        uploader = traffixai.FirebaseUploader(self.test_config)
        
        self.assertEqual(uploader.config.DATABASE_URL, "https://test.firebaseio.com")
        self.assertFalse(uploader.running)  # Should not start if disabled
        self.assertEqual(uploader.upload_count, 0)
        self.assertEqual(uploader.error_count, 0)
    
    def test_uploader_status(self):
        """Test that get_status returns correct information."""
        uploader = traffixai.FirebaseUploader(self.test_config)
        
        status = uploader.get_status()
        
        self.assertFalse(status['enabled'])
        self.assertFalse(status['running'])
        self.assertEqual(status['upload_count'], 0)
        self.assertEqual(status['error_count'], 0)
        self.assertEqual(status['last_error'], "None")
    
    @patch('traffixai.requests.put')
    def test_upload_to_firebase(self, mock_put):
        """Test that data is formatted correctly for Firebase upload."""
        # Create uploader with enabled config
        enabled_config = traffixai.FirebaseConfig(
            DATABASE_URL="https://test.firebaseio.com",
            UPDATE_INTERVAL=0.5,
            ENABLED=False  # Still disabled to control threading
        )
        uploader = traffixai.FirebaseUploader(enabled_config)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_put.return_value = mock_response
        
        # Create test data
        lanes_data = {
            "north": {"count": 2, "has_ambulance": False},
            "south": {"count": 1, "has_ambulance": True},
            "east": {"count": 0, "has_ambulance": False},
            "west": {"count": 0, "has_ambulance": False},
            "junction": {"count": 0, "has_ambulance": False}
        }
        
        traffic_data = traffixai.TrafficData(
            timestamp=1705067890,
            total_vehicles=3,
            cars=2,
            ambulances=1,
            emergency_active=True,
            lanes=lanes_data,
            fps=25.0,
            camera_online=True
        )
        
        # Call upload directly
        uploader._upload_to_firebase(traffic_data)
        
        # Verify the PUT request was made
        mock_put.assert_called_once()
        
        # Verify the endpoint
        call_args = mock_put.call_args
        self.assertEqual(call_args[0][0], "https://test.firebaseio.com/traffix.json")
        
        # Verify the payload structure
        payload = call_args[1]['json']
        self.assertIn('traffic_data', payload)
        self.assertIn('system_status', payload)
        
        # Verify traffic_data fields
        self.assertEqual(payload['traffic_data']['timestamp'], 1705067890)
        self.assertEqual(payload['traffic_data']['total_vehicles'], 3)
        self.assertEqual(payload['traffic_data']['cars'], 2)
        self.assertEqual(payload['traffic_data']['ambulances'], 1)
        self.assertTrue(payload['traffic_data']['emergency_active'])
        
        # Verify lane data
        self.assertEqual(payload['traffic_data']['lanes']['south']['count'], 1)
        self.assertTrue(payload['traffic_data']['lanes']['south']['has_ambulance'])
        
        # Verify system_status fields
        self.assertTrue(payload['system_status']['camera_online'])
        self.assertEqual(payload['system_status']['last_update'], 1705067890)
        self.assertEqual(payload['system_status']['fps'], 25.0)
        
        # Verify upload was counted
        self.assertEqual(uploader.upload_count, 1)


if __name__ == '__main__':
    unittest.main()
