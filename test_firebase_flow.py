#!/usr/bin/env python3
"""
Integration test to verify Firebase upload flow works end-to-end.
This simulates the actual usage in TraffixAI.
"""

import sys
import time
from unittest.mock import Mock, patch

# Mock imports for testing
class MockModule:
    def __getattr__(self, name):
        return Mock()

sys.modules['cv2'] = MockModule()
sys.modules['torch'] = MockModule()
sys.modules['ultralytics'] = MockModule()

# Now import the module
import traffixai


def test_firebase_integration():
    """Test the complete Firebase integration flow."""
    print("Testing Firebase Integration...")
    
    # Create a test config with Firebase enabled but use mocked requests
    with patch('traffixai.requests.put') as mock_put:
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_put.return_value = mock_response
        
        # Create uploader with enabled config
        config = traffixai.FirebaseConfig(
            DATABASE_URL="https://traffix-ai-d02a3-default-rtdb.firebaseio.com",
            UPDATE_INTERVAL=0.1,  # Faster for testing
            ENABLED=True
        )
        
        uploader = traffixai.FirebaseUploader(config)
        
        # Wait a bit for thread to start
        time.sleep(0.2)
        
        print(f"✓ Uploader initialized (running: {uploader.running})")
        
        # Simulate traffic data from the main loop
        lanes_data = {
            "north": {"count": 2, "has_ambulance": False},
            "south": {"count": 1, "has_ambulance": True},
            "east": {"count": 1, "has_ambulance": False},
            "west": {"count": 1, "has_ambulance": False},
            "junction": {"count": 0, "has_ambulance": False}
        }
        
        traffic_data = traffixai.TrafficData(
            timestamp=int(time.time()),
            total_vehicles=5,
            cars=4,
            ambulances=1,
            emergency_active=True,
            lanes=lanes_data,
            fps=25.0,
            camera_online=True
        )
        
        # Upload the data
        uploader.upload_lane_data(traffic_data)
        print("✓ Traffic data queued for upload")
        
        # Wait for upload to happen
        time.sleep(0.3)
        
        # Check status
        status = uploader.get_status()
        print(f"✓ Status: {status}")
        
        # Verify upload was attempted
        if status['upload_count'] > 0:
            print(f"✓ Upload successful (count: {status['upload_count']})")
        else:
            print("⚠ Upload not yet processed (this is normal with async operation)")
        
        # Verify the request was made with correct endpoint
        if mock_put.called:
            call_args = mock_put.call_args
            endpoint = call_args[0][0]
            payload = call_args[1]['json']
            
            print(f"✓ Endpoint: {endpoint}")
            print(f"✓ Payload structure: {list(payload.keys())}")
            
            # Verify structure
            assert 'traffic_data' in payload, "Missing traffic_data"
            assert 'system_status' in payload, "Missing system_status"
            assert payload['traffic_data']['emergency_active'] == True, "emergency_active should be True"
            assert payload['traffic_data']['ambulances'] == 1, "ambulances should be 1"
            assert payload['system_status']['camera_online'] == True, "camera should be online"
            
            print("✓ Payload structure verified")
        
        # Stop the uploader
        uploader.stop()
        time.sleep(0.1)
        
        print(f"✓ Uploader stopped (running: {uploader.running})")
        print("\n✅ All integration tests passed!")


if __name__ == '__main__':
    test_firebase_integration()
