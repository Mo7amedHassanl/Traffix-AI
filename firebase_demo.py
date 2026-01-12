#!/usr/bin/env python3
"""
Example script demonstrating Firebase integration usage.
This can be used to test Firebase connectivity independently.
"""

import time
from unittest.mock import Mock
import sys

# Mock CV2, torch, and ultralytics for standalone testing
class MockModule:
    def __getattr__(self, name):
        return Mock()

sys.modules['cv2'] = MockModule()
sys.modules['torch'] = MockModule()
sys.modules['ultralytics'] = MockModule()

# Import after mocking
import traffixai


def main():
    """Demonstrate Firebase integration."""
    print("=" * 60)
    print("Firebase Integration Demo")
    print("=" * 60)
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  Database URL: {traffixai.FIREBASE_CONFIG.DATABASE_URL}")
    print(f"  Update Interval: {traffixai.FIREBASE_CONFIG.UPDATE_INTERVAL}s")
    print(f"  Enabled: {traffixai.FIREBASE_CONFIG.ENABLED}")
    
    # Create uploader
    print(f"\nInitializing Firebase uploader...")
    uploader = traffixai.FirebaseUploader(traffixai.FIREBASE_CONFIG)
    
    # Wait for thread to start
    time.sleep(0.5)
    
    # Check initial status
    status = uploader.get_status()
    print(f"\nInitial Status:")
    print(f"  Running: {status['running']}")
    print(f"  Upload count: {status['upload_count']}")
    print(f"  Error count: {status['error_count']}")
    
    # Simulate traffic data
    print(f"\nSimulating traffic data...")
    
    for i in range(3):
        # Create sample traffic data
        lanes_data = {
            "north": {"count": 2 + i, "has_ambulance": False},
            "south": {"count": 1, "has_ambulance": i == 1},  # Ambulance in second update
            "east": {"count": 1, "has_ambulance": False},
            "west": {"count": 1, "has_ambulance": False},
            "junction": {"count": 0, "has_ambulance": False}
        }
        
        traffic_data = traffixai.TrafficData(
            timestamp=int(time.time()),
            total_vehicles=5 + i,
            cars=4 + i if i != 1 else 4,
            ambulances=1 if i == 1 else 0,
            emergency_active=(i == 1),
            lanes=lanes_data,
            fps=25.0 + i,
            camera_online=True
        )
        
        # Upload data
        uploader.upload_lane_data(traffic_data)
        print(f"  Update {i+1}: {traffic_data.total_vehicles} vehicles, "
              f"emergency={traffic_data.emergency_active}")
        
        # Wait for upload interval
        time.sleep(0.6)
    
    # Check final status
    print(f"\nFinal Status:")
    status = uploader.get_status()
    print(f"  Upload count: {status['upload_count']}")
    print(f"  Error count: {status['error_count']}")
    print(f"  Last error: {status['last_error']}")
    print(f"  Queue size: {status['queue_size']}")
    
    # Stop uploader
    print(f"\nStopping uploader...")
    uploader.stop()
    
    print(f"\n✅ Demo complete!")
    print(f"\nView data at: {traffixai.FIREBASE_CONFIG.DATABASE_URL}/traffix.json")
    print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
