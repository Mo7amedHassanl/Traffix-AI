# Firebase Integration Documentation

## Overview
This document describes the Firebase Realtime Database integration added to TRAFFIX-AI.

## Configuration

### FirebaseConfig
Located in `traffixai.py` after the `Config` class:
```python
@dataclass(frozen=True)
class FirebaseConfig:
    DATABASE_URL: str = "https://traffix-ai-d02a3-default-rtdb.firebaseio.com"
    UPDATE_INTERVAL: float = 0.5  # seconds (2 updates per second)
    ENABLED: bool = True
```

To disable Firebase uploads, set `ENABLED = False` in the FirebaseConfig class.

## Data Structure

### TrafficData
The `TrafficData` dataclass structures all traffic information:
- `timestamp`: Unix timestamp (int)
- `total_vehicles`: Total count of all vehicles (int)
- `cars`: Count of cars (int)
- `ambulances`: Count of ambulances (int)
- `emergency_active`: True if any ambulance detected (bool)
- `lanes`: Dictionary with per-lane data (Dict[str, Dict])
- `fps`: Current FPS of the system (float)
- `camera_online`: Camera status (bool)

### JSON Format
Data is uploaded to Firebase in this structure:
```json
{
  "traffic_data": {
    "timestamp": 1705067890,
    "total_vehicles": 5,
    "cars": 4,
    "ambulances": 1,
    "emergency_active": true,
    "lanes": {
      "north": {"count": 2, "has_ambulance": false},
      "south": {"count": 1, "has_ambulance": true},
      "east": {"count": 1, "has_ambulance": false},
      "west": {"count": 1, "has_ambulance": false},
      "junction": {"count": 0, "has_ambulance": false}
    }
  },
  "system_status": {
    "camera_online": true,
    "last_update": 1705067890,
    "fps": 25.0
  }
}
```

## Implementation Details

### FirebaseUploader Class
The `FirebaseUploader` class handles all Firebase communication:

#### Features
- **Background Thread**: Uses a daemon thread to avoid blocking the main detection loop
- **Queue-based Buffering**: Uses a Queue with maxsize=10 to buffer uploads
- **Rate Limiting**: Respects the UPDATE_INTERVAL (0.5s) to avoid flooding Firebase
- **Error Handling**: Catches `requests.exceptions.RequestException` gracefully
- **Statistics Tracking**: Tracks upload_count and error_count
- **Graceful Shutdown**: `stop()` method cleanly terminates the background thread

#### Methods
- `__init__(config)`: Initialize uploader and start background thread
- `upload_lane_data(traffic_data)`: Queue traffic data for upload
- `get_status()`: Get current status (enabled, running, upload_count, error_count, etc.)
- `stop()`: Stop the background thread

### Integration Flow

1. **Initialization** (in `TraffixAI.__init__`):
   ```python
   self.firebase = FirebaseUploader(FIREBASE_CONFIG)
   ```

2. **Main Loop** (in `TraffixAI.run()`):
   - Process detections to get vehicle counts and lane data
   - Track ambulances per lane
   - Call `_upload_to_firebase()` with the data
   - Data is queued and uploaded asynchronously

3. **Cleanup** (in `finally` block):
   ```python
   self.firebase.stop()
   ```

### Display
Firebase status is shown in the stats panel:
- Upload count
- Error count
- Green text if no errors, orange if errors occurred

## REST API Details

### Endpoint
`PUT https://traffix-ai-d02a3-default-rtdb.firebaseio.com/traffix.json`

### Authentication
Currently using unauthenticated access (suitable for public read/write during development).
For production, add Firebase authentication credentials.

### Timeout
Requests have a 5-second timeout to prevent blocking.

## Testing

### Unit Tests
Run `test_firebase_integration.py` to test:
- FirebaseConfig initialization
- TrafficData creation
- FirebaseUploader functionality
- JSON payload structure

### Integration Tests
Run `test_firebase_flow.py` to test:
- End-to-end upload flow
- Background thread operation
- Queue management
- Error handling

### All Tests
```bash
python test_tracking.py          # Existing tracking tests (23 tests)
python test_firebase_integration.py  # Firebase unit tests (5 tests)
python test_firebase_flow.py     # Firebase integration test
```

## Monitoring

### Via UI
The stats panel shows:
- `Firebase: X uploads, Y errors`
- Green text = no errors
- Orange text = errors occurred

### Via Code
```python
status = app.firebase.get_status()
print(status)
# Output: {
#   'enabled': True,
#   'running': True,
#   'upload_count': 42,
#   'error_count': 0,
#   'last_error': 'None',
#   'queue_size': 0
# }
```

### Via Firebase Console
View data at: https://traffix-ai-d02a3-default-rtdb.firebaseio.com/traffix.json

## Troubleshooting

### No uploads happening
1. Check if Firebase is enabled: `FirebaseConfig.ENABLED = True`
2. Check if uploader is running: `app.firebase.get_status()['running']`
3. Check for errors: `app.firebase.get_status()['error_count']`

### Errors in uploads
1. Check network connectivity
2. Verify Firebase URL is correct
3. Check Firebase database rules allow writing
4. Review error message: `app.firebase.get_status()['last_error']`

### High error count
1. May indicate network issues
2. May indicate Firebase database rules blocking writes
3. Check Firebase console for access logs

## Performance Impact

### Main Loop
- **Zero blocking**: All uploads happen in background thread
- **Minimal overhead**: Just queuing data (~microseconds)
- **No FPS impact**: Detection loop runs at full speed

### Network
- **2 uploads/second** (configurable via UPDATE_INTERVAL)
- **Small payload**: ~300-500 bytes per upload
- **Timeout**: 5 seconds max per request

## Future Enhancements

1. **Authentication**: Add Firebase credentials for secure access
2. **Compression**: Compress payloads for reduced bandwidth
3. **Batch Uploads**: Combine multiple updates into one request
4. **Local Caching**: Store data locally if offline
5. **Metrics**: Add more detailed statistics (latency, bandwidth, etc.)
