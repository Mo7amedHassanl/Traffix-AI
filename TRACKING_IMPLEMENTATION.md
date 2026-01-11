# TRAFFIX-AI Tracking Features Implementation

## Overview
This implementation adds comprehensive tracking and detection features to the TRAFFIX-AI system, including filtering, duplicate removal, object tracking, and visual enhancements.

## Features Implemented

### 1. Detection Filtering & Validation
Filters invalid detections based on configurable thresholds:

- **Size Filtering**: Rejects detections with area < 400px² or > 150,000px²
- **Aspect Ratio Check**: Rejects detections with width/height ratio < 0.3 or > 3.5  
- **Confidence Threshold**: Requires minimum confidence of 0.35 after initial detection
- **Configuration**: All thresholds defined in `FilteringConfig` dataclass

**Implementation**: `Detection.is_valid()` method checks all criteria.

### 2. Duplicate Detection Removal
Removes overlapping and duplicate detections to improve quality:

- **IoU-based Removal**: Removes detections with IoU > 0.5 (configurable)
- **Center Distance Removal**: Removes detections with center points within 50px
- **Confidence Priority**: Keeps highest confidence detection when duplicates found

**Implementation**: `remove_duplicate_detections()` function processes detection list.

### 3. Object Tracking with Lane Transition Counting
Implements centroid-based tracking with unique IDs:

- **Unique IDs**: Each vehicle gets a persistent ID (e.g., #1, #2, #3)
- **Centroid Matching**: Tracks vehicles across frames (max distance: 100px)
- **Track Confirmation**: Requires 3+ consecutive detections before confirming
- **Track Removal**: Removes tracks after 15 frames without detection
- **Lane Transitions**: Counts when vehicles move between lanes
- **Total Tracked**: Maintains count of all vehicles tracked

**Implementation**: `ObjectTracker` class with `TrackedObject` instances.

**Note**: Lane entry/exit counts track transitions between lanes rather than absolute entry/exit from the monitored area. For true entry/exit counting, entry/exit zones would need to be defined.

### 4. Detection Stabilization
Reduces jitter and improves visual quality:

- **Position Smoothing**: Exponential smoothing with α=0.7 for bounding box positions
- **Size Smoothing**: Exponential smoothing with α=0.8 for bounding box dimensions
- **Lane Stability**: Requires 3 consecutive frames in a lane before changing assignment
- **Confidence Averaging**: Averages confidence over last 5 frames

**Implementation**: `TrackedObject.update()` method applies smoothing.

## New Classes

### `FilteringConfig`
Dataclass containing configurable filtering thresholds:
- `MIN_AREA`, `MAX_AREA`: Detection area limits
- `MIN_ASPECT_RATIO`, `MAX_ASPECT_RATIO`: Aspect ratio limits
- `MIN_CONFIDENCE`: Minimum confidence threshold

### `TrackingConfig`
Dataclass containing tracking parameters:
- `MAX_CENTROID_DISTANCE`: Maximum distance for matching (100px)
- `MIN_CONSECUTIVE_DETECTIONS`: Required detections for confirmation (3)
- `MAX_FRAMES_WITHOUT_DETECTION`: Frames before removal (15)
- `DUPLICATE_IOU_THRESHOLD`: IoU threshold for duplicates (0.5)
- `DUPLICATE_CENTER_DISTANCE`: Center distance threshold (50px)
- `POSITION_SMOOTHING_ALPHA`: Position smoothing factor (0.7)
- `SIZE_SMOOTHING_ALPHA`: Size smoothing factor (0.8)
- `LANE_STABILITY_FRAMES`: Frames for lane stability (3)
- `CONFIDENCE_HISTORY_SIZE`: Frames for confidence averaging (5)
- `MAX_TRAIL_HISTORY`: Maximum trail points to keep (30)

### `Detection`
Data class for single YOLO detection with computed properties:
- `box`: Bounding box in xyxy format
- `class_id`: Vehicle class (0=Ambulance, 1=Car)
- `confidence`: Detection confidence score
- **Properties**:
  - `center`: Center point (cx, cy)
  - `area`: Bounding box area
  - `aspect_ratio`: Width/height ratio
- **Methods**:
  - `is_valid()`: Check if passes filtering
  - `iou(other)`: Calculate IoU with another detection
  - `center_distance(other)`: Calculate center distance

### `TrackedObject`
Represents a tracked vehicle with history and smoothing:
- `track_id`: Unique identifier
- `class_id`: Vehicle class
- `confirmed`: Whether track is confirmed (3+ detections)
- `smoothed_center`: Smoothed center position
- `smoothed_box`: Smoothed bounding box
- `position_history`: Trail of positions
- `confidence_history`: Recent confidence values
- `current_lane`: Current lane
- `lane_history`: Recent lane assignments
- **Properties**:
  - `vehicle_type`: "Car" or "Ambulance"
  - `avg_confidence`: Average of recent confidences
  - `stable_lane`: Lane after stability check
- **Methods**:
  - `update()`: Update with new detection
  - `mark_missed()`: Mark frame without detection
  - `should_remove()`: Check if should be removed
  - `get_display_box()`: Get smoothed box for display

### `ObjectTracker`
Main tracker class handling matching and lifecycle:
- `tracks`: List of active TrackedObject instances
- `frame_num`: Current frame number
- `total_tracked`: Total vehicles tracked
- `lane_entries`: Count per lane
- `lane_exits`: Count per lane
- **Methods**:
  - `update()`: Update with new detections
  - `get_confirmed_tracks()`: Get confirmed tracks only
  - `get_tentative_tracks()`: Get unconfirmed tracks
  - `clear_all_tracks()`: Clear all tracks

## Visual Updates

### Detection Boxes
- Show track ID: "#3 Car: 0.87"
- Two-tone label: vehicle color on top, lane color on bottom
- Lane badge in corner (N, S, E, W, J)
- Border style indicates confirmation status:
  - Solid border: Confirmed track
  - Dashed/gray border: Tentative track

### Track Trails
- Position history shown as fading line
- Toggleable with [T] key
- Color matches lane color
- Center point marked with circle

### Stats Panel
Shows comprehensive tracking information:
```
TRAFFIX-AI + Tracking
ROI: ON (auto)
FPS: 28

Tracking Stats:
  Active: 4 confirmed, 1 tentative
  Total tracked: 12

Cars: 3  |  Ambulances: 1

Vehicles per Lane:
■ North: 1
■ South: 2
■ East: 0
■ West: 1
■ Junction: 0
```

## New Keyboard Controls

- **[C]**: Clear/reset all tracks
- **[T]**: Toggle track trail visualization

Updated controls bar shows all shortcuts split across two lines for better readability.

## Algorithm Details

### Duplicate Removal
1. Sort detections by confidence (highest first)
2. For each detection:
   - Check IoU with all kept detections
   - Check center distance with all kept detections
   - Keep only if neither threshold exceeded
3. Return filtered list

### Track Matching (Greedy)
1. For each existing track:
   - Find closest detection within max distance
   - Must match class_id
   - Match to detection with minimum distance
2. Update matched tracks with new detections
3. Create new tracks for unmatched detections
4. Remove tracks without detections for 15+ frames

### Smoothing (Exponential)
- Position: `new_pos = α * detected + (1-α) * old_pos`
- Size: `new_box = α * detected_box + (1-α) * old_box`
- Higher α means more responsive (less smooth)

### Lane Stability
- Track maintains history of last 3 lane assignments
- Lane only changes if all 3 recent frames agree
- Prevents flickering between lanes

## Testing

The implementation includes 23 comprehensive unit tests covering:
- Detection filtering (area, aspect ratio, confidence)
- Duplicate removal (IoU and center distance)
- Track creation and lifecycle
- Position and size smoothing
- Confidence averaging
- Track confirmation logic

Run tests with: `python3 test_tracking.py -v`

## Configuration

All thresholds are configurable through the dataclasses:

```python
# Modify filtering thresholds
FILTERING_CONFIG = FilteringConfig(
    MIN_AREA=400.0,
    MAX_AREA=150000.0,
    MIN_ASPECT_RATIO=0.3,
    MAX_ASPECT_RATIO=3.5,
    MIN_CONFIDENCE=0.35
)

# Modify tracking parameters
TRACKING_CONFIG = TrackingConfig(
    MAX_CENTROID_DISTANCE=100.0,
    MIN_CONSECUTIVE_DETECTIONS=3,
    MAX_FRAMES_WITHOUT_DETECTION=15,
    POSITION_SMOOTHING_ALPHA=0.7,
    SIZE_SMOOTHING_ALPHA=0.8,
    LANE_STABILITY_FRAMES=3,
    CONFIDENCE_HISTORY_SIZE=5
)
```

## Performance Considerations

- Greedy matching is O(n*m) where n=tracks, m=detections
- Typically fast enough for real-time (< 100 objects)
- For larger scenes, consider Hungarian algorithm
- Smoothing adds minimal overhead
- Trail history limited to 30 points per track

## Future Enhancements

Potential improvements not in current scope:
- True entry/exit zones (require zone definition)
- Hungarian algorithm for optimal matching
- Kalman filter for prediction
- Re-identification after long occlusions
- Speed and direction estimation
- Trajectory prediction
- Multi-camera tracking

## Acceptance Criteria Status

✅ Detections are filtered by size, aspect ratio, and confidence  
✅ Duplicate/overlapping detections are removed  
✅ Each vehicle gets a unique persistent ID  
✅ Vehicles are tracked across frames  
✅ Bounding boxes are stable without jitter  
✅ Lane assignments are stable (no flickering)  
✅ Lane transitions are tracked per lane  
✅ Track trails are visible (toggleable with [T])  
✅ [C] key clears all tracks  
✅ Stats panel shows tracking information
