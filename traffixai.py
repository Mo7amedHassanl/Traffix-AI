"""
TRAFFIX-AI - Traffic Detection with Lane Detection
Features:
- Multi-point polygon ROI
- 4-lane detection (North, South, East, West)
- Junction/roundabout detection
- Lane displayed on detection box
- Persistent configuration
"""

from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time
import json
import os
import requests
import threading
from queue import Queue, Empty, Full
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict
from enum import Enum


# ============================================================
# CONFIGURATION
# ============================================================
class Lane(Enum):
    """Lane identifiers."""
    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"
    JUNCTION = "Junction"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class FilteringConfig:
    """Detection filtering configuration."""
    MIN_AREA: float = 400.0  # Minimum detection area in pixels²
    MAX_AREA: float = 150000.0  # Maximum detection area in pixels²
    MIN_ASPECT_RATIO: float = 0.3  # Minimum width/height ratio
    MAX_ASPECT_RATIO: float = 3.5  # Maximum width/height ratio
    MIN_CONFIDENCE: float = 0.35  # Minimum confidence after initial detection


@dataclass(frozen=True)
class TrackingConfig:
    """Object tracking configuration."""
    MAX_CENTROID_DISTANCE: float = 100.0  # Max distance for centroid matching
    MIN_CONSECUTIVE_DETECTIONS: int = 3  # Detections required before confirming track
    MAX_FRAMES_WITHOUT_DETECTION: int = 15  # Frames before removing track
    DUPLICATE_IOU_THRESHOLD: float = 0.5  # IoU threshold for duplicate removal
    DUPLICATE_CENTER_DISTANCE: float = 50.0  # Center distance for duplicate removal
    POSITION_SMOOTHING_ALPHA: float = 0.7  # Exponential smoothing for position
    SIZE_SMOOTHING_ALPHA: float = 0.8  # Exponential smoothing for size
    LANE_STABILITY_FRAMES: int = 3  # Consecutive frames in lane before changing
    CONFIDENCE_HISTORY_SIZE: int = 5  # Number of frames for confidence averaging
    MAX_TRAIL_HISTORY: int = 30  # Maximum trail history points to keep


@dataclass(frozen=True)
class Config:
    """Application configuration constants."""
    MODEL_PATH: str = "best.pt"
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT:  int = 720
    CONFIDENCE_THRESHOLD: float = 0.50
    IOU_THRESHOLD:  float = 0.45
    ROI_FILE: str = "roi_config.json"
    ROAD_SENSITIVITY: int = 80
    MASK_UPDATE_INTERVAL: int = 30
    MIN_POLYGON_POINTS: int = 3


@dataclass(frozen=True)
class FirebaseConfig:
    """Firebase Realtime Database configuration."""
    DATABASE_URL: str = "https://traffix-ai-d02a3-default-rtdb.firebaseio.com"
    UPDATE_INTERVAL: float = 0.5  # seconds (2 updates per second)
    ENABLED: bool = True


@dataclass(frozen=True)
class Colors:
    """BGR color constants."""
    CAR:  Tuple[int, int, int] = (16, 185, 129)
    AMBULANCE: Tuple[int, int, int] = (68, 68, 239)
    ROI_BORDER: Tuple[int, int, int] = (255, 128, 0)
    ROAD_OVERLAY: Tuple[int, int, int] = (0, 200, 0)
    CORNER:  Tuple[int, int, int] = (0, 255, 255)
    POINT: Tuple[int, int, int] = (0, 255, 0)
    LINE: Tuple[int, int, int] = (255, 200, 0)
    PREVIEW:  Tuple[int, int, int] = (100, 100, 255)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    GRAY: Tuple[int, int, int] = (128, 128, 128)
    LIGHT_GRAY: Tuple[int, int, int] = (200, 200, 200)
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    
    # Lane-specific colors
    NORTH:  Tuple[int, int, int] = (255, 100, 100)    # Blue-ish
    SOUTH:  Tuple[int, int, int] = (100, 100, 255)    # Red-ish
    EAST:  Tuple[int, int, int] = (100, 255, 100)     # Green-ish
    WEST: Tuple[int, int, int] = (255, 255, 100)     # Cyan-ish
    JUNCTION: Tuple[int, int, int] = (255, 100, 255) # Magenta
    
    # Trail visualization constants
    TRAIL_CENTER_RADIUS: int = 4  # Radius of center point circle
    TRAIL_OUTLINE_RADIUS: int = 6  # Radius of outline circle


# Lane color mapping
LANE_COLORS:  Dict[Lane, Tuple[int, int, int]] = {
    Lane.NORTH: Colors.NORTH,
    Lane.SOUTH: Colors.SOUTH,
    Lane.EAST: Colors.EAST,
    Lane.WEST: Colors.WEST,
    Lane.JUNCTION: Colors.JUNCTION,
    Lane.UNKNOWN:  Colors.GRAY,
}

CONFIG = Config()
FIREBASE_CONFIG = FirebaseConfig()
COLORS = Colors()
FILTERING_CONFIG = FilteringConfig()
TRACKING_CONFIG = TrackingConfig()

# Pre-computed kernels
KERNEL_5x5 = np.ones((5, 5), dtype=np.uint8)

# Yellow color range for road detection (HSV)
YELLOW_LOWER = np.array([15, 50, 50], dtype=np.uint8)
YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)


# ============================================================
# FIREBASE INTEGRATION
# ============================================================
@dataclass
class TrafficData:
    """Traffic data structure for Firebase uploads."""
    timestamp: int
    total_vehicles: int
    cars: int
    ambulances: int
    emergency_active: bool  # True if any ambulance detected
    lanes: Dict[str, Dict]  # Per-lane data
    fps: float
    camera_online: bool


class FirebaseUploader:
    """Asynchronous Firebase Realtime Database uploader."""
    
    def __init__(self, config: FirebaseConfig):
        self.config = config
        self.upload_queue: Queue = Queue(maxsize=10)
        self.upload_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.upload_count = 0
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.running = False
        
        if self.config.ENABLED:
            self._start_uploader()
    
    def _start_uploader(self) -> None:
        """Start the background uploader thread."""
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()
        self.running = True
    
    def _upload_worker(self) -> None:
        """Background worker that uploads data to Firebase."""
        last_upload_time = 0
        
        while not self.stop_flag.is_set():
            try:
                # Rate limiting - respect update interval
                current_time = time.time()
                time_since_last = current_time - last_upload_time
                
                if time_since_last < self.config.UPDATE_INTERVAL:
                    time.sleep(0.1)
                    continue
                
                # Get data from queue (non-blocking)
                try:
                    data = self.upload_queue.get_nowait()
                except Empty:
                    time.sleep(0.1)
                    continue
                
                # Upload to Firebase
                self._upload_to_firebase(data)
                last_upload_time = current_time
                
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                time.sleep(1)
    
    def _upload_to_firebase(self, traffic_data: TrafficData) -> None:
        """Upload data to Firebase via REST API."""
        try:
            # Prepare JSON payload
            payload = {
                "traffic_data": {
                    "timestamp": traffic_data.timestamp,
                    "total_vehicles": traffic_data.total_vehicles,
                    "cars": traffic_data.cars,
                    "ambulances": traffic_data.ambulances,
                    "emergency_active": traffic_data.emergency_active,
                    "lanes": traffic_data.lanes
                },
                "system_status": {
                    "camera_online": traffic_data.camera_online,
                    "last_update": traffic_data.timestamp,
                    "fps": traffic_data.fps
                }
            }
            
            # Upload via PUT request
            endpoint = f"{self.config.DATABASE_URL}/traffix.json"
            response = requests.put(endpoint, json=payload, timeout=5)
            response.raise_for_status()
            
            self.upload_count += 1
            
        except requests.exceptions.RequestException as e:
            self.error_count += 1
            self.last_error = f"Upload failed: {e}"
    
    def upload_lane_data(self, traffic_data: TrafficData) -> None:
        """Queue traffic data for upload."""
        if not self.config.ENABLED or not self.running:
            return
        
        try:
            # Try to add to queue, drop if full
            self.upload_queue.put_nowait(traffic_data)
        except Full:
            # Queue full, silently drop the oldest data and add new
            # This ensures we always have the latest data
            try:
                self.upload_queue.get_nowait()
                self.upload_queue.put_nowait(traffic_data)
            except (Empty, Full):
                # If we still can't add, just drop this update
                pass
    
    def get_status(self) -> Dict[str, Union[int, str, bool]]:
        """Get uploader status for debugging."""
        return {
            "enabled": self.config.ENABLED,
            "running": self.running,
            "upload_count": self.upload_count,
            "error_count": self.error_count,
            "last_error": self.last_error or "None",
            "queue_size": self.upload_queue.qsize()
        }
    
    def stop(self) -> None:
        """Stop the uploader thread."""
        if self.running:
            self.stop_flag.set()
            if self.upload_thread:
                self.upload_thread.join(timeout=2)
            self.running = False


# ============================================================
# DETECTION AND TRACKING CLASSES
# ============================================================
@dataclass
class Detection:
    """Single YOLO detection with computed properties."""
    box: np.ndarray  # xyxy format
    class_id: int
    confidence: float
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point."""
        cx = int((self.box[0] + self.box[2]) * 0.5)
        cy = int((self.box[1] + self.box[3]) * 0.5)
        return (cx, cy)
    
    @property
    def area(self) -> float:
        """Get bounding box area."""
        width = self.box[2] - self.box[0]
        height = self.box[3] - self.box[1]
        return float(width * height)
    
    @property
    def aspect_ratio(self) -> float:
        """Get width/height ratio."""
        width = self.box[2] - self.box[0]
        height = self.box[3] - self.box[1]
        return float(width / height) if height > 0 else 0.0
    
    def is_valid(self) -> bool:
        """Check if detection passes filtering criteria."""
        # Size filtering
        if self.area < FILTERING_CONFIG.MIN_AREA or self.area > FILTERING_CONFIG.MAX_AREA:
            return False
        
        # Aspect ratio check
        if self.aspect_ratio < FILTERING_CONFIG.MIN_ASPECT_RATIO or \
           self.aspect_ratio > FILTERING_CONFIG.MAX_ASPECT_RATIO:
            return False
        
        # Confidence threshold
        if self.confidence < FILTERING_CONFIG.MIN_CONFIDENCE:
            return False
        
        return True
    
    def iou(self, other: 'Detection') -> float:
        """Calculate IoU with another detection."""
        x1 = max(self.box[0], other.box[0])
        y1 = max(self.box[1], other.box[1])
        x2 = min(self.box[2], other.box[2])
        y2 = min(self.box[3], other.box[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return float(intersection / union) if union > 0 else 0.0
    
    def center_distance(self, other: 'Detection') -> float:
        """Calculate center distance to another detection."""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return float(np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2))


class TrackedObject:
    """Tracked vehicle with history and smoothing."""
    
    _next_id = 1
    
    def __init__(self, detection: Detection, lane: Lane, frame_num: int):
        self.track_id = TrackedObject._next_id
        TrackedObject._next_id += 1
        
        self.class_id = detection.class_id
        self.confirmed = False
        self.consecutive_detections = 1
        self.frames_without_detection = 0
        self.last_seen_frame = frame_num
        
        # Position and size (smoothed)
        cx, cy = detection.center
        self.smoothed_center = (float(cx), float(cy))
        self.smoothed_box = detection.box.copy().astype(float)
        
        # History
        self.position_history: List[Tuple[float, float]] = [self.smoothed_center]
        self.confidence_history: List[float] = [detection.confidence]
        
        # Lane tracking
        self.current_lane = lane
        self.lane_history: List[Lane] = [lane]
        self.entry_lane = lane
        
    @property
    def vehicle_type(self) -> str:
        """Get vehicle type string."""
        return "Ambulance" if self.class_id == 0 else "Car"
    
    @property
    def avg_confidence(self) -> float:
        """Get average confidence over recent frames."""
        history = self.confidence_history[-TRACKING_CONFIG.CONFIDENCE_HISTORY_SIZE:]
        return sum(history) / len(history) if history else 0.0
    
    @property
    def stable_lane(self) -> Lane:
        """Get stable lane (requires consistency across frames)."""
        recent_lanes = self.lane_history[-TRACKING_CONFIG.LANE_STABILITY_FRAMES:]
        if len(recent_lanes) >= TRACKING_CONFIG.LANE_STABILITY_FRAMES:
            # Check if all recent lanes are the same
            if all(l == recent_lanes[0] for l in recent_lanes):
                return recent_lanes[0]
        return self.current_lane
    
    def update(self, detection: Detection, lane: Lane, frame_num: int) -> None:
        """Update tracked object with new detection."""
        self.consecutive_detections += 1
        self.frames_without_detection = 0
        self.last_seen_frame = frame_num
        
        # Confirm track if enough consecutive detections
        if not self.confirmed and \
           self.consecutive_detections >= TRACKING_CONFIG.MIN_CONSECUTIVE_DETECTIONS:
            self.confirmed = True
        
        # Smooth position
        cx, cy = detection.center
        alpha = TRACKING_CONFIG.POSITION_SMOOTHING_ALPHA
        self.smoothed_center = (
            alpha * cx + (1 - alpha) * self.smoothed_center[0],
            alpha * cy + (1 - alpha) * self.smoothed_center[1]
        )
        
        # Smooth bounding box size
        alpha_size = TRACKING_CONFIG.SIZE_SMOOTHING_ALPHA
        self.smoothed_box = (
            alpha_size * detection.box + (1 - alpha_size) * self.smoothed_box
        )
        
        # Update history
        self.position_history.append(self.smoothed_center)
        if len(self.position_history) > TRACKING_CONFIG.MAX_TRAIL_HISTORY:
            self.position_history.pop(0)
        
        self.confidence_history.append(detection.confidence)
        if len(self.confidence_history) > TRACKING_CONFIG.CONFIDENCE_HISTORY_SIZE:
            self.confidence_history.pop(0)
        
        # Update lane
        self.current_lane = lane
        self.lane_history.append(lane)
        if len(self.lane_history) > TRACKING_CONFIG.LANE_STABILITY_FRAMES:
            self.lane_history.pop(0)
    
    def mark_missed(self) -> None:
        """Mark that this track was not matched in current frame."""
        self.frames_without_detection += 1
        self.consecutive_detections = 0
    
    def should_remove(self) -> bool:
        """Check if track should be removed."""
        return self.frames_without_detection >= TRACKING_CONFIG.MAX_FRAMES_WITHOUT_DETECTION
    
    def get_display_box(self) -> np.ndarray:
        """Get smoothed bounding box for display."""
        return self.smoothed_box.astype(np.int32)


class ObjectTracker:
    """
    Main tracker class handling centroid matching and track lifecycle.
    
    Note: Entry/exit counting currently tracks lane transitions rather than
    actual entry/exit from the monitored area. For true entry/exit counting,
    entry/exit zones would need to be defined.
    """
    
    def __init__(self):
        self.tracks: List[TrackedObject] = []
        self.frame_num = 0
        self.total_tracked = 0
        
        # Entry/exit counting per lane (tracks lane transitions)
        self.lane_entries: Dict[Lane, int] = {lane: 0 for lane in Lane}
        self.lane_exits: Dict[Lane, int] = {lane: 0 for lane in Lane}
    
    def update(self, detections: List[Detection], lanes: List[Lane]) -> None:
        """Update tracks with new detections."""
        self.frame_num += 1
        
        if not detections:
            # No detections, mark all tracks as missed
            for track in self.tracks:
                track.mark_missed()
            self._cleanup_tracks()
            return
        
        # Match detections to existing tracks using greedy matching
        matched_tracks = set()
        matched_detections = set()
        for i, track in enumerate(self.tracks):
            best_match = None
            best_distance = float('inf')
            
            for j, detection in enumerate(detections):
                if j in matched_detections:
                    continue
                
                # Calculate distance from track to detection
                cx, cy = detection.center
                tx, ty = track.smoothed_center
                distance = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                
                # Only consider if within max distance and same class
                if distance < TRACKING_CONFIG.MAX_CENTROID_DISTANCE and \
                   detection.class_id == track.class_id:
                    if distance < best_distance:
                        best_distance = distance
                        best_match = j
            
            if best_match is not None:
                # Update track with matched detection
                track.update(detections[best_match], lanes[best_match], self.frame_num)
                matched_tracks.add(i)
                matched_detections.add(best_match)
                
                # Track lane transitions (not true entry/exit from monitored area)
                old_lane = track.lane_history[-2] if len(track.lane_history) >= 2 else None
                new_lane = track.current_lane
                if old_lane and old_lane != new_lane and \
                   old_lane != Lane.UNKNOWN and new_lane != Lane.UNKNOWN:
                    self.lane_exits[old_lane] += 1
                    self.lane_entries[new_lane] += 1
        
        # Mark unmatched tracks as missed
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.mark_missed()
        
        # Create new tracks for unmatched detections
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                new_track = TrackedObject(detection, lanes[j], self.frame_num)
                self.tracks.append(new_track)
                self.total_tracked += 1
                
                # Count initial lane appearance (may include re-detections)
                if lanes[j] != Lane.UNKNOWN:
                    self.lane_entries[lanes[j]] += 1
        
        # Remove old tracks
        self._cleanup_tracks()
    
    def _cleanup_tracks(self) -> None:
        """Remove tracks that haven't been seen for too long."""
        tracks_to_remove = []
        for track in self.tracks:
            if track.should_remove():
                # Count as lane exit when track is lost (may include temporary losses)
                if track.current_lane != Lane.UNKNOWN:
                    self.lane_exits[track.current_lane] += 1
                tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            self.tracks.remove(track)
    
    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """Get only confirmed tracks."""
        return [t for t in self.tracks if t.confirmed]
    
    def get_tentative_tracks(self) -> List[TrackedObject]:
        """Get only tentative (unconfirmed) tracks."""
        return [t for t in self.tracks if not t.confirmed]
    
    def clear_all_tracks(self) -> None:
        """Clear all tracks."""
        self.tracks.clear()
        self.frame_num = 0
        # Don't reset total_tracked or entry/exit counts


def remove_duplicate_detections(detections: List[Detection]) -> List[Detection]:
    """Remove duplicate detections using IoU and center distance."""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    
    keep = []
    for det in sorted_dets:
        is_duplicate = False
        
        for kept_det in keep:
            # Check IoU overlap
            if det.iou(kept_det) > TRACKING_CONFIG.DUPLICATE_IOU_THRESHOLD:
                is_duplicate = True
                break
            
            # Check center distance
            if det.center_distance(kept_det) < TRACKING_CONFIG.DUPLICATE_CENTER_DISTANCE:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep.append(det)
    
    return keep


# ============================================================
# SETUP FUNCTIONS
# ============================================================
def setup_device() -> str:
    """Configure and return the best available device."""
    device = 'cuda' if torch.cuda. is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    return device


def load_model(model_path: str, device: str) -> YOLO:
    """Load the YOLO model."""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    if device == 'cuda':
        model.to(device)
    
    print("✅ Model loaded successfully")
    return model


def setup_webcam(camera_index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """Initialize and configure the webcam."""
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")
    
    print(f"✅ Webcam initialized ({width}x{height})")
    return cap


# ============================================================
# LANE DETECTION SYSTEM
# ============================================================
class LaneDetector:
    """
    Intelligent lane detection using zone-based masks. 
    Each lane (North, South, East, West) has its own polygon mask.
    The junction/roundabout center is a separate zone.
    """
    
    __slots__ = ('lane_masks', 'lane_points', 'frame_shape', '_initialized')
    
    def __init__(self):
        self.lane_masks:  Dict[Lane, Optional[np.ndarray]] = {
            Lane.NORTH: None,
            Lane.SOUTH:  None,
            Lane.EAST: None,
            Lane.WEST: None,
            Lane.JUNCTION: None,
        }
        self.lane_points: Dict[Lane, List[List[int]]] = {
            Lane.NORTH: [],
            Lane.SOUTH: [],
            Lane.EAST: [],
            Lane.WEST: [],
            Lane.JUNCTION: [],
        }
        self.frame_shape:  Optional[Tuple[int, int]] = None
        self._initialized = False
    
    @property
    def is_configured(self) -> bool:
        """Check if lane detection is configured."""
        return self._initialized and any(
            mask is not None for mask in self.lane_masks.values()
        )
    
    def configure_from_points(self, frame_shape: Tuple[int, ... ],
                              lane_points: Dict[str, List[List[int]]]) -> None:
        """Configure lane masks from saved points."""
        self.frame_shape = frame_shape[: 2]
        h, w = self.frame_shape
        
        for lane_name, points in lane_points.items():
            try:
                lane = Lane(lane_name)
                if points and len(points) >= CONFIG.MIN_POLYGON_POINTS:
                    self.lane_points[lane] = points
                    mask = np.zeros((h, w), dtype=np.uint8)
                    pts = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                    self.lane_masks[lane] = mask
            except ValueError:
                continue
        
        self._initialized = True
        configured_lanes = [l. value for l, m in self.lane_masks.items() if m is not None]
        print(f"✅ Lane detection configured: {', '.join(configured_lanes)}")
    
    def detect_lane(self, cx: int, cy: int) -> Lane:
        """
        Detect which lane a point belongs to.
        Priority: Junction > Specific Lanes > Unknown
        """
        if not self._initialized:
            return Lane.UNKNOWN
        
        # Check bounds
        if self.frame_shape:
            h, w = self. frame_shape
            if not (0 <= cx < w and 0 <= cy < h):
                return Lane.UNKNOWN
        
        # Check junction first (highest priority for roundabout center)
        if self.lane_masks[Lane.JUNCTION] is not None:
            if self.lane_masks[Lane.JUNCTION][cy, cx] > 0:
                return Lane.JUNCTION
        
        # Check each lane
        for lane in [Lane.NORTH, Lane.SOUTH, Lane.EAST, Lane.WEST]:
            if self.lane_masks[lane] is not None:
                if self. lane_masks[lane][cy, cx] > 0:
                    return lane
        
        return Lane.UNKNOWN
    
    def get_lane_counts(self, detections: List[Tuple[int, int]]) -> Dict[Lane, int]: 
        """Count detections per lane."""
        counts = {lane: 0 for lane in Lane}
        for cx, cy in detections:
            lane = self.detect_lane(cx, cy)
            counts[lane] += 1
        return counts
    
    def get_visualization_overlay(self, frame: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        """Create lane visualization overlay."""
        if not self._initialized:
            return frame
        
        overlay = frame.copy()
        
        for lane, mask in self.lane_masks.items():
            if mask is not None:
                color = LANE_COLORS.get(lane, COLORS.GRAY)
                colored = np.zeros_like(frame)
                colored[: ] = color
                lane_overlay = cv2.bitwise_and(colored, colored, mask=mask)
                overlay = cv2.addWeighted(overlay, 1, lane_overlay, alpha, 0)
        
        return overlay
    
    def draw_lane_boundaries(self, frame: np.ndarray) -> None:
        """Draw lane boundary lines on frame."""
        for lane, points in self.lane_points.items():
            if points and len(points) >= CONFIG.MIN_POLYGON_POINTS: 
                pts = np.array(points, dtype=np.int32)
                color = LANE_COLORS.get(lane, COLORS.GRAY)
                cv2.polylines(frame, [pts], True, color, 2)
    
    def to_dict(self) -> Dict[str, List[List[int]]]:
        """Convert lane points to dictionary for saving."""
        return {lane.value: points for lane, points in self.lane_points.items() if points}
    
    @classmethod
    def from_dict(cls, data: Dict[str, List[List[int]]], frame_shape: Tuple[int, ... ]) -> 'LaneDetector':
        """Create LaneDetector from saved dictionary."""
        detector = cls()
        if data:
            detector.configure_from_points(frame_shape, data)
        return detector


# ============================================================
# ROI PERSISTENCE (Save/Load)
# ============================================================
def save_roi_config(filepath: str, prototype_points: np.ndarray,
                    road_config: Union[str, List] = None,
                    lane_config: Dict[str, List[List[int]]] = None) -> None:
    """Save ROI and lane configuration to JSON file."""
    config = {
        "prototype_points": prototype_points. tolist() if prototype_points is not None else None,
        "num_points": len(prototype_points) if prototype_points is not None else 0,
        "road_config": road_config if not isinstance(road_config, np.ndarray) else road_config.tolist(),
        "lane_config": lane_config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to {filepath}")


def load_roi_config(filepath: str) -> Tuple[Optional[np.ndarray], Optional[Union[str, List]], Optional[Dict]]:
    """Load ROI and lane configuration from JSON file."""
    if not os.path.exists(filepath):
        return None, None, None
    
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        prototype_points = None
        if config.get("prototype_points"):
            prototype_points = np. array(config["prototype_points"], dtype=np.int32)
        
        road_config = config.get("road_config")
        lane_config = config. get("lane_config")
        num_points = config.get("num_points", 0)
        
        print(f"✅ Configuration loaded from {filepath} ({num_points} prototype points)")
        return prototype_points, road_config, lane_config
    
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"⚠️ Failed to load config: {e}")
        return None, None, None


# ============================================================
# ROI & LANE SETUP INTERFACE
# ============================================================
class ROISetup:
    """Interactive ROI and Lane setup interface."""
    
    __slots__ = ('cap', 'prototype_points', 'road_points', 'lane_points',
                 'current_mode', 'current_lane', 'mouse_pos')
    
    LANE_ORDER = [Lane.NORTH, Lane.SOUTH, Lane.EAST, Lane.WEST, Lane.JUNCTION]
    
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.prototype_points: List[List[int]] = []
        self.road_points: List[List[int]] = []
        self.lane_points: Dict[Lane, List[List[int]]] = {lane: [] for lane in self. LANE_ORDER}
        self.current_mode = "prototype"
        self.current_lane:  Optional[Lane] = None
        self.mouse_pos = (0, 0)
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events."""
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            target_list = self._get_current_target_list()
            if target_list is not None:
                target_list.append([x, y])
                mode_name = self. current_lane.value if self.current_lane else self.current_mode
                print(f"{mode_name} point {len(target_list)}: ({x}, {y})")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            target_list = self._get_current_target_list()
            if target_list: 
                removed = target_list.pop()
                print(f"Removed point:  ({removed[0]}, {removed[1]})")
    
    def _get_current_target_list(self) -> Optional[List[List[int]]]: 
        """Get the current target list based on mode."""
        if self.current_mode == "prototype":
            return self.prototype_points
        elif self.current_mode == "road":
            return self.road_points
        elif self.current_mode == "lane" and self.current_lane:
            return self.lane_points[self.current_lane]
        return None
    
    def run_setup(self) -> Tuple[Optional[np.ndarray], Optional[Union[str, List]], Optional[Dict]]:
        """Run the complete setup wizard."""
        self._print_setup_instructions()
        
        cv2.namedWindow('ROI Setup')
        cv2.setMouseCallback('ROI Setup', self._mouse_callback)
        
        # Step 1: Prototype boundary
        if not self._setup_prototype_boundary():
            return None, None, None
        
        # Step 2: Road selection
        prototype_points, road_config = self._setup_road_selection()
        if prototype_points is None:
            return None, None, None
        
        # Step 3: Lane configuration
        lane_config = self._setup_lanes(prototype_points)
        
        return prototype_points, road_config, lane_config
    
    def _print_setup_instructions(self) -> None:
        """Print setup instructions."""
        print("\n" + "=" * 60)
        print("ROI SETUP - PROTOTYPE BOUNDARY")
        print("=" * 60)
        print("Click points to trace the boundary of your prototype")
        print("Controls:  [R] Reset | [Z] Undo | [N] Next | [Q] Quit")
        print("=" * 60 + "\n")
    
    def _setup_prototype_boundary(self) -> bool:
        """Setup prototype boundary points."""
        self.current_mode = "prototype"
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            display = frame.copy()
            self._draw_header(display, "STEP 1: Trace PROTOTYPE boundary")
            self._draw_polygon_preview(display, self.prototype_points, COLORS.POINT, COLORS.LINE)
            self._draw_point_count(display, len(self.prototype_points), "Points")
            self._draw_controls(display, "Left:  Add | Right:  Undo | [R] Reset | [N] Next | [Q] Quit")
            
            cv2.imshow('ROI Setup', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow('ROI Setup')
                return False
            elif key == ord('r'):
                self.prototype_points. clear()
            elif key == ord('z') and self.prototype_points:
                self.prototype_points.pop()
            elif key == ord('n') and len(self.prototype_points) >= CONFIG.MIN_POLYGON_POINTS:
                return True
        
        return False
    
    def _setup_road_selection(self) -> Tuple[Optional[np.ndarray], Optional[Union[str, List]]]:
        """Setup road area."""
        print("\n" + "=" * 60)
        print("ROAD SELECTION:  [A] Auto | [M] Manual | [S] Skip")
        print("=" * 60 + "\n")
        
        prototype_points = np.array(self.prototype_points, dtype=np.int32)
        
        while True: 
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            display = frame.copy()
            self._draw_prototype_overlay(display, prototype_points)
            self._draw_header(display, "STEP 2: Choose road detection method")
            
            y = 80
            for text in ["[A] Automatic - detect dark areas",
                        "[M] Manual - draw road boundary",
                        "[S] Skip - use entire prototype"]:
                cv2.putText(display, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS.WHITE, 2)
                y += 35
            
            self._draw_controls(display, "[A] Auto | [M] Manual | [S] Skip | [Q] Cancel")
            cv2.imshow('ROI Setup', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                return prototype_points, "auto"
            elif key == ord('m'):
                return self._manual_road_drawing(prototype_points)
            elif key == ord('s'):
                return prototype_points, "full"
            elif key == ord('q'):
                cv2.destroyWindow('ROI Setup')
                return None, None
    
    def _manual_road_drawing(self, prototype_points:  np.ndarray) -> Tuple[np.ndarray, List]: 
        """Draw road boundary manually."""
        self.current_mode = "road"
        self.road_points. clear()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            display = frame.copy()
            self._draw_prototype_overlay(display, prototype_points, dim=True)
            self._draw_header(display, "STEP 2:  Trace ROAD boundary")
            self._draw_polygon_preview(display, self.road_points, COLORS.ROAD_OVERLAY, COLORS.ROAD_OVERLAY)
            self._draw_point_count(display, len(self.road_points), "Road points")
            self._draw_controls(display, "Left: Add | Right: Undo | [R] Reset | [C] Complete | [Q] Cancel")
            
            cv2.imshow('ROI Setup', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return prototype_points, "auto"
            elif key == ord('r'):
                self.road_points.clear()
            elif key == ord('z') and self.road_points:
                self.road_points.pop()
            elif key == ord('c') and len(self.road_points) >= CONFIG.MIN_POLYGON_POINTS:
                return prototype_points, self.road_points. copy()
    
    def _setup_lanes(self, prototype_points: np. ndarray) -> Optional[Dict[str, List[List[int]]]]:
        """Setup lane zones."""
        print("\n" + "=" * 60)
        print("LANE SETUP")
        print("=" * 60)
        print("Define zones for each lane:")
        print("  • NORTH (left side)")
        print("  • SOUTH (right side)")
        print("  • EAST (top)")
        print("  • WEST (bottom)")
        print("  • JUNCTION (center roundabout)")
        print("=" * 60 + "\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            display = frame.copy()
            self._draw_prototype_overlay(display, prototype_points)
            self._draw_header(display, "STEP 3: Setup lane zones? ")
            
            cv2.putText(display, "[Y] Yes - configure lane zones", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS.WHITE, 2)
            cv2.putText(display, "[N] No - skip lane detection", (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS.WHITE, 2)
            
            self._draw_controls(display, "[Y] Yes | [N] No/Skip")
            cv2.imshow('ROI Setup', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                cv2.destroyWindow('ROI Setup')
                return None
            elif key == ord('y'):
                break
            elif key == ord('q'):
                cv2.destroyWindow('ROI Setup')
                return None
        
        # Configure each lane
        for lane in self.LANE_ORDER:
            if not self._setup_single_lane(prototype_points, lane):
                break
        
        cv2.destroyWindow('ROI Setup')
        
        # Return only lanes that have points
        result = {lane. value: points for lane, points in self.lane_points.items()
                  if len(points) >= CONFIG.MIN_POLYGON_POINTS}
        
        return result if result else None
    
    def _setup_single_lane(self, prototype_points: np.ndarray, lane: Lane) -> bool:
        """Setup a single lane zone."""
        self.current_mode = "lane"
        self. current_lane = lane
        self.lane_points[lane].clear()
        
        lane_color = LANE_COLORS.get(lane, COLORS.GRAY)
        
        # Direction hints
        direction_hints = {
            Lane.NORTH: "LEFT side of prototype",
            Lane.SOUTH:  "RIGHT side of prototype",
            Lane.EAST: "TOP of prototype",
            Lane.WEST: "BOTTOM of prototype",
            Lane.JUNCTION: "CENTER roundabout area",
        }
        hint = direction_hints.get(lane, "")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            display = frame.copy()
            
            # Draw existing lane zones
            for prev_lane, points in self.lane_points.items():
                if points and len(points) >= CONFIG.MIN_POLYGON_POINTS and prev_lane != lane:
                    pts = np.array(points, dtype=np.int32)
                    prev_color = LANE_COLORS.get(prev_lane, COLORS.GRAY)
                    overlay = display.copy()
                    cv2.fillPoly(overlay, [pts], prev_color)
                    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
                    cv2.polylines(display, [pts], True, prev_color, 2)
            
            self._draw_prototype_overlay(display, prototype_points, dim=True)
            self._draw_header(display, f"LANE SETUP:  {lane.value} ({hint})")
            self._draw_polygon_preview(display, self.lane_points[lane], lane_color, lane_color)
            self._draw_point_count(display, len(self.lane_points[lane]), f"{lane.value} points")
            self._draw_controls(display, "Left: Add | Right: Undo | [R] Reset | [C] Complete | [S] Skip | [Q] Quit")
            
            cv2.imshow('ROI Setup', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            elif key == ord('s'):
                self.lane_points[lane].clear()
                return True
            elif key == ord('r'):
                self.lane_points[lane].clear()
            elif key == ord('z') and self.lane_points[lane]:
                self.lane_points[lane].pop()
            elif key == ord('c'):
                return True
        
        return False
    
    def _draw_polygon_preview(self, frame: np.ndarray, points: List[List[int]],
                              point_color: Tuple[int, int, int],
                              line_color:  Tuple[int, int, int]) -> None:
        """Draw polygon with preview."""
        num_points = len(points)
        if num_points == 0:
            return
        
        pts_array = np.array(points, dtype=np.int32)
        
        # Draw filled preview
        if num_points >= CONFIG.MIN_POLYGON_POINTS:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_array], point_color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.polylines(frame, [pts_array], True, line_color, 2)
        elif num_points >= 2:
            cv2.polylines(frame, [pts_array], False, line_color, 2)
        
        # Preview lines to mouse
        if num_points >= 1:
            cv2.line(frame, tuple(points[-1]), self.mouse_pos, COLORS.PREVIEW, 1, cv2.LINE_AA)
            if num_points >= 2:
                cv2.line(frame, self.mouse_pos, tuple(points[0]), COLORS.PREVIEW, 1, cv2.LINE_AA)
        
        # Draw points
        for i, pt in enumerate(points):
            cv2.circle(frame, tuple(pt), 8, point_color, -1)
            cv2.circle(frame, tuple(pt), 10, COLORS.WHITE, 2)
            cv2.putText(frame, str(i + 1), (pt[0] + 12, pt[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.WHITE, 2)
    
    def _draw_prototype_overlay(self, frame: np.ndarray, points: np.ndarray, dim: bool = False) -> None:
        """Draw prototype boundary."""
        overlay = frame.copy()
        color = (50, 50, 50) if dim else (0, 100, 0)
        alpha = 0.3 if dim else 0.2
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [points], True, COLORS.ROI_BORDER, 2 if dim else 3)
    
    def _draw_point_count(self, frame: np.ndarray, count: int, label: str) -> None:
        """Draw point count info."""
        cv2.putText(frame, f"{label}: {count} (min {CONFIG.MIN_POLYGON_POINTS})",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS.WHITE, 2)
        
        if count >= CONFIG.MIN_POLYGON_POINTS:
            cv2.putText(frame, "Press [C] to complete or keep adding",
                       (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.POINT, 2)
    
    @staticmethod
    def _draw_header(frame: np.ndarray, text: str) -> None:
        """Draw header."""
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), COLORS.BLACK, -1)
        cv2.putText(frame, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS.CORNER, 2)
    
    @staticmethod
    def _draw_controls(frame: np.ndarray, text: str) -> None:
        """Draw controls bar."""
        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 40), (frame.shape[1], h), COLORS.BLACK, -1)
        cv2.putText(frame, text, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS.LIGHT_GRAY, 1)


# ============================================================
# MASK GENERATION
# ============================================================
def create_prototype_mask(frame_shape: Tuple[int, ... ], points: np.ndarray) -> Optional[np.ndarray]:
    """Create binary mask from prototype points."""
    if points is None or len(points) < CONFIG.MIN_POLYGON_POINTS:
        return None
    
    h, w = frame_shape[: 2]
    mask = np. zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    return mask


def create_road_mask_auto(frame:  np.ndarray, prototype_mask: np.ndarray,
                          sensitivity: int = 80) -> Optional[np.ndarray]:
    """Automatically detect road within prototype."""
    if prototype_mask is None:
        return None
    
    masked_frame = cv2.bitwise_and(frame, frame, mask=prototype_mask)
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    road_mask = cv2.inRange(gray, 1, sensitivity)
    
    hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    road_mask = cv2.bitwise_or(road_mask, yellow_mask)
    
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, KERNEL_5x5, iterations=3)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, KERNEL_5x5, iterations=1)
    cv2.bitwise_and(road_mask, prototype_mask, dst=road_mask)
    
    return road_mask


def create_road_mask_manual(frame_shape: Tuple[int, ... ], road_points: List[List[int]],
                            prototype_mask: np.ndarray) -> Optional[np.ndarray]: 
    """Create road mask from manual points."""
    if road_points is None or len(road_points) < CONFIG.MIN_POLYGON_POINTS:
        return prototype_mask. copy() if prototype_mask is not None else None
    
    h, w = frame_shape[: 2]
    road_mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(road_points, dtype=np.int32)
    cv2.fillPoly(road_mask, [pts], 255)
    
    if prototype_mask is not None: 
        cv2.bitwise_and(road_mask, prototype_mask, dst=road_mask)
    
    return road_mask


# ============================================================
# VISUALIZATION & DETECTION DRAWING
# ============================================================
def create_visualization(frame: np.ndarray, prototype_points: np.ndarray,
                         road_mask: np.ndarray, lane_detector: LaneDetector,
                         show_lanes: bool = True) -> np.ndarray:
    """Create visualization with ROI and lane overlays."""
    display = frame.copy()
    
    # Draw lane zones if configured
    if show_lanes and lane_detector.is_configured: 
        display = lane_detector.get_visualization_overlay(display, alpha=0.2)
        lane_detector.draw_lane_boundaries(display)
    elif road_mask is not None:
        display[road_mask > 0] = cv2.addWeighted(
            display[road_mask > 0], 0.8,
            np.full_like(display[road_mask > 0], COLORS.ROAD_OVERLAY), 0.2, 0
        )
    
    # Draw prototype boundary
    if prototype_points is not None and len(prototype_points) >= CONFIG.MIN_POLYGON_POINTS:
        cv2.polylines(display, [prototype_points], True, COLORS.ROI_BORDER, 2)
    
    return display


def is_detection_in_roi(box, mask:  np.ndarray) -> bool:
    """Check if detection center is within ROI."""
    if mask is None:
        return True
    
    xyxy = box. xyxy[0]. cpu().numpy()
    cx = int((xyxy[0] + xyxy[2]) * 0.5)
    cy = int((xyxy[1] + xyxy[3]) * 0.5)
    
    h, w = mask.shape
    if not (0 <= cx < w and 0 <= cy < h):
        return False
    
    return mask[cy, cx] > 0


def draw_detection(frame: np.ndarray, box, class_id: int, confidence: float,
                   in_roi: bool, lane:  Lane, show_lane: bool = True, 
                   track_id: Optional[int] = None, confirmed: bool = True) -> Tuple[str, bool]:
    """
    Draw detection bounding box with lane information and tracking ID.
    
    The box shows:
    - Track ID (if provided)
    - Vehicle type (Car/Ambulance)
    - Confidence score
    - Lane name (displayed prominently)
    - Lane-colored indicator stripe
    - Confirmation status (border color)
    """
    # Extract bounding box coordinates (handle both numpy array and YOLO result)
    if isinstance(box, np.ndarray):
        # Already in xyxy format (from tracked object)
        xyxy = box
    else:
        # YOLO result object - extract and convert
        xyxy = box.xyxy[0].cpu().numpy().astype(np.int32)
    
    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Determine vehicle type and base color
    if class_id == 0:
        vehicle_type = "Ambulance"
        base_color = COLORS.AMBULANCE
    else:
        vehicle_type = "Car"
        base_color = COLORS.CAR
    
    if not in_roi:
        # Filtered detection (outside ROI)
        box_color = COLORS.GRAY
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
        
        label_text = f"[FILTERED] {vehicle_type}:  {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), box_color, -1)
        cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.WHITE, 1)
        
        return vehicle_type, False
    
    # Valid detection - get lane color
    lane_color = LANE_COLORS.get(lane, COLORS.GRAY)
    
    # Draw main bounding box with vehicle type color
    # Use different thickness/style for confirmed vs tentative
    if confirmed:
        cv2.rectangle(frame, (x1, y1), (x2, y2), base_color, 2)
    else:
        # Tentative track - dashed border
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS.GRAY, 2)
    
    # Draw lane-colored stripe on the left side of box
    stripe_width = max(4, box_width // 20)
    cv2.rectangle(frame, (x1 - stripe_width - 2, y1), (x1 - 2, y2), lane_color, -1)
    
    # Prepare label text with track ID
    if track_id is not None:
        vehicle_label = f"#{track_id} {vehicle_type}: {confidence:.2f}"
    else:
        vehicle_label = f"{vehicle_type}: {confidence:.2f}"
    
    if show_lane and lane != Lane.UNKNOWN:
        # Two-line label:  Vehicle info on top, Lane on bottom
        line1 = vehicle_label
        line2 = f"Lane:  {lane.value}"
        
        # Calculate text sizes
        (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        
        total_width = max(tw1, tw2) + 14
        total_height = th1 + th2 + 20
        
        # Draw label background (two-tone:  vehicle color on top, lane color on bottom)
        # Top part - vehicle info
        cv2.rectangle(frame, (x1, y1 - total_height), (x1 + total_width, y1 - th2 - 10), base_color, -1)
        # Bottom part - lane info
        cv2.rectangle(frame, (x1, y1 - th2 - 10), (x1 + total_width, y1), lane_color, -1)
        
        # Draw text
        cv2.putText(frame, line1, (x1 + 5, y1 - th2 - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS.WHITE, 2, cv2.LINE_AA)
    else:
        # Single line label (no lane info)
        label_text = vehicle_label
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), base_color, -1)
        cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.WHITE, 1)
    
    # Draw small lane indicator badge at bottom-right of box
    if show_lane and lane != Lane.UNKNOWN:
        badge_text = lane.value[0]  # First letter (N, S, E, W, J)
        badge_size = 24
        badge_x = x2 - badge_size - 2
        badge_y = y2 - badge_size - 2
        
        # Badge background
        cv2.rectangle(frame, (badge_x, badge_y), (x2 - 2, y2 - 2), lane_color, -1)
        cv2.rectangle(frame, (badge_x, badge_y), (x2 - 2, y2 - 2), COLORS.WHITE, 1)
        
        # Badge text
        cv2.putText(frame, badge_text, (badge_x + 6, y2 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.WHITE, 2, cv2.LINE_AA)
    
    return vehicle_type, True


def draw_track_trail(frame: np.ndarray, track: TrackedObject, lane_color: Tuple[int, int, int]) -> None:
    """Draw position history trail for a tracked object."""
    if len(track.position_history) < 2:
        return
    
    # Draw trail with fading effect
    num_points = len(track.position_history)
    for i in range(1, num_points):
        pt1 = (int(track.position_history[i-1][0]), int(track.position_history[i-1][1]))
        pt2 = (int(track.position_history[i][0]), int(track.position_history[i][1]))
        
        # Fade from transparent to opaque
        alpha = i / num_points
        thickness = int(2 + alpha * 2)
        
        cv2.line(frame, pt1, pt2, lane_color, thickness, cv2.LINE_AA)
    
    # Draw center point
    cx, cy = int(track.smoothed_center[0]), int(track.smoothed_center[1])
    cv2.circle(frame, (cx, cy), COLORS.TRAIL_CENTER_RADIUS, lane_color, -1)
    cv2.circle(frame, (cx, cy), COLORS.TRAIL_OUTLINE_RADIUS, COLORS.WHITE, 2)


def draw_stats(frame: np.ndarray, fps: float, car_count: int, ambulance_count: int,
               filtered_count: int, roi_active: bool, roi_mode: str,
               lane_counts: Dict[Lane, int], show_lanes: bool,
               tracker: Optional['ObjectTracker'] = None,
               firebase: Optional['FirebaseUploader'] = None) -> None:
    """Draw statistics overlay with lane counts and tracking info."""
    # Calculate panel height based on content
    panel_height = 200
    if tracker is not None:
        panel_height = 240
    if firebase is not None and firebase.config.ENABLED:
        panel_height += 25
    if show_lanes and any(v > 0 for k, v in lane_counts.items() if k != Lane.UNKNOWN):
        panel_height += 120
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, panel_height), COLORS.BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    y = 40
    cv2.putText(frame, "TRAFFIX-AI + Tracking", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS.CORNER, 2)
    
    y += 30
    roi_color = COLORS.POINT if roi_active else (0, 0, 255)
    cv2.putText(frame, f"ROI: {'ON' if roi_active else 'OFF'} ({roi_mode})",
               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 1)
    
    y += 30
    cv2.putText(frame, f"FPS:  {int(fps)}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.WHITE, 1)
    
    # Firebase status
    if firebase is not None and firebase.config.ENABLED:
        y += 25
        status = firebase.get_status()
        fb_color = COLORS.POINT if status['error_count'] == 0 else (0, 165, 255)
        cv2.putText(frame, f"Firebase: {status['upload_count']} uploads, {status['error_count']} errors",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fb_color, 1)
    
    # Tracking stats
    if tracker is not None:
        y += 30
        cv2.putText(frame, "Tracking Stats:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.CORNER, 1)
        
        confirmed = len(tracker.get_confirmed_tracks())
        tentative = len(tracker.get_tentative_tracks())
        
        y += 22
        cv2.putText(frame, f"  Active: {confirmed} confirmed, {tentative} tentative",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.WHITE, 1)
        
        y += 22
        cv2.putText(frame, f"  Total tracked: {tracker.total_tracked}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.WHITE, 1)
    
    y += 30
    cv2.putText(frame, f"Cars: {car_count}  |  Ambulances: {ambulance_count}",
               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.WHITE, 1)
    
    # Lane counts with colored indicators
    if show_lanes and any(v > 0 for k, v in lane_counts.items() if k != Lane.UNKNOWN):
        y += 30
        cv2.putText(frame, "Vehicles per Lane:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.CORNER, 1)
        
        for lane in [Lane.NORTH, Lane.SOUTH, Lane.EAST, Lane.WEST, Lane.JUNCTION]:
            count = lane_counts.get(lane, 0)
            y += 22
            lane_color = LANE_COLORS.get(lane, COLORS.GRAY)
            
            # Draw colored square indicator
            cv2.rectangle(frame, (20, y - 12), (32, y), lane_color, -1)
            cv2.rectangle(frame, (20, y - 12), (32, y), COLORS.WHITE, 1)
            
            # Lane name and count
            cv2.putText(frame, f"{lane.value}: {count}",
                       (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.WHITE, 1)


def draw_controls(frame: np.ndarray) -> None:
    """Draw control bar."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 35), (w, h), COLORS.BLACK, -1)
    # Split controls into two lines for better readability
    cv2.putText(frame, "[Q] Quit | [R] ROI | [L] Lanes | [S] Setup | [D] Delete",
               (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS.LIGHT_GRAY, 1)
    cv2.putText(frame, "[C] Clear Tracks | [T] Trails | [+/-] Sensitivity",
               (20, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS.LIGHT_GRAY, 1)


# ============================================================
# FPS COUNTER
# ============================================================
class FPSCounter: 
    """Efficient FPS counter."""
    
    __slots__ = ('frame_count', 'start_time', 'fps')
    
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.fps = 0.0
    
    def update(self) -> float:
        """Update and return FPS."""
        self.frame_count += 1
        elapsed = time.perf_counter() - self.start_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.perf_counter()
        
        return self.fps


# ============================================================
# MAIN APPLICATION
# ============================================================
class TraffixAI:
    """Main application class."""
    
    def __init__(self):
        self.device = setup_device()
        self.model = load_model(CONFIG.MODEL_PATH, self.device)
        self.cap = setup_webcam(CONFIG.CAMERA_INDEX, CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT)
        
        # Load configuration
        self.prototype_points, self.road_config, lane_config = load_roi_config(CONFIG.ROI_FILE)
        
        # State
        self.roi_active = self.prototype_points is not None
        self.sensitivity = CONFIG.ROAD_SENSITIVITY
        self.roi_mode = "none"
        self.show_lanes = True
        self.show_trails = True
        
        # Masks
        self.prototype_mask:  Optional[np.ndarray] = None
        self.road_mask: Optional[np.ndarray] = None
        
        # Lane detector
        self.lane_detector = LaneDetector()
        
        # Object tracker
        self.tracker = ObjectTracker()
        
        # Firebase uploader
        self.firebase = FirebaseUploader(FIREBASE_CONFIG)
        
        # Counters
        self.fps_counter = FPSCounter()
        self.update_counter = 0
        
        # Initialize
        self._initialize_masks(lane_config)
    
    def _initialize_masks(self, lane_config: Optional[Dict]) -> None:
        """Initialize masks from saved configuration."""
        if self.prototype_points is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.prototype_mask = create_prototype_mask(frame. shape, self.prototype_points)
        
        if self.road_config == "auto":
            self.road_mask = create_road_mask_auto(frame, self.prototype_mask, self.sensitivity)
            self.roi_mode = "auto"
        elif self.road_config == "full":
            self.road_mask = self.prototype_mask. copy()
            self.roi_mode = "full"
        elif isinstance(self.road_config, list):
            self.road_mask = create_road_mask_manual(frame. shape, self.road_config, self.prototype_mask)
            self.roi_mode = "manual"
        
        # Initialize lane detector
        if lane_config:
            self. lane_detector.configure_from_points(frame. shape, lane_config)
    
    def run(self) -> None:
        """Main application loop."""
        self._print_instructions()
        use_half = self.device == 'cuda'
        
        try:
            while True:
                ret, frame = self.cap. read()
                if not ret: 
                    continue
                
                self.update_counter += 1
                
                # Update road mask periodically
                if (self.roi_mode == "auto" and
                    self.update_counter % CONFIG. MASK_UPDATE_INTERVAL == 0 and
                    self.prototype_mask is not None):
                    self.road_mask = create_road_mask_auto(frame, self.prototype_mask, self.sensitivity)
                
                # Create display
                if self.roi_active and self.prototype_points is not None:
                    display_frame = create_visualization(
                        frame, self.prototype_points, self.road_mask,
                        self.lane_detector, self.show_lanes
                    )
                else: 
                    display_frame = frame. copy()
                
                # Run detection
                with torch.no_grad():
                    results = self.model(
                        frame,
                        conf=CONFIG.CONFIDENCE_THRESHOLD,
                        iou=CONFIG. IOU_THRESHOLD,
                        device=self.device,
                        half=use_half,
                        verbose=False
                    )[0]
                
                # Process detections
                car_count, ambulance_count, filtered_count, lane_counts, lane_ambulances = self._process_detections(
                    display_frame, results
                )
                
                # Update FPS
                fps = self.fps_counter.update()
                
                # Upload to Firebase
                self._upload_to_firebase(car_count, ambulance_count, lane_counts, lane_ambulances, fps)
                
                # Draw UI
                draw_stats(display_frame, fps, car_count, ambulance_count,
                          filtered_count, self.roi_active, self.roi_mode,
                          lane_counts, self.show_lanes and self.lane_detector.is_configured,
                          self.tracker, self.firebase)
                draw_controls(display_frame)
                
                cv2.imshow('TRAFFIX-AI', display_frame)
                
                if not self._handle_input(frame):
                    break
        
        finally:
            self.firebase.stop()
            self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Cleanup complete")
    
    def _process_detections(self, frame: np.ndarray, results) -> Tuple[int, int, int, Dict[Lane, int], Dict[Lane, bool]]:
        """Process detections with filtering, duplicate removal, and tracking."""
        car_count = 0
        ambulance_count = 0
        filtered_count = 0
        lane_counts = {lane: 0 for lane in Lane}
        lane_ambulances = {lane: False for lane in Lane}  # Track ambulances per lane
        
        active_mask = self.road_mask if self.roi_active else None
        
        # Convert YOLO results to Detection objects with filtering
        detections = []
        for box in results. boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0]. cpu().numpy()
            
            detection = Detection(
                box=xyxy,
                class_id=class_id,
                confidence=confidence
            )
            
            # Apply filtering
            if not detection.is_valid():
                filtered_count += 1
                continue
            
            # Check ROI
            if active_mask is not None:
                if not is_detection_in_roi(box, active_mask):
                    filtered_count += 1
                    continue
            
            detections.append(detection)
        
        # Remove duplicates
        detections = remove_duplicate_detections(detections)
        
        # Detect lanes for each detection
        lanes = []
        for detection in detections:
            cx, cy = detection.center
            lane = Lane.UNKNOWN
            if self.lane_detector.is_configured:
                lane = self.lane_detector.detect_lane(cx, cy)
            lanes.append(lane)
        
        # Update tracker
        self.tracker.update(detections, lanes)
        
        # Draw tracks with trails
        if self.show_trails:
            for track in self.tracker.tracks:
                lane_color = LANE_COLORS.get(track.stable_lane, COLORS.GRAY)
                draw_track_trail(frame, track, lane_color)
        
        # Draw tracked objects
        for track in self.tracker.tracks:
            lane = track.stable_lane
            
            # Count for lane
            if lane != Lane.UNKNOWN:
                lane_counts[lane] += 1
                # Track ambulances in each lane
                if track.class_id == 0:  # Ambulance
                    lane_ambulances[lane] = True
            
            # Count by vehicle type (only confirmed tracks)
            if track.confirmed:
                if track.vehicle_type == "Car":
                    car_count += 1
                else:
                    ambulance_count += 1
            
            # Draw detection box
            show_lane_info = self.show_lanes and self.lane_detector.is_configured
            draw_detection(
                frame, 
                track.get_display_box(),
                track.class_id,
                track.avg_confidence,
                True,  # in_roi (already filtered)
                lane,
                show_lane_info,
                track_id=track.track_id,
                confirmed=track.confirmed
            )
        
        return car_count, ambulance_count, filtered_count, lane_counts, lane_ambulances
    
    def _upload_to_firebase(self, car_count: int, ambulance_count: int, 
                           lane_counts: Dict[Lane, int], lane_ambulances: Dict[Lane, bool],
                           fps: float) -> None:
        """Prepare and upload traffic data to Firebase."""
        # Prepare lane data
        lanes_data = {}
        for lane in Lane:
            # Skip UNKNOWN lane
            if lane == Lane.UNKNOWN:
                continue
            lane_name = lane.value.lower()
            lanes_data[lane_name] = {
                "count": lane_counts.get(lane, 0),
                "has_ambulance": lane_ambulances.get(lane, False)
            }
        
        # Create TrafficData object
        traffic_data = TrafficData(
            timestamp=int(time.time()),
            total_vehicles=car_count + ambulance_count,
            cars=car_count,
            ambulances=ambulance_count,
            emergency_active=ambulance_count > 0,
            lanes=lanes_data,
            fps=fps,
            camera_online=True
        )
        
        # Upload to Firebase
        self.firebase.upload_lane_data(traffic_data)
    
    def _handle_input(self, frame: np.ndarray) -> bool:
        """Handle keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.roi_active = not self.roi_active
            print(f"ROI {'enabled' if self.roi_active else 'disabled'}")
        elif key == ord('l'):
            self.show_lanes = not self.show_lanes
            print(f"Lane display {'enabled' if self.show_lanes else 'disabled'}")
        elif key == ord('c'):
            self.tracker.clear_all_tracks()
            print("All tracks cleared")
        elif key == ord('t'):
            self.show_trails = not self.show_trails
            print(f"Track trails {'enabled' if self.show_trails else 'disabled'}")
        elif key == ord('s'):
            self._run_setup(frame)
        elif key == ord('d'):
            self._delete_config()
        elif key in (ord('+'), ord('=')):
            self._adjust_sensitivity(5, frame)
        elif key in (ord('-'), ord('_')):
            self._adjust_sensitivity(-5, frame)
        
        return True
    
    def _run_setup(self, frame: np.ndarray) -> None:
        """Run setup wizard."""
        setup = ROISetup(self.cap)
        new_points, new_road_config, new_lane_config = setup.run_setup()
        
        if new_points is None:
            return
        
        self.prototype_points = new_points
        self.prototype_mask = create_prototype_mask(frame.shape, self.prototype_points)
        
        if new_road_config == "auto":
            self.road_mask = create_road_mask_auto(frame, self.prototype_mask, self.sensitivity)
            self.roi_mode = "auto"
        elif new_road_config == "full":
            self.road_mask = self. prototype_mask.copy()
            self.roi_mode = "full"
        elif isinstance(new_road_config, list):
            self.road_mask = create_road_mask_manual(frame. shape, new_road_config, self.prototype_mask)
            self.roi_mode = "manual"
        
        # Configure lane detector
        if new_lane_config:
            self.lane_detector.configure_from_points(frame.shape, new_lane_config)
        else:
            self.lane_detector = LaneDetector()
        
        # Save configuration
        save_roi_config(CONFIG.ROI_FILE, self.prototype_points,
                       new_road_config if not isinstance(new_road_config, list) else new_road_config,
                       new_lane_config)
        
        self.roi_active = True
        print(f"✅ Setup complete")
    
    def _delete_config(self) -> None:
        """Delete configuration."""
        if os.path.exists(CONFIG.ROI_FILE):
            os.remove(CONFIG.ROI_FILE)
        
        self.prototype_points = None
        self.prototype_mask = None
        self. road_mask = None
        self.roi_active = False
        self. roi_mode = "none"
        self.lane_detector = LaneDetector()
        self.tracker.clear_all_tracks()
        print("Configuration deleted")
    
    def _adjust_sensitivity(self, delta: int, frame: np.ndarray) -> None:
        """Adjust sensitivity."""
        self. sensitivity = max(30, min(150, self.sensitivity + delta))
        if self.roi_mode == "auto" and self.prototype_mask is not None:
            self. road_mask = create_road_mask_auto(frame, self. prototype_mask, self.sensitivity)
        print(f"Sensitivity:  {self.sensitivity}")
    
    @staticmethod
    def _print_instructions() -> None:
        """Print instructions."""
        print("\n" + "=" * 60)
        print("Controls:")
        print("  [Q] - Quit")
        print("  [R] - Toggle ROI")
        print("  [L] - Toggle lane display")
        print("  [S] - Setup ROI & lanes")
        print("  [D] - Delete configuration")
        print("  [C] - Clear all tracks")
        print("  [T] - Toggle track trails")
        print("  [+/-] - Adjust sensitivity")
        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    print("=" * 60)
    print("TRAFFIX-AI - Traffic Detection with Lane Detection")
    print("=" * 60)
    
    app = TraffixAI()
    app.run()


if __name__ == "__main__": 
    main()
    