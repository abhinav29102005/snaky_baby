import cv2
import numpy as np
import mediapipe as mp
import random
import time
import threading
from enum import Enum
from collections import deque
import math

class GameState(Enum):
    MENU = 1
    PLAYER_COUNT_SELECTION = 2
    PLAYER_NAME_INPUT = 3
    CALIBRATION = 4
    GAME_INSTRUCTIONS = 5
    PLAYING = 6
    PLAYER_TRANSITION = 7
    FINAL_RESULTS = 8

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class AdvancedGestureTracker:
    def __init__(self):
        self.position_history = deque(maxlen=10)
        self.direction_history = deque(maxlen=5)
        self.calibration_samples = []
        self.movement_baseline = None
        self.sensitivity = 0.03
        self.dead_zone = 0.015
        self.gesture_confidence = 0.0
        self.last_stable_direction = None
        self.direction_lock_time = 0
        self.direction_lock_duration = 500  # ms
        
    def add_position(self, palm_pos):
        """Add palm position with timestamp"""
        timestamp = time.time() * 1000
        self.position_history.append((palm_pos.x, palm_pos.y, timestamp))
        
    def calibrate_movement_range(self, palm_pos):
        """Calibrate user's natural movement range"""
        self.calibration_samples.append((palm_pos.x, palm_pos.y))
        
        if len(self.calibration_samples) >= 30:
            # Calculate movement baseline from samples
            xs = [s[0] for s in self.calibration_samples]
            ys = [s[1] for s in self.calibration_samples]
            
            self.movement_baseline = {
                'center_x': np.mean(xs),
                'center_y': np.mean(ys),
                'range_x': np.std(xs) * 2,
                'range_y': np.std(ys) * 2
            }
            return True
        return False
        
    def get_smoothed_direction(self):
        """Get direction with advanced smoothing and confidence"""
        if len(self.position_history) < 3:
            return None, 0.0
            
        current_time = time.time() * 1000
        
        # Check direction lock
        if (current_time - self.direction_lock_time < self.direction_lock_duration and 
            self.last_stable_direction is not None):
            return self.last_stable_direction, 1.0
            
        # Calculate weighted movement vector
        recent_positions = list(self.position_history)[-5:]
        if len(recent_positions) < 3:
            return None, 0.0
            
        # Use exponential weighting (more recent positions have higher weight)
        weights = np.exp(np.linspace(0, 2, len(recent_positions)))
        weights = weights / np.sum(weights)
        
        # Calculate movement vector
        start_pos = recent_positions[0]
        end_pos = recent_positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Apply calibration if available
        if self.movement_baseline:
            dx = dx / max(self.movement_baseline['range_x'], 0.01)
            dy = dy / max(self.movement_baseline['range_y'], 0.01)
        
        # Calculate movement magnitude and confidence
        magnitude = math.sqrt(dx*dx + dy*dy)
        
        if magnitude < self.dead_zone:
            return None, 0.0
            
        # Determine direction with confidence
        confidence = min(magnitude / self.sensitivity, 1.0)
        
        if confidence < 0.3:
            return None, confidence
            
        # Get primary direction
        if abs(dx) > abs(dy):
            direction = Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            direction = Direction.DOWN if dy > 0 else Direction.UP
            
        # Update direction history for stability
        self.direction_history.append(direction)
        
        # Check for consistent direction
        if len(self.direction_history) >= 3:
            recent_dirs = list(self.direction_history)[-3:]
            if recent_dirs.count(direction) >= 2:
                if direction != self.last_stable_direction:
                    self.last_stable_direction = direction
                    self.direction_lock_time = current_time
                return direction, confidence
                
        return None, confidence

class EnhancedSnakeGame:
    def __init__(self):
        # Initialize MediaPipe with enhanced settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.9
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize advanced gesture tracker
        self.gesture_tracker = AdvancedGestureTracker()
        
        # Fullscreen and window management
        self.fullscreen = False
        self.original_window_size = None
        self.screen_width = 1920  # Will be updated with actual screen size
        self.screen_height = 1080
        self.window_name = '🐍 Enhanced Gesture Snake Game 🎮'
        
        # Dynamic window settings (will adapt to screen size)
        self.WINDOW_WIDTH = 1200
        self.WINDOW_HEIGHT = 800
        
        # Game area settings (will scale with window)
        self.GRID_SIZE = 25
        self.GAME_AREA_WIDTH = 500
        self.GAME_AREA_HEIGHT = 500
        self.GAME_AREA_X = 50
        self.GAME_AREA_Y = 150
        
        # Camera feed settings (will scale)
        self.CAMERA_WIDTH = 320
        self.CAMERA_HEIGHT = 240
        self.CAMERA_X = 0
        self.CAMERA_Y = 0
        
        self.update_layout()
        
        # Enhanced colors with better contrast
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.BRIGHT_GREEN = (100, 255, 100)
        self.RED = (0, 0, 255)
        self.BLUE = (255, 0, 0)
        self.YELLOW = (0, 255, 255)
        self.PURPLE = (255, 0, 255)
        self.ORANGE = (0, 165, 255)
        self.DARK_BLUE = (139, 69, 19)
        self.LIGHT_BLUE = (255, 200, 100)
        self.CYAN = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (64, 64, 64)
        self.NEON_GREEN = (57, 255, 20)
        self.NEON_PINK = (255, 20, 147)
        
        # Game state
        self.state = GameState.MENU
        self.players = []
        self.current_player_index = 0
        self.max_players = 1
        self.current_player_name = ""
        self.calibration_progress = 0
        
        # Snake game variables
        self.snake = []
        self.food = None
        self.direction = Direction.RIGHT
        self.score = 0
        self.game_running = False
        self.last_move_time = 0
        self.move_delay = 120
        self.paused = False
        self.game_start_time = 0
        
        # Enhanced gesture control variables
        self.last_palm_pos = None
        self.palm_center = None
        self.fist_detected = False
        self.last_fist_state = False
        self.hand_detected = False
        self.tracking_quality = 0.0
        
        # UI Animation variables
        self.blink_counter = 0
        self.menu_animation = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Initialize camera with enhanced settings
        self.cap = cv2.VideoCapture(0)
        self.setup_camera()
        
        # Camera feed frame
        self.camera_frame = None
        self.camera_thread = None
        self.running = True
        self.start_camera_thread()
        
        # Player colors with better visibility
        self.player_colors = [
            self.NEON_GREEN, self.NEON_PINK, self.CYAN, self.ORANGE
        ]
        
        # Initialize window
        self.setup_window()
    
    def setup_camera(self):
        """Setup camera with optimal settings"""
        if self.cap.isOpened():
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Auto-exposure and focus settings
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    
    def setup_window(self):
        """Setup OpenCV window with proper configuration"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        
        # Get actual screen resolution
        import tkinter as tk
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Center window on screen
        pos_x = (self.screen_width - self.WINDOW_WIDTH) // 2
        pos_y = (self.screen_height - self.WINDOW_HEIGHT) // 2
        cv2.moveWindow(self.window_name, pos_x, pos_y)
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode safely"""
        if not self.fullscreen:
            # Enter fullscreen
            self.original_window_size = (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            # Update dimensions for fullscreen
            self.WINDOW_WIDTH = self.screen_width
            self.WINDOW_HEIGHT = self.screen_height
            self.fullscreen = True
        else:
            # Exit fullscreen
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            
            if self.original_window_size:
                self.WINDOW_WIDTH, self.WINDOW_HEIGHT = self.original_window_size
                cv2.resizeWindow(self.window_name, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
                
                # Re-center window
                pos_x = (self.screen_width - self.WINDOW_WIDTH) // 2
                pos_y = (self.screen_height - self.WINDOW_HEIGHT) // 2
                cv2.moveWindow(self.window_name, pos_x, pos_y)
            
            self.fullscreen = False
        
        self.update_layout()
    
    def update_layout(self):
        """Update layout based on current window size"""
        # Scale game area
        scale_factor = min(self.WINDOW_WIDTH / 1200, self.WINDOW_HEIGHT / 800)
        
        self.GAME_AREA_WIDTH = int(500 * scale_factor)
        self.GAME_AREA_HEIGHT = int(500 * scale_factor)
        self.GAME_AREA_X = int(50 * scale_factor)
        self.GAME_AREA_Y = int(150 * scale_factor)
        
        # Scale camera feed
        self.CAMERA_WIDTH = int(320 * scale_factor)
        self.CAMERA_HEIGHT = int(240 * scale_factor)
        self.CAMERA_X = self.WINDOW_WIDTH - self.CAMERA_WIDTH - int(20 * scale_factor)
        self.CAMERA_Y = int(20 * scale_factor)
        
        # Ensure grid size scales properly
        self.GRID_SIZE = max(15, int(25 * scale_factor))
    
    def start_camera_thread(self):
        """Start enhanced camera capture in separate thread"""
        def capture_frames():
            frame_count = 0
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    # Enhanced frame processing
                    frame = cv2.flip(frame, 1)
                    
                    # Apply image enhancement
                    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
                    
                    # Gaussian blur for noise reduction
                    frame = cv2.GaussianBlur(frame, (3, 3), 0)
                    
                    self.camera_frame = frame
                    frame_count += 1
                
                time.sleep(0.025)  # ~40 FPS for better responsiveness
        
        self.camera_thread = threading.Thread(target=capture_frames, daemon=True)
        self.camera_thread.start()
    
    def detect_advanced_gesture(self, landmarks):
        """Advanced gesture detection with confidence scoring"""
        if not landmarks:
            return None, False, False, 0.0
            
        # Enhanced palm center calculation
        key_points = [0, 5, 9, 13, 17]  # Wrist and knuckles
        palm_x = np.mean([landmarks[i].x for i in key_points])
        palm_y = np.mean([landmarks[i].y for i in key_points])
        palm_center = type('obj', (object,), {'x': palm_x, 'y': palm_y})()
        
        # Advanced fist detection with confidence
        finger_tips = [4, 8, 12, 16, 20]
        finger_mcp = [2, 5, 9, 13, 17]
        
        closed_fingers = 0
        finger_confidences = []
        
        for i, (tip, mcp) in enumerate(zip(finger_tips, finger_mcp)):
            if i == 0:  # Thumb
                is_closed = landmarks[tip].x < landmarks[mcp].x - 0.02
            else:  # Other fingers
                is_closed = landmarks[tip].y > landmarks[mcp].y + 0.02
            
            if is_closed:
                closed_fingers += 1
            
            # Calculate finger bend angle for confidence
            bend_amount = abs(landmarks[tip].y - landmarks[mcp].y) if i > 0 else abs(landmarks[tip].x - landmarks[mcp].x)
            finger_confidences.append(bend_amount)
        
        # Fist confidence based on how many fingers are closed
        fist_confidence = closed_fingers / 5.0
        fist_detected = closed_fingers >= 4
        
        # Open hand detection
        open_hand = closed_fingers <= 1
        
        # Overall tracking confidence
        tracking_confidence = np.mean(finger_confidences) * 2
        tracking_confidence = min(tracking_confidence, 1.0)
        
        return palm_center, fist_detected, open_hand, tracking_confidence
    
    def draw_calibration_screen(self, frame):
        """Draw calibration screen with visual guidance"""
        # Title
        cv2.putText(frame, "HAND TRACKING CALIBRATION", 
                   (self.WINDOW_WIDTH//2 - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.YELLOW, 3)
        
        current_player = self.players[self.current_player_index] if self.players else {'name': 'Player'}
        cv2.putText(frame, f"{current_player['name']}, please calibrate your hand tracking", 
                   (self.WINDOW_WIDTH//2 - 350, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.WHITE, 2)
        
        # Instructions
        instructions = [
            "1. Show your open palm to the camera",
            "2. Slowly move your hand in different directions", 
            "3. Keep your hand visible until calibration completes",
            "4. Try to use natural, comfortable movements"
        ]
        
        y_start = 200
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (100, y_start + i * 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.CYAN, 2)
        
        # Progress bar
        progress_width = 400
        progress_height = 30
        progress_x = (self.WINDOW_WIDTH - progress_width) // 2
        progress_y = 400
        
        # Background
        cv2.rectangle(frame, (progress_x, progress_y), 
                     (progress_x + progress_width, progress_y + progress_height), 
                     self.DARK_GRAY, -1)
        cv2.rectangle(frame, (progress_x, progress_y), 
                     (progress_x + progress_width, progress_y + progress_height), 
                     self.WHITE, 2)
        
        # Progress fill
        fill_width = int((len(self.gesture_tracker.calibration_samples) / 30) * progress_width)
        if fill_width > 0:
            cv2.rectangle(frame, (progress_x, progress_y), 
                         (progress_x + fill_width, progress_y + progress_height), 
                         self.GREEN, -1)
        
        # Progress text
        progress_text = f"Calibration: {len(self.gesture_tracker.calibration_samples)}/30 samples"
        cv2.putText(frame, progress_text, (progress_x, progress_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 2)
        
        # Tracking quality indicator
        if self.tracking_quality > 0:
            quality_text = f"Tracking Quality: {self.tracking_quality:.1%}"
            quality_color = self.GREEN if self.tracking_quality > 0.7 else self.ORANGE if self.tracking_quality > 0.4 else self.RED
            cv2.putText(frame, quality_text, (progress_x, progress_y + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
    
    def draw_enhanced_camera_feed(self, frame):
        """Draw enhanced camera feed with advanced tracking visualization"""
        if self.camera_frame is not None:
            camera_display = cv2.resize(self.camera_frame, (self.CAMERA_WIDTH, self.CAMERA_HEIGHT))
            
            # Process for hand detection
            rgb_frame = cv2.cvtColor(camera_display, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            self.hand_detected = False
            tracking_info = ""
            
            if results.multi_hand_landmarks:
                self.hand_detected = True
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw enhanced hand landmarks
                    self.mp_draw.draw_landmarks(
                        camera_display, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=3),
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=(255, 255, 255), thickness=2)
                    )
                    
                    # Advanced gesture detection
                    palm_pos, fist_detected, open_hand, tracking_confidence = self.detect_advanced_gesture(hand_landmarks.landmark)
                    self.tracking_quality = tracking_confidence
                    
                    if palm_pos:
                        # Handle calibration
                        if self.state == GameState.CALIBRATION:
                            if tracking_confidence > 0.6:
                                completed = self.gesture_tracker.calibrate_movement_range(palm_pos)
                                if completed:
                                    self.state = GameState.GAME_INSTRUCTIONS
                        
                        # Handle gameplay
                        elif self.state == GameState.PLAYING:
                            # Add position to tracker
                            self.gesture_tracker.add_position(palm_pos)
                            
                            # Handle fist gesture for pause/resume
                            if fist_detected and not self.last_fist_state:
                                self.paused = not self.paused
                            self.last_fist_state = fist_detected
                            
                            # Update snake direction with advanced tracking
                            if not self.paused and open_hand:
                                direction, confidence = self.gesture_tracker.get_smoothed_direction()
                                if direction and confidence > 0.5:
                                    # Check if direction change is valid
                                    opposite_dirs = {
                                        Direction.UP: Direction.DOWN,
                                        Direction.DOWN: Direction.UP,
                                        Direction.LEFT: Direction.RIGHT,
                                        Direction.RIGHT: Direction.LEFT
                                    }
                                    
                                    if direction != opposite_dirs.get(self.direction):
                                        self.direction = direction
                        
                        # Visual indicators
                        palm_screen_x = int(palm_pos.x * self.CAMERA_WIDTH)
                        palm_screen_y = int(palm_pos.y * self.CAMERA_HEIGHT)
                        
                        # Palm center with confidence ring
                        confidence_radius = int(10 + tracking_confidence * 15)
                        palm_color = self.RED if fist_detected else self.GREEN
                        cv2.circle(camera_display, (palm_screen_x, palm_screen_y), 
                                 confidence_radius, palm_color, 2)
                        cv2.circle(camera_display, (palm_screen_x, palm_screen_y), 
                                 5, palm_color, -1)
                        
                        # Direction indicator
                        if self.state == GameState.PLAYING and not self.paused:
                            direction, confidence = self.gesture_tracker.get_smoothed_direction()
                            if direction and confidence > 0.3:
                                # Draw direction arrow
                                arrow_length = int(30 * confidence)
                                end_x = palm_screen_x
                                end_y = palm_screen_y
                                
                                if direction == Direction.UP:
                                    end_y -= arrow_length
                                elif direction == Direction.DOWN:
                                    end_y += arrow_length
                                elif direction == Direction.LEFT:
                                    end_x -= arrow_length
                                elif direction == Direction.RIGHT:
                                    end_x += arrow_length
                                
                                cv2.arrowedLine(camera_display, 
                                              (palm_screen_x, palm_screen_y),
                                              (end_x, end_y), 
                                              self.CYAN, 3, tipLength=0.3)
                        
                        tracking_info = f"Quality: {tracking_confidence:.1%}"
            
            # Draw camera feed with enhanced border
            border_color = self.GREEN if self.hand_detected else self.RED
            cv2.rectangle(frame, (self.CAMERA_X-3, self.CAMERA_Y-3), 
                         (self.CAMERA_X + self.CAMERA_WIDTH + 3, self.CAMERA_Y + self.CAMERA_HEIGHT + 3), 
                         border_color, 3)
            
            # Place camera feed
            frame[self.CAMERA_Y:self.CAMERA_Y + self.CAMERA_HEIGHT, 
                  self.CAMERA_X:self.CAMERA_X + self.CAMERA_WIDTH] = camera_display
            
            # Enhanced status display
            status_y = self.CAMERA_Y + self.CAMERA_HEIGHT + 15
            
            # Hand detection status
            status_text = "✅ HAND DETECTED" if self.hand_detected else "❌ NO HAND"
            status_color = self.GREEN if self.hand_detected else self.RED
            cv2.putText(frame, status_text, (self.CAMERA_X, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            # Tracking quality
            if tracking_info:
                cv2.putText(frame, tracking_info, (self.CAMERA_X, status_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.WHITE, 1)
            
            # Gesture state
            if self.state == GameState.PLAYING:
                gesture_text = "🤛 PAUSED" if self.paused else "✋ ACTIVE"
                cv2.putText(frame, gesture_text, (self.CAMERA_X, status_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.YELLOW, 1)
    
    def draw_fps_counter(self, frame):
        """Draw FPS counter and performance info"""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
        
        if hasattr(self, 'current_fps'):
            fps_text = f"FPS: {self.current_fps:.1f}"
            fps_color = self.GREEN if self.current_fps > 25 else self.ORANGE if self.current_fps > 15 else self.RED
            cv2.putText(frame, fps_text, (10, self.WINDOW_HEIGHT - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
            
            # Window size info
            size_text = f"Window: {self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}"
            if self.fullscreen:
                size_text += " (Fullscreen)"
            cv2.putText(frame, size_text, (10, self.WINDOW_HEIGHT - 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.GRAY, 1)
    
    def init_snake(self):
        """Initialize snake game with better starting position"""
        grid_width = self.GAME_AREA_WIDTH // self.GRID_SIZE
        grid_height = self.GAME_AREA_HEIGHT // self.GRID_SIZE
        
        start_x = grid_width // 2
        start_y = grid_height // 2
        
        # Start with a small snake
        self.snake = [(start_x, start_y), (start_x-1, start_y), (start_x-2, start_y)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.game_running = True
        self.paused = False
        self.game_start_time = time.time()
        
        # Reset gesture tracker for new game
        self.gesture_tracker.position_history.clear()
        self.gesture_tracker.direction_history.clear()
        self.gesture_tracker.last_stable_direction = None
        
        self.spawn_food()
    
    def spawn_food(self):
        """Spawn food at random location avoiding snake"""
        grid_width = self.GAME_AREA_WIDTH // self.GRID_SIZE
        grid_height = self.GAME_AREA_HEIGHT // self.GRID_SIZE
        
        attempts = 0
        while attempts < 100:
            x = random.randint(0, grid_width - 1)
            y = random.randint(0, grid_height - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break
            attempts += 1
    
    def update_snake(self):
        """Enhanced snake update with better collision detection"""
        if not self.game_running or self.paused:
            return
            
        current_time = time.time() * 1000
        if current_time - self.last_move_time < self.move_delay:
            return
        
        self.last_move_time = current_time
        
        head_x, head_y = self.snake[0]
        
        # Move based on direction
        if self.direction == Direction.UP:
            head_y -= 1
        elif self.direction == Direction.DOWN:
            head_y += 1
        elif self.direction == Direction.LEFT:
            head_x -= 1
        elif self.direction == Direction.RIGHT:
            head_x += 1
        
        new_head = (head_x, head_y)
        
        # Check collision with walls
        grid_width = self.GAME_AREA_WIDTH // self.GRID_SIZE
        grid_height = self.GAME_AREA_HEIGHT // self.GRID_SIZE
        
        if (head_x < 0 or head_x >= grid_width or 
            head_y < 0 or head_y >= grid_height or 
            new_head in self.snake):
            self.game_running = False
            return
        
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 10
            # Adaptive difficulty - speed increases with score
            if self.score % 50 == 0 and self.move_delay > 60:
                self.move_delay -= 5
            self.spawn_food()
        else:
            self.snake.pop()
    
    def draw_enhanced_ui(self, frame):
        """Draw enhanced UI with dynamic backgrounds"""
        # Dynamic background based on game state
        if self.state == GameState.PLAYING:
            # Subtle animated background during gameplay
            for i in range(0, self.WINDOW_HEIGHT, 40):
                intensity = int(15 + 10 * abs(np.sin((i + self.menu_animation) * 0.01)))
                cv2.line(frame, (0, i), (self.WINDOW_WIDTH, i), (0, intensity, 0), 1)
        else:
            # Gradient background for menus
            for i in range(self.WINDOW_HEIGHT):
                intensity = int(30 * (1 - i / self.WINDOW_HEIGHT))
                cv2.line(frame, (0, i), (self.WINDOW_WIDTH, i), (intensity, intensity, intensity), 1)
        
        self.menu_animation += 1
    
    def draw_snake_game(self, frame):
        """Draw enhanced snake game with better graphics"""
        # Draw game area with animated border
        border_pulse = int(50 + 50 * abs(np.sin(time.time() * 2)))
        border_color = (0, border_pulse, 0)
        
        for i in range(5):
            cv2.rectangle(frame, 
                         (self.GAME_AREA_X - i, self.GAME_AREA_Y - i), 
                         (self.GAME_AREA_X + self.GAME_AREA_WIDTH + i, 
                          self.GAME_AREA_Y + self.GAME_AREA_HEIGHT + i), 
                         border_color, 1)
        
        # Draw adaptive grid
        grid_width = self.GAME_AREA_WIDTH // self.GRID_SIZE
        grid_height = self.GAME_AREA_HEIGHT // self.GRID_SIZE
        
        for i in range(grid_width + 1):
            x = self.GAME_AREA_X + i * self.GRID_SIZE
            cv2.line(frame, (x, self.GAME_AREA_Y), 
                    (x, self.GAME_AREA_Y + self.GAME_AREA_HEIGHT), (0, 30, 0), 1)
        
        for i in range(grid_height + 1):
            y = self.GAME_AREA_Y + i * self.GRID_SIZE
            cv2.line(frame, (self.GAME_AREA_X, y), 
                    (self.GAME_AREA_X + self.GAME_AREA_WIDTH, y), (0, 30, 0), 1)
        
        # Draw snake with enhanced graphics
        player_color = self.player_colors[self.current_player_index % len(self.player_colors)]
        
        for i, segment in enumerate(self.snake):
            x = self.GAME_AREA_X + segment[0] * self.GRID_SIZE + 2
            y = self.GAME_AREA_Y + segment[1] * self.GRID_SIZE + 2
            size = self.GRID_SIZE - 4
            
            if i == 0:  # Head
                # Pulsing head effect
                head_pulse = int(10 * abs(np.sin(time.time() * 4)))
                cv2.rectangle(frame, (x - head_pulse//2, y - head_pulse//2), 
                             (x + size + head_pulse//2, y + size + head_pulse//2), 
                             player_color, -1)
                
                # Enhanced eyes based on direction
                eye_size = max(2, self.GRID_SIZE // 8)
                if self.direction == Direction.RIGHT:
                    cv2.circle(frame, (x + size - 8, y + size//3), eye_size, self.BLACK, -1)
                    cv2.circle(frame, (x + size - 8, y + 2*size//3), eye_size, self.BLACK, -1)
                elif self.direction == Direction.LEFT:
                    cv2.circle(frame, (x + 8, y + size//3), eye_size, self.BLACK, -1)
                    cv2.circle(frame, (x + 8, y + 2*size//3), eye_size, self.BLACK, -1)
                elif self.direction == Direction.UP:
                    cv2.circle(frame, (x + size//3, y + 8), eye_size, self.BLACK, -1)
                    cv2.circle(frame, (x + 2*size//3, y + 8), eye_size, self.BLACK, -1)
                else:  # DOWN
                    cv2.circle(frame, (x + size//3, y + size - 8), eye_size, self.BLACK, -1)
                    cv2.circle(frame, (x + 2*size//3, y + size - 8), eye_size, self.BLACK, -1)
            else:  # Body
                # Gradient body effect
                intensity = max(0.3, 1.0 - i * 0.05)
                body_color = tuple(int(c * intensity) for c in player_color)
                cv2.rectangle(frame, (x, y), (x + size, y + size), body_color, -1)
                
                # Body segments outline
                cv2.rectangle(frame, (x, y), (x + size, y + size), self.WHITE, 1)
        
        # Enhanced food with particle effect
        if self.food:
            food_x = self.GAME_AREA_X + self.food[0] * self.GRID_SIZE + self.GRID_SIZE//2
            food_y = self.GAME_AREA_Y + self.food[1] * self.GRID_SIZE + self.GRID_SIZE//2
            
            # Pulsing food
            pulse = abs(np.sin(time.time() * 5))
            food_size = int((self.GRID_SIZE//2 - 2) * (0.7 + 0.3 * pulse))
            
            # Multiple food layers for glow effect
            for layer in range(3):
                layer_size = food_size + layer * 3
                layer_intensity = int(255 * (1 - layer * 0.3))
                food_color = (0, layer_intensity, layer_intensity)
                cv2.circle(frame, (food_x, food_y), layer_size, food_color, -1 if layer == 0 else 2)
        
        # Enhanced UI panel
        panel_x = self.GAME_AREA_X + self.GAME_AREA_WIDTH + 30
        panel_y = self.GAME_AREA_Y
        panel_width = min(300, self.WINDOW_WIDTH - panel_x - 20)
        panel_height = 250
        
        # Panel background with gradient
        for i in range(panel_height):
            intensity = int(64 * (1 - i / panel_height))
            cv2.line(frame, (panel_x, panel_y + i), (panel_x + panel_width, panel_y + i), 
                    (intensity, intensity, intensity), 1)
        
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     self.WHITE, 2)
        
        # Player info
        current_player = self.players[self.current_player_index] if self.players else {'name': 'Player'}
        
        font_scale = min(0.8, panel_width / 300)
        
        cv2.putText(frame, "CURRENT PLAYER", (panel_x + 10, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, self.YELLOW, 2)
        cv2.putText(frame, current_player['name'][:15], (panel_x + 10, panel_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, player_color, 2)
        
        # Score with animation
        score_pulse = int(5 * abs(np.sin(time.time() * 3)))
        cv2.putText(frame, f"SCORE: {self.score}", (panel_x + 10 - score_pulse, panel_y + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, self.WHITE, 2)
        
        # Game stats
        if self.game_start_time > 0:
            elapsed = int(time.time() - self.game_start_time)
            cv2.putText(frame, f"TIME: {elapsed}s", (panel_x + 10, panel_y + 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, self.WHITE, 1)
        
        cv2.putText(frame, f"LENGTH: {len(self.snake)}", (panel_x + 10, panel_y + 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, self.WHITE, 1)
        
        cv2.putText(frame, f"SPEED: {max(1, 10 - self.move_delay//20)}/10", 
                   (panel_x + 10, panel_y + 180), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, self.WHITE, 1)
        
        # Enhanced controls info
        controls_y = panel_y + 220
        cv2.putText(frame, "CONTROLS:", (panel_x, controls_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, self.CYAN, 1)
        
        control_texts = [
            "• Move palm = Control snake",
            "• Close fist = Pause/Resume", 
            "• F = Toggle fullscreen",
            "• Q = Quit game"
        ]
        
        for i, text in enumerate(control_texts):
            cv2.putText(frame, text[:25], (panel_x, controls_y + 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.35, self.WHITE, 1)
        
        # Enhanced status indicators
        status_y = 50
        
        # Multiplayer progress
        if len(self.players) > 1:
            progress_text = f"Player {self.current_player_index + 1}/{len(self.players)}"
            cv2.putText(frame, progress_text, (self.WINDOW_WIDTH - 200, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.YELLOW, 2)
        
        # Pause indicator with animation
        if self.paused:
            pause_alpha = abs(np.sin(time.time() * 3))
            pause_color = tuple(int(c * pause_alpha) for c in self.YELLOW)
            pause_text = "⏸️ GAME PAUSED - Open hand to resume ⏸️"
            text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (self.WINDOW_WIDTH - text_size[0]) // 2
            cv2.putText(frame, pause_text, (text_x, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, pause_color, 2)
        
        # Game over with enhanced effects
        if not self.game_running:
            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.WINDOW_WIDTH, self.WINDOW_HEIGHT), 
                         self.BLACK, -1)
            cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
            
            game_over_text = "💀 GAME OVER! 💀"
            final_score_text = f"Final Score: {self.score}"
            
            pulse = abs(np.sin(time.time() * 4))
            text_color = tuple(int(255 * (0.5 + 0.5 * pulse)) if i == 0 else 0 for i in range(3))
            
            text_size = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (self.WINDOW_WIDTH - text_size[0]) // 2
            cv2.putText(frame, game_over_text, (text_x, self.WINDOW_HEIGHT // 2 - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3)
            
            text_size2 = cv2.getTextSize(final_score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x2 = (self.WINDOW_WIDTH - text_size2[0]) // 2
            cv2.putText(frame, final_score_text, (text_x2, self.WINDOW_HEIGHT // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.WHITE, 2)
            
            next_text = "Press ENTER for next player" if self.current_player_index < len(self.players) - 1 else "Press ENTER to see final results"
            text_size3 = cv2.getTextSize(next_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x3 = (self.WINDOW_WIDTH - text_size3[0]) // 2
            cv2.putText(frame, next_text, (text_x3, self.WINDOW_HEIGHT // 2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.GREEN, 2)
    
    def draw_menu(self, frame):
        """Draw animated main menu with fullscreen support"""
        self.menu_animation += 1
        
        title = "🐍 GESTURE CONTROLLED SNAKE GAME 🎮"
        pulse = int(100 * abs(np.sin(self.menu_animation * 0.03)))
        title_color = (50 + pulse//2, 255, 50 + pulse//2)
        
        title_scale = min(1.5, self.WINDOW_WIDTH / 800)
        title_x = (self.WINDOW_WIDTH - len(title) * int(20 * title_scale)) // 2
        cv2.putText(frame, title, (max(50, title_x), 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_color, 3)
        
        menu_items = [
            "🎮 SELECT NUMBER OF PLAYERS:",
            "",
            "1️⃣  Press 1 for Single Player",
            "2️⃣  Press 2 for Two Players", 
            "3️⃣  Press 3 for Three Players",
            "4️⃣  Press 4 for Four Players",
            "",
            "🖥️  Press F for Fullscreen",
            "❌ Press Q to Quit"
        ]
        
        font_scale = min(0.8, self.WINDOW_WIDTH / 1200)
        y_start = 200
        
        for i, item in enumerate(menu_items):
            if item == "":
                continue
            
            color = self.WHITE
            if "Press" in item and any(c.isdigit() for c in item):
                color = self.YELLOW
                if (self.menu_animation // 60 + i) % 4 == 0:
                    color = self.CYAN
            elif "Press F" in item:
                color = self.PURPLE
            elif "Press Q" in item:
                color = self.RED
            elif "SELECT" in item:
                color = self.NEON_GREEN
            
            text_x = max(50, (self.WINDOW_WIDTH - len(item) * int(12 * font_scale)) // 2)
            cv2.putText(frame, item, (text_x, y_start + i * int(50 * font_scale)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

        info_text = f"Resolution: {self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}"
        if self.fullscreen:
            info_text += " (Fullscreen Mode)"
        cv2.putText(frame, info_text, (20, self.WINDOW_HEIGHT - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.GRAY, 1)
    
    def draw_player_name_input(self, frame):
        """Draw enhanced player name input screen"""
        player_num = len(self.players) + 1
        
        player_color = self.player_colors[(player_num - 1) % len(self.player_colors)]
        title_scale = min(1.2, self.WINDOW_WIDTH / 1000)
        
        title_text = f"🎮 ENTER NAME FOR PLAYER {player_num} 🎮"
        title_x = (self.WINDOW_WIDTH - len(title_text) * int(15 * title_scale)) // 2
        cv2.putText(frame, title_text, (title_x, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, title_scale, player_color, 3)
        
        box_width = min(500, self.WINDOW_WIDTH - 100)
        box_height = 80
        box_x = (self.WINDOW_WIDTH - box_width) // 2
        box_y = 200
        
        border_pulse = int(50 + 50 * abs(np.sin(time.time() * 3)))
        border_color = tuple(int(c * (150 + border_pulse) / 255) for c in player_color)
        
        cv2.rectangle(frame, (box_x - 3, box_y - 3), 
                     (box_x + box_width + 3, box_y + box_height + 3), 
                     border_color, 3)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     self.DARK_GRAY, -1)
        
        display_text = self.current_player_name
        if len(display_text) * 20 > box_width - 40:
            display_text = "..." + display_text[-(box_width//20):]
        
        if (self.blink_counter // 30) % 2 == 0:
            display_text += "█"
        else:
            display_text += " "
        self.blink_counter += 1
        
        text_scale = min(1.0, box_width / 400)
        cv2.putText(frame, display_text, (box_x + 15, box_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, self.WHITE, 2)
        
        instructions = [
            "✏️ Type player name and press ENTER",
            "⌫ Press BACKSPACE to delete characters", 
            "↩️ Press ESC to go back to menu",
            f"📝 Maximum length: 20 characters ({len(self.current_player_name)}/20)"
        ]
        
        inst_y = 320
        for i, instruction in enumerate(instructions):
            color = self.CYAN if i == 0 else self.WHITE
            inst_x = (self.WINDOW_WIDTH - len(instruction) * 10) // 2
            cv2.putText(frame, instruction, (inst_x, inst_y + i * 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if self.players:
            cv2.putText(frame, "✅ PLAYERS REGISTERED:", (50, 500), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.GREEN, 2)
            
            for i, player in enumerate(self.players):
                p_color = self.player_colors[i % len(self.player_colors)]
                cv2.putText(frame, f"{i+1}. {player['name']}", (70, 535 + i * 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, p_color, 2)
    
    def draw_game_instructions(self, frame):
        """Draw enhanced game instructions"""
        current_player = self.players[self.current_player_index]
        player_color = self.player_colors[self.current_player_index % len(self.player_colors)]
        
        
        title_pulse = int(30 * abs(np.sin(time.time() * 2)))
        enhanced_color = tuple(min(255, c + title_pulse) for c in player_color)
        
        title_text = f"🚀 GET READY, {current_player['name'].upper()}! 🚀"
        title_scale = min(1.5, self.WINDOW_WIDTH / 700)
        title_x = (self.WINDOW_WIDTH - len(title_text) * int(12 * title_scale)) // 2
        cv2.putText(frame, title_text, (title_x, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, title_scale, enhanced_color, 3)
        
        
        instructions = [
            ("🎯", "HOW TO PLAY:", self.YELLOW, 1.0),
            ("", "", self.WHITE, 0.6),
            ("✋", "Show your OPEN PALM to the camera", self.CYAN, 0.7),
            ("↗️", "Move your palm to control snake direction", self.CYAN, 0.7),
            ("✊", "Close your FIST to pause/resume the game", self.ORANGE, 0.7), 
            ("🟡", "Eat the food (yellow circles) to grow", self.YELLOW, 0.7),
            ("💀", "Avoid hitting walls or your own body", self.RED, 0.7),
            ("🏆", "Try to get the highest score possible!", self.GREEN, 0.7),
            ("", "", self.WHITE, 0.6),
            ("⚡", "Game will start after hand calibration", self.PURPLE, 0.8),
            ("🎮", "Press ENTER when ready to begin!", self.GREEN, 0.9)
        ]
        
        y_start = 180
        font_scale = min(0.8, self.WINDOW_WIDTH / 1200)
        
        for i, (icon, instruction, color, scale) in enumerate(instructions):
            if instruction == "":
                continue
            
            
            if "Press ENTER" in instruction:
                if (self.blink_counter // 45) % 2 == 0:
                    color = self.NEON_GREEN
            elif "HOW TO PLAY" in instruction:
                pulse = int(50 * abs(np.sin(time.time() * 2)))
                color = (color[0], min(255, color[1] + pulse), color[2])
            
            full_text = f"{icon} {instruction}" if icon else instruction
            text_x = max(50, (self.WINDOW_WIDTH - len(full_text) * int(10 * font_scale * scale)) // 2)
            cv2.putText(frame, full_text, (text_x, y_start + i * int(40 * font_scale)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * scale, color, 2)
        
        self.blink_counter += 1
    
    def draw_player_transition(self, frame):
        """Draw enhanced transition screen between players"""
        if self.current_player_index < len(self.players):
            
            if self.current_player_index > 0:
                prev_player = self.players[self.current_player_index - 1]
                prev_color = self.player_colors[(self.current_player_index - 1) % len(self.player_colors)]
                
                result_text = f"🏁 {prev_player['name']} finished with {prev_player['score']} points! 🏁"
                text_scale = min(1.2, self.WINDOW_WIDTH / 600)
                text_x = (self.WINDOW_WIDTH - len(result_text) * int(12 * text_scale)) // 2
                cv2.putText(frame, result_text, (text_x, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, text_scale, prev_color, 3)
            
            
            next_player = self.players[self.current_player_index]
            next_color = self.player_colors[self.current_player_index % len(self.player_colors)]
            
            
            pulse = int(50 * abs(np.sin(time.time() * 3)))
            animated_color = tuple(min(255, c + pulse) for c in next_color)
            
            next_text = f"⏭️ Next up: {next_player['name']} ⏭️"
            next_scale = min(1.5, self.WINDOW_WIDTH / 500)
            next_x = (self.WINDOW_WIDTH - len(next_text) * int(12 * next_scale)) // 2
            cv2.putText(frame, next_text, (next_x, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, next_scale, animated_color, 3)
            
            
            if len(self.players) > 1:
                cv2.putText(frame, "📊 CURRENT STANDINGS:", 
                           ((self.WINDOW_WIDTH - 250) // 2, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.YELLOW, 2)
                

                completed_players = [(i, p) for i, p in enumerate(self.players[:self.current_player_index]) if 'score' in p]
                completed_players.sort(key=lambda x: x[1]['score'], reverse=True)
                
                for rank, (idx, player) in enumerate(completed_players[:3]):
                    rank_color = self.player_colors[idx % len(self.player_colors)]
                    rank_text = f"{rank + 1}. {player['name']}: {player['score']} pts"
                    cv2.putText(frame, rank_text, 
                               ((self.WINDOW_WIDTH - 300) // 2, 440 + rank * 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, rank_color, 2)
            
            
            ready_text = "🎯 Press ENTER when ready to start! 🎯"
            if (self.blink_counter // 40) % 2 == 0:
                ready_color = self.GREEN
            else:
                ready_color = self.NEON_GREEN
            
            ready_x = (self.WINDOW_WIDTH - len(ready_text) * 12) // 2
            cv2.putText(frame, ready_text, (ready_x, 550), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, ready_color, 2)
        
        self.blink_counter += 1
    
    def draw_final_results(self, frame):
        """Draw enhanced final results with celebration effects"""
        
        title_pulse = int(100 * abs(np.sin(time.time() * 2)))
        title_color = (50 + title_pulse//2, 255, 50 + title_pulse//2)
        
        title_text = "🏆 CHAMPIONSHIP RESULTS 🏆"
        title_scale = min(1.8, self.WINDOW_WIDTH / 600)
        title_x = (self.WINDOW_WIDTH - len(title_text) * int(12 * title_scale)) // 2
        cv2.putText(frame, title_text, (title_x, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_color, 4)
        
        
        sorted_players = sorted(enumerate(self.players), key=lambda x: x[1]['score'], reverse=True)
        
        
        if sorted_players:
            winner_idx, winner = sorted_players[0]
            winner_color = self.player_colors[winner_idx % len(self.player_colors)]
            

            winner_pulse = int(80 * abs(np.sin(time.time() * 4)))
            enhanced_winner_color = tuple(min(255, c + winner_pulse) for c in winner_color)
            
            winner_text = f"👑 CHAMPION: {winner['name'].upper()} 👑"
            winner_scale = min(1.5, self.WINDOW_WIDTH / 500)
            winner_x = (self.WINDOW_WIDTH - len(winner_text) * int(12 * winner_scale)) // 2
            cv2.putText(frame, winner_text, (winner_x, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, winner_scale, enhanced_winner_color, 3)
        
        
        cv2.putText(frame, "🥇 FINAL STANDINGS:", 
                   ((self.WINDOW_WIDTH - 250) // 2, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.YELLOW, 2)
        
        medals = ["🥇", "🥈", "🥉", "4️⃣"]
        y_start = 300
        rank_scale = min(1.0, self.WINDOW_WIDTH / 1000)
        
        for rank, (player_idx, player) in enumerate(sorted_players):
            medal = medals[rank] if rank < len(medals) else f"{rank+1}️⃣"
            player_color = self.player_colors[player_idx % len(self.player_colors)]

            if rank == 0:
                border_pulse = int(20 * abs(np.sin(time.time() * 3)))
                cv2.rectangle(frame, 
                             (80, y_start + rank * 60 - 15 - border_pulse//2), 
                             (self.WINDOW_WIDTH - 80, y_start + rank * 60 + 35 + border_pulse//2), 
                             winner_color, 3)
                
                cv2.rectangle(frame,
                             (85, y_start + rank * 60 - 10),
                             (self.WINDOW_WIDTH - 85, y_start + rank * 60 + 30),
                             (0, 50, 100), -1)
            
            rank_text = f"{medal} {player['name']}: {player['score']} points"
            text_x = max(100, (self.WINDOW_WIDTH - len(rank_text) * int(12 * rank_scale)) // 2)
            cv2.putText(frame, rank_text, (text_x, y_start + rank * 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, rank_scale, player_color, 2)
        
        instruction_y = min(600, y_start + len(sorted_players) * 60 + 80)
        
        instructions = [
            "🔄 Press ENTER to play again",
            "❌ Press Q to quit game"
        ]
        
        for i, instruction in enumerate(instructions):
            color = self.GREEN if "ENTER" in instruction else self.RED
            if "ENTER" in instruction and (self.blink_counter // 50) % 2 == 0:
                color = self.NEON_GREEN
            
            inst_x = (self.WINDOW_WIDTH - len(instruction) * 12) // 2
            cv2.putText(frame, instruction, (inst_x, instruction_y + i * 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        self.blink_counter += 1
    
    def handle_keyboard_input(self, key):
        """Enhanced keyboard input handling with fullscreen support"""
        if key == ord('q') or key == ord('Q'):
            return False
        
        if key == ord('f') or key == ord('F'):
            self.toggle_fullscreen()
            return True
        
        if self.state == GameState.MENU:
            if key >= ord('1') and key <= ord('4'):
                self.max_players = key - ord('0')
                if self.max_players == 1:
                    
                    self.players = [{'name': 'Player 1', 'score': 0}]
                    self.current_player_index = 0
                    self.state = GameState.CALIBRATION
                    self.gesture_tracker = AdvancedGestureTracker()  
                else:
                    
                    self.players = []
                    self.current_player_name = ""
                    self.state = GameState.PLAYER_NAME_INPUT
        
        elif self.state == GameState.PLAYER_NAME_INPUT:
            if key == 13:  
                if self.current_player_name.strip():
                    self.players.append({
                        'name': self.current_player_name.strip(), 
                        'score': 0
                    })
                    self.current_player_name = ""
                    
                    if len(self.players) >= self.max_players:
                        self.current_player_index = 0
                        self.state = GameState.CALIBRATION
                        self.gesture_tracker = AdvancedGestureTracker()  
            elif key == 27:  
                self.state = GameState.MENU
                self.players = []
                self.current_player_name = ""
            elif key == 8:  
                self.current_player_name = self.current_player_name[:-1]
            elif 32 <= key <= 126:  
                if len(self.current_player_name) < 20:
                    self.current_player_name += chr(key)
        
        elif self.state == GameState.CALIBRATION:
            if key == 27:  
                self.state = GameState.GAME_INSTRUCTIONS
        
        elif self.state == GameState.GAME_INSTRUCTIONS:
            if key == 13:  
                self.state = GameState.PLAYING
                self.init_snake()
        
        elif self.state == GameState.PLAYING:
            if key == 13:  
                if not self.game_running:
                    self.players[self.current_player_index]['score'] = self.score

                    self.current_player_index += 1
                    if self.current_player_index >= len(self.players):
                        self.state = GameState.FINAL_RESULTS
                    else:
                        self.state = GameState.PLAYER_TRANSITION
            elif key == 32:  
                if self.game_running:
                    self.paused = not self.paused
            elif key == 27:  
                self.state = GameState.MENU
                self.players = []
                self.current_player_index = 0
        
        elif self.state == GameState.PLAYER_TRANSITION:
            if key == 13:
                self.state = GameState.CALIBRATION
                self.gesture_tracker = AdvancedGestureTracker() 
        
        elif self.state == GameState.FINAL_RESULTS:
            if key == 13:  
                self.state = GameState.MENU
                self.players = []
                self.current_player_index = 0
                self.current_player_name = ""
                self.max_players = 1
        
        return True
    
    def cleanup(self):
        """Enhanced cleanup with proper resource management"""
        print("🔄 Cleaning up resources...")
        self.running = False
        
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        try:
            cv2.waitKey(1)  
        except:
            pass
        
        print("✅ Cleanup complete!")
    
    def run(self):
        """Enhanced main game loop with better performance and error handling"""
        print("=" * 80)
        print("🐍 ENHANCED GESTURE CONTROLLED SNAKE GAME 🎮")
        print("=" * 80)
        print("🚀 Features:")
        print("   • Advanced hand tracking with MediaPipe")
        print("   • Fullscreen support (Press F)")
        print("   • Real-time gesture calibration")  
        print("   • Enhanced multiplayer (1-4 players)")
        print("   • Smooth gesture controls with confidence scoring")
        print("   • Professional UI with animations")
        print("   • Adaptive difficulty and responsive design")
        print("=" * 80)
        
        # Initialize performance counters
        frame_count = 0
        start_time = time.time()
        
        try:
            # Wait for camera to initialize
            print("📷 Initializing camera...")
            time.sleep(2)
            
            if not self.cap.isOpened():
                print("❌ Error: Cannot access camera!")
                return
                
            print("✅ Camera ready!")
            print("🎯 Controls:")
            print("   • F = Toggle Fullscreen")
            print("   • Q = Quit Game")
            print("   • ESC = Back/Menu")
            print("   • Enter = Confirm/Next")
            print("=" * 80)
            print("🎮 Game Starting...")
            
            while self.running:
                frame_start = time.time()
                
                # Create adaptive frame
                frame = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
                
                # Draw enhanced background
                self.draw_enhanced_ui(frame)
                
                # Draw camera feed with advanced tracking
                self.draw_enhanced_camera_feed(frame)
                
                # Draw based on game state
                if self.state == GameState.MENU:
                    self.draw_menu(frame)
                elif self.state == GameState.PLAYER_NAME_INPUT:
                    self.draw_player_name_input(frame)
                elif self.state == GameState.CALIBRATION:
                    self.draw_calibration_screen(frame)
                elif self.state == GameState.GAME_INSTRUCTIONS:
                    self.draw_game_instructions(frame)
                elif self.state == GameState.PLAYING:
                    self.update_snake()
                    self.draw_snake_game(frame)
                elif self.state == GameState.PLAYER_TRANSITION:
                    self.draw_player_transition(frame)
                elif self.state == GameState.FINAL_RESULTS:
                    self.draw_final_results(frame)
                
                # Draw performance info
                self.draw_fps_counter(frame)
                
                # Fullscreen indicator
                if self.fullscreen:
                    cv2.putText(frame, "FULLSCREEN MODE - Press F to exit", 
                               (self.WINDOW_WIDTH - 350, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.YELLOW, 1)
                
                # Display frame
                cv2.imshow(self.window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break
                
                # Performance management
                frame_count += 1
                frame_time = time.time() - frame_start
                
                # Target 60 FPS
                target_frame_time = 1.0 / 60.0
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
        
        except KeyboardInterrupt:
            print("\n⚠️ Game interrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Performance stats
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"\n📊 Performance Stats:")
            print(f"   • Average FPS: {avg_fps:.1f}")
            print(f"   • Total Frames: {frame_count}")
            print(f"   • Runtime: {total_time:.1f}s")
            
            self.cleanup()

def main():
    """Enhanced main function with comprehensive error handling"""
    print("🎮 INITIALIZING ENHANCED SNAKE GAME...")
    
    try:
        # Check camera availability
        print("🔍 Checking camera...")
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            print("❌ ERROR: Cannot access camera!")
            print("   📝 Solutions:")
            print("      • Check if camera is connected")
            print("      • Close other applications using camera")
            print("      • Try running as administrator")
            print("      • Check camera permissions")
            return False
        test_cap.release()
        print("✅ Camera available!")
        
        # Check OpenCV version
        cv_version = cv2.__version__
        print(f"📋 OpenCV Version: {cv_version}")
        
        # Check screen resolution
        import tkinter as tk
        root = tk.Tk()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        print(f"🖥️ Screen Resolution: {screen_w}x{screen_h}")
        
        # Initialize and run game
        print("🚀 Starting Enhanced Snake Game...")
        print("   ✨ New Features:")
        print("      • Fullscreen Mode (Press F)")
        print("      • Advanced Gesture Tracking")
        print("      • Hand Calibration System")
        print("      • Enhanced Visual Effects")
        print("      • Improved Performance")
        print("      • Professional UI Design")
        
        time.sleep(1)
        
        game = EnhancedSnakeGame()
        game.run()
        
        return True
        
    except ImportError as e:
        print(f"❌ MISSING LIBRARY: {e}")
        print("📦 Please install required packages:")
        print("   pip install opencv-python mediapipe numpy tkinter")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n" + "=" * 80)
        print("🎮 THANK YOU FOR PLAYING ENHANCED SNAKE GAME! 🐍")
        print("   💝 Hope you enjoyed the enhanced experience!")
        print("   🌟 Rate us and share with friends!")
        print("=" * 80)

if __name__ == "__main__":
    success = main()
    if not success:
        input("\nPress Enter to exit...")
    
    # Final cleanup
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except:
        pass