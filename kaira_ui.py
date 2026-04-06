import pygame
import numpy as np
import threading
import time
import math
import cv2
import os
import sys

# Qt WebEngine for embedded browser
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineSettings
    from PyQt6.QtCore import QUrl, QTimer, QSize, Qt
    from PyQt6.QtGui import QImage, QPainter
    QTWEBENGINE_AVAILABLE = True
except ImportError:
    QTWEBENGINE_AVAILABLE = False
    print("⚠️ PyQt6-WebEngine not installed. Website viewing will not work.")

# Skip heavy imports in demo mode
if os.environ.get('KAIRA_DEMO_MODE') != '1':
    from kaira_core import KAIRACore
else:
    KAIRACore = None  # Type hint only

from camera_service import demo as cam_demo

# Qt WebEngine offscreen renderer
if QTWEBENGINE_AVAILABLE:
    class QtWebRenderer:
        """Offscreen Qt WebEngine renderer for embedding in pygame."""

        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.qapp = None
            self.webview = None
            self.current_image = None
            self.is_loaded = False

            # Create QApplication if it doesn't exist
            if QApplication.instance() is None: # type: ignore
                self.qapp = QApplication(sys.argv) # type: ignore
            else:
                self.qapp = QApplication.instance() # type: ignore

            # Create offscreen webview
            self.webview = QWebEngineView() # type: ignore
            self.webview.resize(width, height)

            # Enable features
            settings = self.webview.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True) # type: ignore
            settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True) # type: ignore

            # Connect load finished signal
            self.webview.loadFinished.connect(self._on_load_finished)

            print(f"✅ QtWebRenderer initialized ({width}x{height})")

        def _on_load_finished(self, ok):
            """Called when page finishes loading."""
            self.is_loaded = ok
            if ok:
                print("✅ Webpage loaded successfully")
            else:
                print("❌ Webpage failed to load")

        def load_url(self, url):
            """Load a URL in the webview."""
            self.webview.setUrl(QUrl(url))
            self.is_loaded = False

        def render_to_image(self):
            """Render current webpage to QImage."""
            if not self.webview:
                return None

            # Create image
            image = QImage(self.width, self.height, QImage.Format.Format_RGBA8888)
            image.fill(Qt.GlobalColor.black)

            # Render webview to image
            painter = QPainter(image)
            self.webview.render(painter)
            painter.end()

            return image

        def get_pygame_surface(self):
            """Get current render as pygame surface."""
            # Process Qt events
            if self.qapp:
                self.qapp.processEvents()

            # Render to image
            qimage = self.render_to_image()
            if not qimage:
                return None

            # Convert QImage to bytes
            ptr = qimage.bits()
            ptr.setsize(qimage.sizeInBytes())
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((self.height, self.width, 4))

            # Convert RGBA to pygame surface
            surface = pygame.image.frombuffer(arr.tobytes(), (self.width, self.height), "RGBA")
            return surface

        def send_mouse_click(self, x, y, button=1):
            """Send mouse click event to webview."""
            # Qt will handle this through the widget
            pass

        def send_mouse_move(self, x, y):
            """Send mouse move event to webview."""
            pass

        def cleanup(self):
            """Clean up resources."""
            if self.webview:
                self.webview.close()
                self.webview = None
            print("✅ QtWebRenderer cleaned up")

class KAIRAUI:
    def __init__(self, core):
        self.core = core
        pygame.init()

        # Use fullscreen mode
        try:
            self.screen = pygame.display.set_mode((0, 0), pygame.WINDOWMAXIMIZED)
        except Exception:
            # Fallback to windowed mode if fullscreen fails
            self.screen = pygame.display.set_mode((1280, 720))

        self.demo_process = None  # Track the photo booth process
        
        # Screen setup (if not set above)
        if not hasattr(self, 'screen'):
            self.screen = pygame.display.set_mode((0, 0), pygame.WINDOWMAXIMIZED)
        self.screen_width, self.screen_height = self.screen.get_size()
        pygame.display.set_caption("KAIRA (Waiting for 'hey kaira')")
        
        # --- Fonts ---
        self.caption_font = pygame.font.SysFont("Arial", 36, bold=True)
        # NEW: Font for KAIRA's response
        self.kaira_response_font = pygame.font.SysFont("Arial", 34, italic=True) 

        # --- Colors ---
        self.bg_color = (0, 0, 0)
        self.accent_color = (0, 208, 255)      # Bright blue (listening)
        self.accent_color_dim = (0, 150, 200)  # Dim blue (waiting)
        self.caption_color = (255, 255, 255)   # User's final text
        self.realtime_color = (180, 180, 180)  # User's realtime text
        self.kaira_response_color = (200, 200, 255) # KAIRA's text

        # --- UI State (independent of core state) ---
        self.current_mouth_scale = 1.0
        self.is_blinking = False
        self.blink_scale = 1.0
        self.last_blink_time = time.time()
        self.animation_time = 0.0
        self.mic_particles = []
        self.clock = pygame.time.Clock()
        
        # UI's local copy of core state
        self.caption_display_duration = 3.0 # Fade-out time
        self.current_display_text = ""      # User's text
        self.kaira_response_text = ""       # KAIRA's text
        self.is_final_sentence = False
        self.is_kaira_speaking = False
        self.last_sentence_time = 0
        self.normalized_amplitude = 0.0
        self.listening_state = 'WAITING'

        # --- Add a microphone button (top right corner) ---
        self.mic_button_rect = pygame.Rect(self.screen_width - 200, 20, 180, 60)
        self.mic_button_color = (0, 208, 255)  # Bright blue
        self.mic_button_active_color = (255, 50, 50)  # Red when active
        self.mic_button_font = pygame.font.SysFont("Arial", 18, bold=True)
        self.mic_button_text_talk = "Talk to Kaira"
        self.mic_button_text_stop = "Stop Kaira"


        self.total_content_height = self.screen_height * 2  # 2 pages
        self.scroll_y = 0  # Current scroll position (0 = top, screen_height = bottom page)
        self.max_scroll = self.screen_height  # Can scroll down one full screen
        
        # Scrollbar appearance: make it wider for touch but invisible visually
        # Keep the hit area large so touch pens can reliably hit it
        self.scrollbar_width = 40
        # We intentionally do not draw visible colors to keep it invisible
        self.scrollbar_color = (0, 0, 0, 0)
        self.scrollbar_hover_color = (0, 0, 0, 0)
        self.scrollbar_drag_color = (0, 0, 0, 0)
        self.scrollbar_dragging = False
        self.scrollbar_drag_start_y = 0
        
        # ZMQ frame receiver thread (started lazily)
        self.camera_thread_started = False
        self.camera_receiver_thread = None

        # --- Team Images Viewer State ---
        self.viewing_team_images = False
        self.team_images = []
        self.team_images_loaded = False
        self.team_scroll_y = 0
        self.team_max_scroll = 0
        self.team_scrollbar_dragging = False
        self.team_scrollbar_drag_start_y = 0

        # --- Website Viewer State ---
        self.viewing_website = False
        self.website_url = ""
        self.website_opened = False
        self.web_renderer = None

        # Overlay message when launching external services (e.g., photo booth)
        self.overlay_message = None

        # Tracking service state
        self.tracking_process = None  # Track the tracking service process
        self.jetson_socket = None     # Persistent socket connection to Jetson Nano

        print("KAIRA UI initialized.")   

    def update_animations(self, dt):
        """Update smooth animations"""
        self.animation_time += dt
        scale_speed = 5.0
        
        # Decay visual amplitude
        self.normalized_amplitude *= (1.0 - 4.0 * dt)

        # Get latest state from core
        core_state = self.core.get_state()
        if core_state['normalized_amplitude'] > self.normalized_amplitude:
             self.normalized_amplitude = core_state['normalized_amplitude']
             
        # --- Update UI state from core ---
        self.listening_state = core_state['listening_state']
        self.current_display_text = core_state['display_text']
        self.kaira_response_text = core_state['kaira_response_text']
        self.is_final_sentence = core_state['is_final_sentence']
        self.is_kaira_speaking = core_state['is_kaira_speaking']
        self.last_sentence_time = core_state['last_sentence_time']
        
        if self.listening_state == 'WAITING':
             pygame.display.set_caption("KAIRA (Press Spacebar to Talk)")
        else:
             pygame.display.set_caption("KAIRA (Listening...)")

        # Mouth scale (always reacts to sound)
        target = 1.0 + self.normalized_amplitude * 0.5
        self.current_mouth_scale += (target - self.current_mouth_scale) * scale_speed * dt
        
        # Blinking (Unchanged)
        current_time = time.time()
        if current_time - self.last_blink_time > 3.0 + np.random.rand() * 2.0:
            self.trigger_blink()
            self.last_blink_time = current_time
        if self.is_blinking:
            self.blink_scale = max(0.05, self.blink_scale - 10.0 * dt)
            if self.blink_scale <= 0.05:
                self.blink_scale = 0.05
                threading.Timer(0.15, self.end_blink).start()
        else:
            self.blink_scale = min(1.0, self.blink_scale + 10.0 * dt)
        
        # Mic animation particles (Unchanged)
        if len(self.mic_particles) < 3 and np.random.rand() < 0.1:
            self.mic_particles.append({'radius': 20, 'alpha': 255, 'growth_rate': 60})
        for particle in self.mic_particles[:]:
            particle['radius'] += particle['growth_rate'] * dt
            particle['alpha'] = max(0, particle['alpha'] - 200 * dt)
            if particle['alpha'] <= 0:
                self.mic_particles.remove(particle)
            
        # --- NEW FADE-OUT LOGIC ---
        # This one timer now handles fading for BOTH user text and AI text
        time_since_last_sentence = time.time() - self.last_sentence_time
        
        # Check if we should fade the user's final sentence
        if self.is_final_sentence and not self.is_kaira_speaking and (time_since_last_sentence > self.caption_display_duration):
            # Fade user's text if KAIRA doesn't respond
            self.core.state['display_text'] = "" 
            self.core.state['is_final_sentence'] = False

        # Check if we should fade KAIRA's final response
        if not self.is_kaira_speaking and self.kaira_response_text and (time_since_last_sentence > self.caption_display_duration):
             # Fade KAIRA's text after she finishes
             self.core.state['kaira_response_text'] = ""


    def trigger_blink(self):
        self.is_blinking = True

    def end_blink(self):
        self.is_blinking = False

    def draw_3d_mic_animation_on_surface(self, surface):
        """Draw 3D mic animation on a given surface"""
        mic_x = self.screen_width - 120
        mic_y = self.screen_height - 120
        
        if self.listening_state == 'LISTENING':
            if self.normalized_amplitude > 0.3: 
                base_color = (255, 50, 50) # Red
                pulse = 0.8 + 0.2 * self.normalized_amplitude
            else: 
                base_color = self.accent_color # Bright Blue
                pulse = 0.8 + 0.1 * math.sin(self.animation_time * 4) 
        else: # 'WAITING' state
            base_color = self.accent_color_dim # Dim Blue
            pulse = 0.7
        
        particle_color = self.accent_color if self.listening_state == 'LISTENING' else self.accent_color_dim
        for particle in self.mic_particles:
            alpha = int(particle['alpha'])
            color = (*particle_color, alpha)
            size = int(particle['radius'] * 2 + 10)
            if size > 0:
                surf = pygame.Surface((size, size), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (size // 2, size // 2), int(particle['radius']), 3)
                surface.blit(surf, (mic_x - size // 2, mic_y - size // 2))
        
        mic_size = int(35 * pulse)
        shadow_offset = 4
        pygame.draw.circle(surface, (20, 20, 20), 
                          (mic_x + shadow_offset, mic_y + shadow_offset), mic_size + 5)
        
        for i in range(3, 0, -1):
            alpha = int(100 / i)
            glow_color = (*base_color, alpha)
            glow_surf = pygame.Surface((mic_size * 3, mic_size * 3), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (mic_size * 3 // 2, mic_size * 3 // 2), mic_size + i * 8)
            surface.blit(glow_surf, (mic_x - mic_size * 3 // 2, mic_y - mic_size * 3 // 2))
        
        # ... (rest of mic drawing unchanged) ...
        for i in range(5):
            shade = tuple(max(0, c - i * 20) for c in base_color)
            pygame.draw.circle(surface, shade, (mic_x, mic_y - i), mic_size - i * 2)
        highlight_offset = int(mic_size * 0.3); pygame.draw.circle(surface, (255, 255, 255), (mic_x - highlight_offset, mic_y - highlight_offset), mic_size // 4)
        stem_width, stem_height = 12, 20; stem_rect = pygame.Rect(mic_x - stem_width // 2, mic_y + mic_size - 5, stem_width, stem_height); pygame.draw.rect(surface, base_color, stem_rect, border_radius=6)
        base_width, base_height = 30, 8; base_rect = pygame.Rect(mic_x - base_width // 2, mic_y + mic_size + stem_height - 8, base_width, base_height); pygame.draw.rect(surface, base_color, base_rect, border_radius=4)


    def draw_dashboard_page(self, surface):
        """Draw the KAIRA dashboard (first page)"""
        surface.fill(self.bg_color)
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        scale = min(self.screen_width, self.screen_height) / 100
        
        eye_color = self.accent_color if self.listening_state == 'LISTENING' else self.accent_color_dim
        eye_width, eye_height = int(20 * scale), int(30 * scale * self.blink_scale)
        eye_y = center_y - int(20 * scale)
        
        left_eye_rect = pygame.Rect(center_x - int(30 * scale) - eye_width // 2, eye_y - eye_height // 2, eye_width, eye_height)
        pygame.draw.rect(surface, eye_color, left_eye_rect, border_radius=int(5 * scale))
        right_eye_rect = pygame.Rect(center_x + int(30 * scale) - eye_width // 2, eye_y - eye_height // 2, eye_width, eye_height)
        pygame.draw.rect(surface, eye_color, right_eye_rect, border_radius=int(5 * scale))
        
        mouth_y, mouth_width = center_y + int(30 * scale), int(30 * scale)
        pygame.draw.line(surface, eye_color, (center_x - mouth_width, mouth_y), (center_x + mouth_width, mouth_y), int(5 * scale * self.current_mouth_scale))

        # Draw mic animation on the dashboard surface
        self.draw_3d_mic_animation_on_surface(surface)
        
        # Caption text
        if self.kaira_response_text:
            self.draw_wrapped_text_on_surface(
                surface,
                self.kaira_response_text,
                self.kaira_response_font,
                self.kaira_response_color,
                self.screen_width // 2, 
                self.screen_height - 80,
                max_width=self.screen_width - 100
            )
        elif self.current_display_text:
            color = self.caption_color if self.is_final_sentence else self.realtime_color
            self.draw_wrapped_text_on_surface(
                surface,
                self.current_display_text,
                self.caption_font,
                color,
                self.screen_width // 2, 
                self.screen_height - 80,
                max_width=self.screen_width - 100
            )
        
        # Mic button - changes based on listening state
        is_listening = self.listening_state != 'WAITING'
        button_color = self.mic_button_active_color if is_listening else self.mic_button_color
        button_text = self.mic_button_text_stop if is_listening else self.mic_button_text_talk
        
        pygame.draw.rect(surface, button_color, self.mic_button_rect, border_radius=10)
        text_surf = self.mic_button_font.render(button_text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.mic_button_rect.center)
        surface.blit(text_surf, text_rect)
        
        # Scroll hint
        hint_font = pygame.font.SysFont("Arial", 20)
        hint_text = "⬇️ Scroll DOWN to access Camera Service"
        hint_surf = hint_font.render(hint_text, True, (120, 120, 120))
        hint_rect = hint_surf.get_rect(center=(self.screen_width // 2, self.screen_height - 30))
        bg_rect = hint_rect.inflate(20, 10)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 150))
        surface.blit(bg_surf, bg_rect)
        surface.blit(hint_surf, hint_rect)

    def draw_wrapped_text_on_surface(self, surface, text, font, color, center_x, bottom_y, max_width):
        """Helper function to draw word-wrapped text on a given surface"""
        words = text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        
        line_height = font.get_linesize()
        for i, line in enumerate(reversed(lines)):
            line_surf = font.render(line, True, color)
            line_rect = line_surf.get_rect(
                center=(center_x, bottom_y - i * line_height - line_height // 2)
            )
            surface.blit(line_surf, line_rect)

    # ---------------- Scrollable content helpers ----------------
    def start_camera_receiver_thread(self):
        """Start camera_service's ZMQ frame receiver in background once."""
        if self.camera_thread_started:
            return
        try:
            self.camera_receiver_thread = threading.Thread(target=cam_demo.zmq_frame_receiver, daemon=True)
            self.camera_receiver_thread.start()
            self.camera_thread_started = True
            print("Camera receiver thread started.")
        except Exception as e:
            print(f"Failed to start camera receiver: {e}")

    def draw_camera_page(self, surface):
        """Draw the camera service page (second page, shown when scrolled down)."""
        # Dark background
        surface.fill((10, 15, 25))
        
        # Title
        title_font = pygame.font.SysFont("Arial", 48, bold=True)
        title = title_font.render("📷 CAMERA SERVICE", True, (0, 220, 255))
        surface.blit(title, (self.screen_width // 2 - title.get_width() // 2, 30))
        
        # Load and display 4 button images in 2x2 grid
        button_images = [
            "camera_service/Icons/about sat buttton button.png",
            "camera_service/Icons/about thapar button blue.png",
            "camera_service/Icons/about the team button blue.png",
            "camera_service/Icons/About Kaira button blue.png",
        ]
        
        # Button layout - scaled down to fit in UI, more dispersed, centered vertically
        btn_scale_width = 350
        btn_scale_height = 150
        spacing_h = 120  # Increased horizontal spacing
        spacing_v = 80   # Increased vertical spacing
        
        # Calculate total height and center vertically
        total_height = (btn_scale_height * 2) + spacing_v
        start_y = (self.screen_height - total_height) // 2 + 20  # Center vertically with slight offset
        
        # Calculate positions for 2x2 grid
        total_width = (btn_scale_width * 2) + spacing_h
        start_x = (self.screen_width - total_width) // 2
        
        self.camera_icon_buttons = []
        
        for i, img_path in enumerate(button_images):
            # Calculate position (2x2 grid)
            row = i // 2
            col = i % 2
            x = start_x + col * (btn_scale_width + spacing_h)
            y = start_y + row * (btn_scale_height + spacing_v)
            
            try:
                # Load and scale image
                img = pygame.image.load(img_path).convert_alpha()
                img = pygame.transform.smoothscale(img, (btn_scale_width, btn_scale_height))
                
                # Draw image
                surface.blit(img, (x, y))
                
            except Exception as e:
                # Fallback if image not found
                pygame.draw.rect(surface, (30, 40, 60), pygame.Rect(x, y, btn_scale_width, btn_scale_height), border_radius=10)
                pygame.draw.rect(surface, (0, 180, 255), pygame.Rect(x, y, btn_scale_width, btn_scale_height), 2, border_radius=10)
            
            # Store rect for click detection
            rect = pygame.Rect(x, y, btn_scale_width, btn_scale_height)
            self.camera_icon_buttons.append(rect)
        
        # Center camera button (circular) with camera icon image
        center_x = self.screen_width // 2
        center_y = start_y + (total_height // 2)  # Perfect center between the 4 buttons
        camera_radius = 80
        
        mouse_pos = pygame.mouse.get_pos()
        adjusted_mouse_y = mouse_pos[1] + self.scroll_y - self.screen_height
        dist_to_center = ((mouse_pos[0] - center_x) ** 2 + (adjusted_mouse_y - center_y) ** 2) ** 0.5
        hover = dist_to_center <= camera_radius
        
        # Draw camera button background (white/light gray)
        if hover:
            pygame.draw.circle(surface, (255, 255, 255), (center_x, center_y), camera_radius + 5)
            pygame.draw.circle(surface, (240, 240, 240), (center_x, center_y), camera_radius)
        else:
            pygame.draw.circle(surface, (220, 220, 220), (center_x, center_y), camera_radius)
        
        pygame.draw.circle(surface, (0, 0, 0), (center_x, center_y), camera_radius, 4)
        
        # Load and draw camera icon image
        try:
            camera_icon = pygame.image.load("camera_service/Icons/camera_icon.png").convert_alpha()
            icon_size = int(camera_radius * 1.2)
            camera_icon = pygame.transform.smoothscale(camera_icon, (icon_size, icon_size))
            surface.blit(camera_icon, (center_x - icon_size // 2, center_y - icon_size // 2))
        except:
            # Fallback to emoji if image not found
            cam_font = pygame.font.SysFont("Arial", 42, bold=True)
            cam_text = cam_font.render("📷", True, (0, 0, 0))
            surface.blit(cam_text, (center_x - cam_text.get_width() // 2, center_y - cam_text.get_height() // 2))
        
        # Store button for click detection
        self.camera_launch_btn = pygame.Rect(center_x - camera_radius, center_y - camera_radius, 
                                             camera_radius * 2, camera_radius * 2)
        
        # Yes Command Button (Tracking Service)
        tracking_btn_width = 250
        tracking_btn_height = 60
        tracking_btn_x = self.screen_width // 2 - tracking_btn_width // 2
        tracking_btn_y = self.screen_height - 120

        # Check if tracking is running
        tracking_active = getattr(self, 'tracking_process', None) is not None and \
                         self.tracking_process.poll() is None if hasattr(self.tracking_process, 'poll') else False

        # Adjust mouse position for scroll
        mouse_pos = pygame.mouse.get_pos()
        adjusted_mouse_y = mouse_pos[1] + self.scroll_y - self.screen_height
        tracking_hover = (tracking_btn_x <= mouse_pos[0] <= tracking_btn_x + tracking_btn_width and
                         tracking_btn_y <= adjusted_mouse_y <= tracking_btn_y + tracking_btn_height)

        # Button color based on state
        if tracking_active:
            btn_color = (255, 100, 100) if tracking_hover else (200, 50, 50)  # Red when active
            btn_text = "STOP TRACKING"
        else:
            btn_color = (100, 255, 100) if tracking_hover else (50, 200, 50)  # Green when inactive
            btn_text = "YES COMMAND"

        # Draw button
        self.tracking_btn_rect = pygame.Rect(tracking_btn_x, tracking_btn_y, tracking_btn_width, tracking_btn_height)
        pygame.draw.rect(surface, btn_color, self.tracking_btn_rect, border_radius=10)
        pygame.draw.rect(surface, (255, 255, 255), self.tracking_btn_rect, 3, border_radius=10)

        # Button text
        btn_font = pygame.font.SysFont("Arial", 24, bold=True)
        btn_text_surf = btn_font.render(btn_text, True, (255, 255, 255))
        btn_text_rect = btn_text_surf.get_rect(center=self.tracking_btn_rect.center)
        surface.blit(btn_text_surf, btn_text_rect)

        # Button description
        desc_font = pygame.font.SysFont("Arial", 14, italic=True)
        desc_text = "Start person tracking mode" if not tracking_active else "Tracking active - click to stop"
        desc_surf = desc_font.render(desc_text, True, (150, 150, 150))
        desc_rect = desc_surf.get_rect(center=(tracking_btn_x + tracking_btn_width // 2, tracking_btn_y - 15))
        surface.blit(desc_surf, desc_rect)

        # Footer
        footer_y = self.screen_height - 40
        footer_font = pygame.font.SysFont("Arial", 16, italic=True)
        footer_text = "💡 Scroll up or press UP arrow to return • Click icons above for information"
        footer_surf = footer_font.render(footer_text, True, (100, 120, 140))
        surface.blit(footer_surf, (self.screen_width // 2 - footer_surf.get_width() // 2, footer_y))

    def load_team_images(self):
        """Load team images from the team folder."""
        if self.team_images_loaded:
            return

        # Clear any previously loaded images
        self.team_images = []

        team_folder = os.path.join(os.path.dirname(__file__), "team")
        if not os.path.exists(team_folder):
            print(f"⚠️ Team folder not found: {team_folder}")
            return

        # Load specific numbered images in order (1.jpg, 2.jpg, 3.jpg)
        image_files = []
        for i in range(1, 4):  # 1, 2, 3
            # Try different extensions
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                img_path = os.path.join(team_folder, f"{i}{ext}")
                if os.path.exists(img_path):
                    image_files.append(img_path)
                    break
            else:
                print(f"⚠️ Image {i} not found in team folder")

        # Load images
        total_height = 0
        for img_path in image_files:
            try:
                img = pygame.image.load(img_path).convert_alpha()

                # Scale image to fit screen width while maintaining aspect ratio
                img_width, img_height = img.get_size()
                scale_factor = self.screen_width / img_width
                new_height = int(img_height * scale_factor)
                scaled_img = pygame.transform.smoothscale(img, (self.screen_width, new_height))
                self.team_images.append(scaled_img)
                total_height += new_height
                print(f"✅ Loaded team image: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"❌ Failed to load {img_path}: {e}")

        # Calculate scrollable area
        self.team_max_scroll = max(0, total_height - self.screen_height)
        self.team_images_loaded = True
        print(f"📸 Loaded {len(self.team_images)} team images, total height: {total_height}px")

    def draw_team_images_view(self):
        """Draw the team images in a scrollable view."""
        # Black background
        self.screen.fill((0, 0, 0))

        if not self.team_images:
            # Show message if no images loaded
            font = pygame.font.SysFont("Arial", 32)
            text = font.render("No team images found", True, (255, 255, 255))
            self.screen.blit(text, (self.screen_width // 2 - text.get_width() // 2,
                                   self.screen_height // 2))
            return

        # Draw images stacked vertically with scroll offset
        y_offset = -self.team_scroll_y
        for img in self.team_images:
            img_height = img.get_height()
            # Only draw if visible on screen
            if y_offset + img_height > 0 and y_offset < self.screen_height:
                self.screen.blit(img, (0, y_offset))
            y_offset += img_height

        # Draw scrollbar if needed
        if self.team_max_scroll > 0:
            self.draw_team_scrollbar()

        # Draw back button (top-left corner)
        back_button_x = 20
        back_button_y = 20
        back_button_width = 120
        back_button_height = 50

        # Store button rect for click detection
        self.team_back_button = pygame.Rect(back_button_x, back_button_y, back_button_width, back_button_height)

        # Check if mouse is hovering
        mouse_pos = pygame.mouse.get_pos()
        is_hovering = self.team_back_button.collidepoint(mouse_pos)

        # Draw button background
        button_color = (0, 150, 255) if is_hovering else (0, 100, 200)
        pygame.draw.rect(self.screen, button_color, self.team_back_button, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.team_back_button, 2, border_radius=10)

        # Draw button text
        button_font = pygame.font.SysFont("Arial", 22, bold=True)
        button_text = button_font.render("← Back", True, (255, 255, 255))
        text_x = back_button_x + (back_button_width - button_text.get_width()) // 2
        text_y = back_button_y + (back_button_height - button_text.get_height()) // 2
        self.screen.blit(button_text, (text_x, text_y))

        # Also show ESC hint in top-right
        hint_font = pygame.font.SysFont("Arial", 16, italic=True)
        hint_text = hint_font.render("(or press ESC)", True, (150, 150, 150))
        self.screen.blit(hint_text, (back_button_x + back_button_width + 15, back_button_y + 17))

    def draw_team_scrollbar(self):
        """Draw scrollbar for team images view."""
        scrollbar_x = self.screen_width - 20
        scrollbar_width = 10

        # Calculate scrollbar thumb
        thumb_height = max(30, int((self.screen_height / (self.team_max_scroll + self.screen_height)) * self.screen_height))
        thumb_y = int((self.team_scroll_y / self.team_max_scroll) * (self.screen_height - thumb_height))

        # Draw track
        pygame.draw.rect(self.screen, (50, 50, 50), (scrollbar_x, 0, scrollbar_width, self.screen_height))
        # Draw thumb
        pygame.draw.rect(self.screen, (150, 150, 150), (scrollbar_x, thumb_y, scrollbar_width, thumb_height), border_radius=5)

    def init_web_renderer(self, url):
        """Initialize Qt web renderer for the given URL."""
        if not QTWEBENGINE_AVAILABLE:
            print("❌ Qt WebEngine not available, cannot open browser")
            return False

        try:
            # Create renderer if it doesn't exist
            if not self.web_renderer:
                self.web_renderer = QtWebRenderer(self.screen_width, self.screen_height)

            # Load URL
            self.web_renderer.load_url(url)
            print(f"✅ Loading {url} in embedded browser")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize web renderer: {e}")
            import traceback
            traceback.print_exc()
            return False

    def draw_website_view(self):
        """Draw the embedded website viewer."""
        # Black background
        self.screen.fill((0, 0, 0))

        if not QTWEBENGINE_AVAILABLE or not self.web_renderer:
            # Fallback: show error message
            center_y = self.screen_height // 2
            error_font = pygame.font.SysFont("Arial", 32)
            error_text = error_font.render("Browser not available", True, (255, 100, 100))
            error_rect = error_text.get_rect(center=(self.screen_width // 2, center_y))
            self.screen.blit(error_text, error_rect)
        else:
            # Render webpage to pygame surface
            web_surface = self.web_renderer.get_pygame_surface()
            if web_surface:
                self.screen.blit(web_surface, (0, 0))
            else:
                # Loading message
                center_y = self.screen_height // 2
                loading_font = pygame.font.SysFont("Arial", 36)
                loading_text = loading_font.render("Loading website...", True, (200, 200, 200))
                loading_rect = loading_text.get_rect(center=(self.screen_width // 2, center_y))
                self.screen.blit(loading_text, loading_rect)

        # Draw back button overlay (top-left corner)
        back_button_x = 20
        back_button_y = 20
        back_button_width = 120
        back_button_height = 50

        # Store button rect for click detection
        self.website_back_button = pygame.Rect(back_button_x, back_button_y, back_button_width, back_button_height)

        # Check if mouse is hovering
        mouse_pos = pygame.mouse.get_pos()
        is_hovering = self.website_back_button.collidepoint(mouse_pos)

        # Draw button background
        button_color = (0, 150, 255) if is_hovering else (0, 100, 200)
        pygame.draw.rect(self.screen, button_color, self.website_back_button, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.website_back_button, 2, border_radius=10)

        # Draw button text
        button_font = pygame.font.SysFont("Arial", 22, bold=True)
        button_text = button_font.render("← Back", True, (255, 255, 255))
        text_x = back_button_x + (back_button_width - button_text.get_width()) // 2
        text_y = back_button_y + (back_button_height - button_text.get_height()) // 2
        self.screen.blit(button_text, (text_x, text_y))

        # Also show ESC hint
        hint_font = pygame.font.SysFont("Arial", 16, italic=True)
        hint_text = hint_font.render("(or press ESC)", True, (150, 150, 150))
        self.screen.blit(hint_text, (back_button_x + back_button_width + 15, back_button_y + 17))

    def close_web_renderer(self):
        """Close the web renderer."""
        if self.web_renderer:
            try:
                self.web_renderer.cleanup()
                self.web_renderer = None
                print("✅ Web renderer closed")
            except Exception as e:
                print(f"⚠️ Error closing web renderer: {e}")

    def get_scrollbar_rect(self):
        """Calculate scrollbar rectangle."""
        # Scrollbar on right edge
        scrollbar_x = self.screen_width - self.scrollbar_width
        # Scrollbar thumb height proportional to visible area
        thumb_height = max(30, int((self.screen_height / self.total_content_height) * self.screen_height))
        # Thumb position based on scroll
        thumb_y = int((self.scroll_y / self.max_scroll) * (self.screen_height - thumb_height))
        return pygame.Rect(scrollbar_x, thumb_y, self.scrollbar_width, thumb_height)
    
    def draw_scrollbar(self):
        """Draw the scrollbar."""
        if self.max_scroll <= 0:
            return  # No scrollbar needed if content fits
        # Intentionally do not draw visible scrollbar; keep hit area for touch input
        # This keeps the scrollbar invisible but touch-sensitive (hitbox is returned by get_scrollbar_rect)
        return
    
    def launch_camera_service(self):
        """Launch the camera service photo booth when icon is clicked"""
        import subprocess
        import sys
        import os
        import socket
        # If demo is already running, bring its window to front instead of launching a new one
        try:
            if getattr(self, 'demo_process', None) is not None and self.demo_process.poll() is None:
                print("Photo booth already running - bringing window to front")
                try:
                    self._bring_demo_to_front(self.demo_process.pid)
                except Exception as e:
                    print(f"⚠️  Could not bring demo to front: {e}")
                # briefly show overlay
                self.overlay_message = "Opening SAT-PHOTO-BOOTH..."
                def _clear():
                    time.sleep(0.5)
                    self.overlay_message = None
                threading.Thread(target=_clear, daemon=True).start()
                return
        except Exception:
            pass
        def _launch_thread():
            try:
                self.overlay_message = "Opening SAT-PHOTO-BOOTH..."

                # Wait for ZMQ port 5555 to be ready
                for _ in range(40):  # ~20 seconds max
                    try:
                        with socket.create_connection(("127.0.0.1", 5555), timeout=0.5):
                            break
                    except Exception:
                        time.sleep(0.5)

                # Launch demo
                kaira_dir = os.path.dirname(__file__)
                venv_python = os.path.join(kaira_dir, '.venv', 'Scripts', 'python.exe')
                python_exec = venv_python if os.path.exists(venv_python) else sys.executable
                
                camera_service_dir = os.path.join(kaira_dir, 'camera_service')
                
                # Prefer pythonw.exe to avoid creating a visible console window on Windows
                venv_pythonw = os.path.join(kaira_dir, '.venv', 'Scripts', 'pythonw.exe')
                if os.path.exists(venv_pythonw):
                    python_exec = venv_pythonw

                self.demo_process = subprocess.Popen(
                    [python_exec, 'demo.py'],
                    cwd=camera_service_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0 # Ensure no console
                )
                print("✅ Photo booth launched")

            except Exception as e:
                print(f"❌ Error launching demo: {e}")
            finally:
                # Clear overlay after a short delay
                time.sleep(0.5)
                self.overlay_message = None

        t = threading.Thread(target=_launch_thread, daemon=True)
        t.start()

    def toggle_tracking_service(self):
        """Toggle the tracking service on/off"""
        import subprocess
        import sys
        import os
        from tracking_config import send_yes_handshake

        # Check if tracking is already running
        if getattr(self, 'tracking_process', None) is not None:
            try:
                if self.tracking_process.poll() is None:
                    # Process is running, stop it
                    print("⏹ Stopping tracking service...")
                    self.tracking_process.terminate()
                    self.tracking_process = None

                    # Close Jetson Nano socket connection
                    if self.jetson_socket:
                        try:
                            print("🔌 Closing Jetson Nano connection...")
                            self.jetson_socket.close()
                            self.jetson_socket = None
                            print("✅ Jetson connection closed")
                        except Exception as socket_err:
                            print(f"⚠️ Error closing Jetson socket: {socket_err}")
                            self.jetson_socket = None

                    # Show overlay message
                    self.overlay_message = "Tracking Service STOPPED"
                    def _clear():
                        time.sleep(1.5)
                        self.overlay_message = None
                    threading.Thread(target=_clear, daemon=True).start()
                    return
            except Exception as e:
                print(f"⚠️ Error stopping tracking service: {e}")
                self.tracking_process = None
                # Also close socket on error
                if self.jetson_socket:
                    try:
                        self.jetson_socket.close()
                    except:
                        pass
                    self.jetson_socket = None

        # Launch tracking service with handshake
        def _launch_thread():
            try:
                # Step 1: Send YES handshake
                self.overlay_message = "Sending YES Handshake..."
                print("🤝 Sending YES handshake to Jetson Nano...")

                success, message, self.jetson_socket = send_yes_handshake()

                if not success:
                    # Handshake failed
                    print(f"❌ Handshake failed: {message}")
                    self.overlay_message = f"Handshake Failed: {message}"
                    time.sleep(3)
                    self.overlay_message = None
                    return

                # Handshake succeeded
                print("✅ Handshake successful!")
                self.overlay_message = "Handshake Successful!"
                time.sleep(1)

                # Step 2: Launch tracking service
                self.overlay_message = "Starting TRACKING SERVICE..."

                kaira_dir = os.path.dirname(__file__)
                venv_python = os.path.join(kaira_dir, '.venv', 'Scripts', 'python.exe')
                python_exec = venv_python if os.path.exists(venv_python) else sys.executable

                tracking_script = os.path.join(kaira_dir, 'tracking_service2_fixed.py')

                if not os.path.exists(tracking_script):
                    print(f"❌ Tracking script not found: {tracking_script}")
                    self.overlay_message = "Error: Tracking script not found!"
                    time.sleep(2)
                    self.overlay_message = None
                    return

                # Prefer pythonw.exe to avoid creating a visible console window on Windows
                venv_pythonw = os.path.join(kaira_dir, '.venv', 'Scripts', 'pythonw.exe')
                if os.path.exists(venv_pythonw):
                    python_exec = venv_pythonw

                print(f"🚀 Launching tracking service from: {tracking_script}")
                self.tracking_process = subprocess.Popen(
                    [python_exec, tracking_script],
                    cwd=kaira_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                print("✅ Tracking service launched")

                # Update overlay message
                self.overlay_message = "Tracking Service ACTIVE"
                time.sleep(1.5)
                self.overlay_message = None

            except Exception as e:
                print(f"❌ Error launching tracking service: {e}")
                self.overlay_message = f"Error: {str(e)}"
                time.sleep(2)
                self.overlay_message = None

        t = threading.Thread(target=_launch_thread, daemon=True)
        t.start()

    def _bring_demo_to_front(self, pid: int) -> bool:
        """Bring the window of the given process id to the foreground (Windows only)."""
        try:
            if os.name != 'nt':
                print("Bring-to-front only implemented on Windows")
                return False
            import ctypes
            import ctypes.wintypes as wintypes

            user32 = ctypes.windll.user32

            EnumWindows = user32.EnumWindows
            EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            hwnds = []

            @EnumWindowsProc
            def _enum(hwnd, lParam):
                pid_buf = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_buf))
                if pid_buf.value == pid and user32.IsWindowVisible(hwnd):
                    hwnds.append(hwnd)
                    return False  # stop enumeration once found
                return True

            EnumWindows(_enum, 0)

            if not hwnds:
                print(f"No window found for pid {pid}")
                return False

            hwnd = hwnds[0]
            SW_RESTORE = 9
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
            print(f"Brought demo window (pid={pid}) to front")
            return True
        except Exception as e:
            print(f"⚠️  Error bringing window to front: {e}")
            return False
    
    def handle_mic_button_click(self, event):
        """Handle click events for the microphone button - toggles talk/stop."""
        if event.type == pygame.MOUSEBUTTONDOWN and self.mic_button_rect.collidepoint(event.pos):
            # Check current state and toggle
            if self.listening_state == 'WAITING':
                print("🎤 Talk to Kaira - Starting recording...")
                self.core.enable_conversation_mode()
                self.core.start_recording()
            else:
                print("🛑 Stop Kaira - Stopping recording...")
                self.core.stop_recording()

    def run(self):
        """Main UI application loop (MODIFIED)"""
        
        # --- NEW: Updated print instructions ---
        print("=" * 70)
        print("🤖 KAIRA - AI Assistant with Camera Service")
        print("=" * 70)
        print("\n🎮 CONTROLS:")
        print("  SPACEBAR or 'Talk to Kaira' BUTTON → Start talking")
        print("  SPACEBAR or 'Stop Kaira' BUTTON → Stop talking")
        print("  SCROLL DOWN or DOWN ARROW → Access Camera Service")
        print("  SCROLL UP or UP ARROW → Back to Dashboard")
        print("  Drag SCROLLBAR → Navigate between pages")
        print("  ESC → Exit")
        print("\n📝 NOTE: Start camera services first with:")
        print("  python launch_camera_services.py")
        print("=" * 70 + "\n")
        # --- END NEW ---
        
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            
            # --- Key Event Handling (Wake Word Mode) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Handle website view events separately
                if self.viewing_website:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        # Exit website view
                        self.close_web_renderer()
                        self.viewing_website = False
                        self.website_opened = False
                        print("🔙 Exiting website view")
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        # Check if back button was clicked
                        if hasattr(self, 'website_back_button') and self.website_back_button.collidepoint(event.pos):
                            self.close_web_renderer()
                            self.viewing_website = False
                            self.website_opened = False
                            print("🔙 Back button clicked - exiting website view")
                    continue  # Skip normal event handling

                # Handle team images view events separately
                if self.viewing_team_images:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        # Exit team images view
                        self.viewing_team_images = False
                        self.team_scroll_y = 0
                        print("🔙 Exiting team images view")
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        # Check if back button was clicked
                        if hasattr(self, 'team_back_button') and self.team_back_button.collidepoint(event.pos):
                            self.viewing_team_images = False
                            self.team_scroll_y = 0
                            print("🔙 Back button clicked - exiting team images view")
                    elif event.type == pygame.MOUSEWHEEL:
                        # Scroll team images
                        scroll_amount = event.y * 50
                        self.team_scroll_y = max(0, min(self.team_max_scroll, self.team_scroll_y - scroll_amount))
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            self.team_scroll_y = max(0, self.team_scroll_y - 100)
                        elif event.key == pygame.K_DOWN:
                            self.team_scroll_y = min(self.team_max_scroll, self.team_scroll_y + 100)
                    continue  # Skip normal event handling

                # Normal mode event handling
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        # Enable conversation mode and start recording
                        # This ensures auto-listen works even when manually triggered
                        self.core.enable_conversation_mode()
                        self.core.start_recording()
                    # Keyboard scrolling
                    if event.key == pygame.K_UP:
                        self.scroll_y = max(0, self.scroll_y - 100)
                        print(f"⬆️ UP arrow: scroll_y = {self.scroll_y}")
                    elif event.key == pygame.K_DOWN:
                        self.scroll_y = min(self.max_scroll, self.scroll_y + 100)
                        print(f"⬇️ DOWN arrow: scroll_y = {self.scroll_y}")
                        if self.scroll_y > 0:
                            self.start_camera_receiver_thread()
                
                # Mouse wheel scrolling
                elif event.type == pygame.MOUSEWHEEL:
                    scroll_amount = event.y * 50  # Scroll sensitivity
                    self.scroll_y = max(0, min(self.max_scroll, self.scroll_y - scroll_amount))
                    print(f"�️ Mouse wheel: scroll_y = {self.scroll_y}")
                    if self.scroll_y > 0:
                        self.start_camera_receiver_thread()
                
                # Scrollbar dragging and button clicks
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    scrollbar_rect = self.get_scrollbar_rect()
                    if scrollbar_rect.collidepoint(event.pos):
                        self.scrollbar_dragging = True
                        self.scrollbar_drag_start_y = event.pos[1] - scrollbar_rect.y
                    # Handle mic button click (only on page 1)
                    elif self.scroll_y < self.screen_height / 2 and self.mic_button_rect.collidepoint(event.pos):
                        print("Microphone button clicked!")
                        self.core.enable_conversation_mode()
                        self.core.start_recording()
                    # Handle camera launch button click (only on page 2)
                    elif self.scroll_y > self.screen_height / 2 and hasattr(self, 'camera_launch_btn'):
                        # Adjust click position for scroll
                        adjusted_y = event.pos[1] + self.scroll_y - self.screen_height
                        adjusted_pos = (event.pos[0], adjusted_y)
                        if self.camera_launch_btn.collidepoint(adjusted_pos):
                            print("🚀 Launching Camera Service Photo Booth...")
                            self.launch_camera_service()
                        # Check camera icon buttons (info buttons)
                        elif hasattr(self, 'camera_icon_buttons'):
                            for i, btn_rect in enumerate(self.camera_icon_buttons):
                                if btn_rect.collidepoint(adjusted_pos):
                                    print(f"📋 Info button {i} clicked")
                                    # Button 0 is "About Saturnalia"
                                    if i == 0:
                                        print("🌐 Opening Saturnalia website...")
                                        self.website_url = "https://saturnalia.in/"
                                        if not self.website_opened:
                                            if self.init_web_renderer(self.website_url):
                                                self.website_opened = True
                                                self.viewing_website = True
                                        else:
                                            self.viewing_website = True
                                    # Button 2 is "About the Team"
                                    elif i == 2:
                                        print("👥 Loading team images...")
                                        self.team_images_loaded = False  # Force reload
                                        self.load_team_images()
                                        self.viewing_team_images = True
                                        self.team_scroll_y = 0
                                    # Add handlers for other buttons here if needed
                                    break
                        # Check tracking button click
                        if hasattr(self, 'tracking_btn_rect') and self.tracking_btn_rect.collidepoint(adjusted_pos):
                            print("🎯 Tracking button clicked - toggling service...")
                            self.toggle_tracking_service()

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.scrollbar_dragging = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.scrollbar_dragging:
                        scrollbar_rect = self.get_scrollbar_rect()
                        thumb_height = scrollbar_rect.height
                        new_thumb_y = event.pos[1] - self.scrollbar_drag_start_y
                        # Convert thumb position to scroll position
                        max_thumb_y = self.screen_height - thumb_height
                        self.scroll_y = int((new_thumb_y / max_thumb_y) * self.max_scroll)
                        self.scroll_y = max(0, min(self.max_scroll, self.scroll_y))
                        if self.scroll_y > 0:
                            self.start_camera_receiver_thread()

            # --- END MODIFIED ---

            # If viewing website, draw that instead of normal UI
            if self.viewing_website:
                self.draw_website_view()
                pygame.display.flip()
                continue

            # If viewing team images, draw that instead of normal UI
            if self.viewing_team_images:
                self.draw_team_images_view()
                pygame.display.flip()
                continue

            self.update_animations(dt)

            # Create a large virtual canvas for all content
            canvas = pygame.Surface((self.screen_width, self.total_content_height))

            # Draw page 1: KAIRA dashboard (top half of canvas)
            page1_surf = pygame.Surface((self.screen_width, self.screen_height))
            self.draw_dashboard_page(page1_surf)
            canvas.blit(page1_surf, (0, 0))

            # Draw page 2: Camera service (bottom half of canvas)
            page2_surf = pygame.Surface((self.screen_width, self.screen_height))
            self.draw_camera_page(page2_surf)
            canvas.blit(page2_surf, (0, self.screen_height))
            
            # Blit visible portion of canvas to screen based on scroll position
            visible_rect = pygame.Rect(0, self.scroll_y, self.screen_width, self.screen_height)
            self.screen.blit(canvas, (0, 0), visible_rect)
            
            # Draw scrollbar on top
            self.draw_scrollbar()
            # Draw overlay message if set (e.g., Opening SAT-PHOTO-BOOTH...)
            if self.overlay_message:
                overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 170))
                self.screen.blit(overlay, (0, 0))
                font_overlay = pygame.font.SysFont('Arial', 40, bold=True)
                msg_surf = font_overlay.render(self.overlay_message, True, (255, 255, 255))
                msg_rect = msg_surf.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
                self.screen.blit(msg_surf, msg_rect)
            
            pygame.display.flip()

        self.cleanup()

    def cleanup(self):
        """Clean up UI resources"""
        # Close web renderer if open
        self.close_web_renderer()

        # Close Jetson Nano socket if connected
        if self.jetson_socket:
            try:
                print("🔌 Closing Jetson Nano connection...")
                self.jetson_socket.close()
                self.jetson_socket = None
                print("✅ Jetson connection closed")
            except Exception as e:
                print(f"⚠️ Error closing Jetson socket: {e}")

        try:
            pygame.quit()
        except:
            pass
        print("KAIRA UI shut down.")