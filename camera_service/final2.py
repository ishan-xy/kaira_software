# Photo Booth Application - ZMQ Socket + UI Optimized
# Author:  You
# Date:    2025-11-08
# Changes:
#  – ✅ Auto redirect to intro screen after QR code
#  – ✅ Shifted all utility buttons down (more vertical padding)
#  – ✅ SIGNATURE POSITIONED AT BLUE CIRCLE LOCATION (detected dynamically)


import os
import cv2
import numpy as np
import pygame
from pygame import gfxdraw
from PIL import Image, ImageOps, ImageDraw, ImageFont
import pandas as pd
import qrcode
import threading
import time
import zmq
import json


# === Constants ===
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
FULLSCREEN_MODE = False

SIGNATURE_RECT_WIDTH = 900
SIGNATURE_RECT_HEIGHT = 250
BRUSH_SIZE = 8
DRAW_COLOUR = (0, 0, 0)

CASUAL_FONT_CANDIDATES = (
    "Comic Sans MS",
    "ComicSansMS",
    "Comic Sans",
    "ComicSans",
)

TYPED_SIGNATURE_TEXT = ""
TYPED_SIGNATURE_RELATIVE_HEIGHT = 0.55
TYPED_SIGNATURE_COLOUR = (255, 255, 255)

FRAME_IMAGE_PATH = "sat robo-semi final.png"
BACKGROUND_IMAGE_PATH = "back.jpg"
COUNTDOWN_FOLDER = "countdown_images"
EXCEL_URL_SHEET = "Satpic.xlsx"

# Socket configuration
SOCKET_URL = "tcp://127.0.0.1:5555"
SOCKET_TOPIC = b"camera_0"

# Global variables
latest_frame = None
frame_lock = threading.Lock()
running = True
SOCKET_READY = False


# ---------- Pygame Enhanced Button ----------
def draw_button(screen, rect, text, font, is_hover, is_disabled=False, is_pressed=False):
    start_color = (43, 90, 230) if not is_disabled else (100, 100, 100)
    end_color = (66, 135, 245) if not is_disabled else (140, 140, 140)

    shadow_rect = rect.copy()
    shadow_rect.y += 4
    shadow_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, (0, 0, 0, 80), shadow_surf.get_rect(), border_radius=15)
    screen.blit(shadow_surf, shadow_rect)

    button_rect = rect.copy()
    if is_pressed and not is_disabled:
        button_rect.move_ip(0, 2)

    # Gradient fill
    gradient_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
    for y in range(rect.height):
        ratio = y / rect.height
        r = int(end_color[0] * (1 - ratio) + start_color[0] * ratio)
        g = int(end_color[1] * (1 - ratio) + start_color[1] * ratio)
        b = int(end_color[2] * (1 - ratio) + start_color[2] * ratio)
        pygame.draw.line(gradient_surf, (r, g, b), (0, y), (rect.width, y))
    screen.blit(gradient_surf, button_rect)

    outline_color = (0, 180, 120) if not is_disabled else (50, 50, 50)
    pygame.draw.rect(screen, outline_color, button_rect, 3, border_radius=15)

    text_color = (230, 230, 230) if not is_disabled else (130, 130, 130)
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=button_rect.center)
    screen.blit(text_surf, text_rect)


# ---------- Socket Frame Receiver Thread ----------
def zmq_frame_receiver():
    """Background thread to receive frames from ZMQ socket (non-blocking)"""
    global latest_frame, frame_lock, running, SOCKET_READY
    
    context = zmq.Context()
    frame_socket = context.socket(zmq.SUB)
    
    try:
        print(f"\n[SOCKET] Connecting to {SOCKET_URL}...")
        frame_socket.connect(SOCKET_URL)
        frame_socket.subscribe(SOCKET_TOPIC)
        print(f"[SOCKET] ✓ Subscribed to {SOCKET_TOPIC.decode()} on {SOCKET_URL}")
        SOCKET_READY = True
        
        poller = zmq.Poller()
        poller.register(frame_socket, zmq.POLLIN)
        
        frame_count = 0
        while running:
            events = dict(poller.poll(100))  # timeout 100ms
            if frame_socket in events:
                try:
                    topic, meta_json, img_bytes = frame_socket.recv_multipart(flags=zmq.NOBLOCK)
                    meta = json.loads(meta_json.decode())
                    frame = np.frombuffer(img_bytes, dtype=meta['dtype']).reshape(meta['shape']).copy()
                    
                    with frame_lock:
                        latest_frame = frame
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"[SOCKET] ✓ Received {frame_count} frames (shape={frame.shape})")
                        
                except Exception as e:
                    print(f"[SOCKET] ✗ Receive error: {e}")
                    
    except Exception as e:
        print(f"[SOCKET] ✗ Connection error: {e}")
        SOCKET_READY = False
    finally:
        frame_socket.close()
        context.term()
        print("[SOCKET] Closed")


def get_latest_frame():
    """Get the latest frame from socket (thread-safe)"""
    global latest_frame, frame_lock
    with frame_lock:
        return latest_frame.copy() if latest_frame is not None else None


# ---------- Utilities ----------
def resize_frame_without_stretch(frame, target_w, target_h):
    h, w = frame.shape[:2]
    aspect = w / h
    target_aspect = target_w / target_h
    if aspect > target_aspect:
        new_w = int(h * target_aspect)
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    else:
        new_h = int(w / target_aspect)
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h, :]
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


# ---------- DETECT BLUE CIRCLE FOR SIGNATURE PLACEMENT ----------
def detect_blue_circle_location(frame_img: Image.Image):
    """
    Detect the blue circle on the frame (signature area indicator).
    Returns the center position (x, y) and radius of the blue circle.
    Returns (None, None, None) if not found.
    """
    rgba = frame_img.convert("RGBA")
    rgb = np.array(rgba)[..., :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    # Detect blue color range (blue circle on frame)
    # Blue in HSV: H: 100-130, S: 100-255, V: 100-255
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("[SIG] No blue circle detected on frame")
        return None, None, None
    
    # Find the largest blue circle
    largest_area = 0
    best_circle = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            # Try to fit a circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 20:  # Minimum radius threshold
                largest_area = area
                best_circle = (int(x), int(y), int(radius))
    
    if best_circle is None:
        print("[SIG] Could not find blue circle")
        return None, None, None
    
    x, y, radius = best_circle
    print(f"[SIG] ✓ Blue circle detected at ({x}, {y}) with radius {radius}")
    return x, y, radius


# ---------- Compose photo ----------
def compose_photo_with_frame(counter, photo_path, frame_path=FRAME_IMAGE_PATH, output_folder="framed_photos"):
    """Compose photo inside frame (signature added later)"""
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{counter}_framed.png")
    
    frame_img = Image.open(frame_path).convert("RGBA")
    photo_img = Image.open(photo_path).convert("RGBA")
    
    alpha = np.array(frame_img)[..., 3]
    y_idx, x_idx = np.where(alpha == 0)
    x1, y1, x2, y2 = np.min(x_idx), np.min(y_idx), np.max(x_idx), np.max(y_idx)
    aperture_w, aperture_h = x2 - x1, y2 - y1
    
    photo_resized = ImageOps.fit(photo_img, (aperture_w, aperture_h), method=Image.LANCZOS, centering=(0.5, 0.5))
    
    canvas = Image.new("RGBA", frame_img.size, (0, 0, 0, 0))
    canvas.paste(photo_resized, (x1, y1))
    canvas = Image.alpha_composite(canvas, frame_img)
    
    canvas.save(out_path)
    print(f"[PHOTO] Framed photo saved: {out_path}")
    return out_path


# ---------- Strip white from signature ----------
def strip_white_pixels(pil_sig: Image.Image):
    rgba = pil_sig.convert("RGBA")
    data = np.array(rgba)
    r, g, b, a = cv2.split(data)
    mask = (r > 200) & (g > 200) & (b > 200)
    a[mask] = 0
    cleaned = cv2.merge([r, g, b, a])
    return Image.fromarray(cleaned, mode="RGBA")


# ---------- Paste signature at blue circle location ----------
def paste_signature_on_frame(counter, framed_photo_path, signature_img, output_folder="signed_photos"):
    """Place signature (white) just right of the photo's top-right corner inside the frame."""
    import numpy as np
    import os
    from PIL import Image

    os.makedirs(output_folder, exist_ok=True)

    framed = Image.open(framed_photo_path).convert("RGBA")

    # Find photo aperture (transparent part in the frame) from existing frame alpha
    frame_alpha = np.array(framed)[..., 3]
    y_idx, x_idx = np.where(frame_alpha == 0)
    if len(x_idx) == 0 or len(y_idx) == 0:
        print("[SIG] Error: Could not detect frame aperture")
        return framed_photo_path

    x1, y1, x2, y2 = np.min(x_idx), np.min(y_idx), np.max(x_idx), np.max(y_idx)
    aperture_w, aperture_h = x2 - x1, y2 - y1

    # Prepare the signature in white
    sig_clean = strip_white_pixels(signature_img)
    sig_array = np.array(sig_clean)
    alpha_thresh = 50
    visible_pixels = sig_array[..., 3] > alpha_thresh
    sig_array[..., 0][visible_pixels] = 255
    sig_array[..., 1][visible_pixels] = 255
    sig_array[..., 2][visible_pixels] = 255
    sig_white = Image.fromarray(sig_array, "RGBA")

    # Resize: signature height about 20% of photo aperture height
    sig_height = int(aperture_h * 0.20)
    sig_aspect = sig_white.width / sig_white.height
    sig_width = int(sig_height * sig_aspect)
    sig_resized = sig_white.resize((sig_width, sig_height), Image.LANCZOS)

    # Calculate position: to the right of top-right aperture corner, with margin
    margin_x = int(sig_width * 0.15)
    margin_y = int(sig_height * 0.05)
    sig_x = x2 + margin_x
    sig_y = y1 + margin_y

    print(f"[SIG] Placing signature to right of photo aperture top-right: ({sig_x}, {sig_y})")
    framed.paste(sig_resized, (sig_x, sig_y), sig_resized)

    out_path = os.path.join(output_folder, f"{counter}_signed.png")
    framed.save(out_path)
    print(f"[SIG] Signed photo saved: {out_path}")
    return out_path



# ---------- Signature Pad with Timeout ----------
def pygame_signature_with_timeout(timeout=4):
    """Signature capture with automatic save on inactivity timeout"""
    pygame.init()
    flags = pygame.FULLSCREEN if FULLSCREEN_MODE else 0
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), flags)
    pygame.display.set_caption("Sign your signature")
    clock = pygame.time.Clock()

    overlay = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    pad_x = (DISPLAY_WIDTH - SIGNATURE_RECT_WIDTH) // 2
    pad_y = (DISPLAY_HEIGHT - SIGNATURE_RECT_HEIGHT) // 2 - 80
    pad_rect = pygame.Rect(pad_x, pad_y, SIGNATURE_RECT_WIDTH, SIGNATURE_RECT_HEIGHT)
    drawing_surface = pygame.Surface((SIGNATURE_RECT_WIDTH, SIGNATURE_RECT_HEIGHT), pygame.SRCALPHA)
    drawing_surface.fill((255, 255, 255))

    btn_w, btn_h = 200, 60
    btn_y = pad_rect.bottom + 40
    save_btn = pygame.Rect(DISPLAY_WIDTH // 2 - btn_w - 20, btn_y, btn_w, btn_h)
    cancel_btn = pygame.Rect(DISPLAY_WIDTH // 2 + 20, btn_y, btn_w, btn_h)
    clear_btn = pygame.Rect(DISPLAY_WIDTH // 2 - btn_w // 2, btn_y + 80, btn_w, btn_h)

    font_btn = pygame.font.Font(None, 44)
    font_title = pygame.font.Font(None, 48)
    drawing = False
    last_pos = None
    last_event_time = time.time()

    while True:
        mouse_pos = pygame.mouse.get_pos()
        save_hover = save_btn.collidepoint(mouse_pos)
        cancel_hover = cancel_btn.collidepoint(mouse_pos)
        clear_hover = clear_btn.collidepoint(mouse_pos)

        events = pygame.event.get()

        # Auto-save if idle timeout
        if not events and (time.time() - last_event_time) > timeout:
            print(f"[SIG] Auto-saving signature after {timeout}s inactivity")
            try:
                out_path = "signature_temp.png"
                pygame.image.save(drawing_surface, out_path)
                return Image.open(out_path)
            except Exception as e:
                print(f"[SIG] Error saving on timeout: {e}")
                return None

        for event in events:
            last_event_time = time.time()
            if event.type == pygame.QUIT:
                try:
                    out_path = "signature_temp.png"
                    pygame.image.save(drawing_surface, out_path)
                    return Image.open(out_path)
                except Exception as e:
                    print(f"[SIG] Error saving at quit: {e}")
                    return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if save_btn.collidepoint(mouse_pos):
                    out_path = "signature_temp.png"
                    pygame.image.save(drawing_surface, out_path)
                    return Image.open(out_path)
                if cancel_btn.collidepoint(mouse_pos):
                    out_path = "signature_temp.png"
                    pygame.image.save(drawing_surface, out_path)
                    return Image.open(out_path)
                if clear_btn.collidepoint(mouse_pos):
                    drawing_surface.fill((255, 255, 255))
                    continue
                if pad_rect.collidepoint(mouse_pos):
                    drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False

        if drawing and pygame.mouse.get_pressed()[0]:
            rel_x = mouse_pos[0] - pad_x
            rel_y = mouse_pos[1] - pad_y
            if 0 <= rel_x < SIGNATURE_RECT_WIDTH and 0 <= rel_y < SIGNATURE_RECT_HEIGHT:
                if last_pos is not None:
                    pygame.draw.line(drawing_surface, DRAW_COLOUR, (last_pos[0] - pad_x, last_pos[1] - pad_y), (rel_x, rel_y), BRUSH_SIZE)
                else:
                    pygame.draw.circle(drawing_surface, DRAW_COLOUR, (rel_x, rel_y), BRUSH_SIZE // 2)
                last_pos = mouse_pos
        else:
            last_pos = None

        screen.fill((30, 30, 30))
        screen.blit(overlay, (0, 0))
        title = font_title.render("Please sign below", True, (255, 255, 255))
        title_rect = title.get_rect(center=(DISPLAY_WIDTH // 2, pad_y - 60))
        screen.blit(title, title_rect)
        pygame.draw.rect(screen, (255, 255, 255), pad_rect, border_radius=10)
        pygame.draw.rect(screen, (200, 200, 200), pad_rect, 3, border_radius=10)
        screen.blit(drawing_surface, (pad_x, pad_y))
        draw_button(screen, save_btn, "SAVE", font_btn, save_hover)
        draw_button(screen, cancel_btn, "CANCEL", font_btn, cancel_hover)
        draw_button(screen, clear_btn, "CLEAR", font_btn, clear_hover)
        pygame.display.flip()
        clock.tick(60)


# ---------- URL and QR Code ----------
def get_url_from_excel(counter, excel_path=EXCEL_URL_SHEET, sheet_name="Sheet1"):
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"[EXCEL] File not found: {excel_path}")
        return None
    if counter <= len(df):
        url = df.iloc[counter - 2, 1]
        if isinstance(url, str) and url.startswith("http"):
            print(f"[EXCEL] Fetched URL for photo {counter}: {url}")
            return url
    print(f"[EXCEL] No valid URL found for photo {counter}")
    return None


def generate_qr_code(url, counter, output_folder="qr_codes"):
    os.makedirs(output_folder, exist_ok=True)
    qr = qrcode.make(url)
    qr_path = os.path.join(output_folder, f"qr_code_{counter}.png")
    qr.save(qr_path)
    print(f"[QR] Generated: {qr_path}")
    return qr_path


# ---------- Helper Functions ----------
def create_gradient_surface(w, h, top, bottom):
    surf = pygame.Surface((w, h))
    for y in range(h):
        ratio = y / h
        r = int(top[0] * (1 - ratio) + bottom[0] * ratio)
        g = int(top[1] * (1 - ratio) + bottom[1] * ratio)
        b = int(top[2] * (1 - ratio) + bottom[2] * ratio)
        pygame.draw.line(surf, (r, g, b), (0, y), (w, y))
    return surf


def preload_countdown_images(folder=COUNTDOWN_FOLDER, scale=0.4):
    images = {}
    for fname in ("3.png", "2.png", "1.png"):
        path = os.path.join(folder, fname)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), cv2.INTER_AREA)
                images[fname] = img
    return images


def blend_countdown_overlay(frame, overlay, x1, x2, y1, y2):
    if y1 < 0 or x1 < 0 or y2 > frame.shape[0] or x2 > frame.shape[1]:
        return frame
    region = frame[y1:y2, x1:x2].astype(float)
    if overlay.shape[2] == 4:
        ov = overlay[..., :3].astype(float)
        mask = overlay[..., 3:] / 255.0
    else:
        ov = overlay.astype(float)
        mask = np.ones((*overlay.shape[:2], 1))
    if region.shape != ov.shape:
        return frame
    blended = (1.0 - mask) * region + mask * ov
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame


def resize_frame_letterbox(frame, target_w, target_h):
    h, w = frame.shape[:2]
    aspect = w / h
    target_aspect = target_w / target_h
    if aspect > target_aspect:
        new_w = target_w
        new_h = int(target_w / aspect)
    else:
        new_h = target_h
        new_w = int(target_h * aspect)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas




# ---------- Home Screen ----------
def show_intro_screen(camera_icon_path, flow_links=None):
    """
    Home screen with 4 icon-PNG buttons (from icons/) and camera button.
    """
    pygame.init()
    flags = pygame.FULLSCREEN if FULLSCREEN_MODE else 0
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), flags)
    pygame.display.set_caption("Photo Booth")
    clock = pygame.time.Clock()

    # PURE BLACK background
    bg_color = (0, 0, 0)
    screen.fill(bg_color)

    # Load PNGs for each utility button from the icons/ folder
    button_pngs = [
        ("Icons/about sat buttton button.png", "Top-Left"),
        ("Icons/about thapar button blue.png", "Top-Right"),
        ("Icons/about the team button blue.png", "Bottom-Left"),
        ("Icons/About Kaira button blue.png", "Bottom-Right"),
    ]
    rect_width = 800
    rect_height = 350
    margin_h = 40
    margin_v = 120

    positions = [
        (margin_h, margin_v),  # Top-Left
        (DISPLAY_WIDTH - rect_width - margin_h, margin_v),  # Top-Right
        (margin_h, DISPLAY_HEIGHT - rect_height - margin_v),  # Bottom-Left
        (DISPLAY_WIDTH - rect_width - margin_h, DISPLAY_HEIGHT - rect_height - margin_v),  # Bottom-Right
    ]

    button_configs = []
    for i, (png_path, label) in enumerate(button_pngs):
        png_img = pygame.image.load(png_path).convert_alpha()
        png_img = pygame.transform.scale(png_img, (rect_width, rect_height))
        rect = png_img.get_rect(topleft=positions[i])
        button_configs.append({
            "image": png_img,
            "rect": rect,
            "label": label
        })

    # Center camera button
    camera_icon = pygame.image.load(camera_icon_path).convert_alpha()
    camera_icon_size = 160
    camera_icon = pygame.transform.smoothscale(camera_icon, (camera_icon_size, camera_icon_size))
    center_x = DISPLAY_WIDTH // 2
    center_y = DISPLAY_HEIGHT // 2
    camera_radius = 120

    if flow_links is None:
        flow_links = {0: 'camera', 1: None, 2: None, 3: None, 4: None}

    while True:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()

        dist_to_center = ((mouse_pos[0] - center_x) ** 2 + (mouse_pos[1] - center_y) ** 2) ** 0.5
        camera_hover = dist_to_center <= camera_radius

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if camera_hover:
                    linked_flow = flow_links.get(0)
                    if linked_flow == 'camera':
                        return 'camera'
                # Utility button PNG clicks:
                for idx, config in enumerate(button_configs):
                    if config["rect"].collidepoint(mouse_pos):
                        print(f"[UI] Clicked utility button: {config['label']}")

        screen.fill(bg_color)

        for config in button_configs:
            screen.blit(config["image"], config["rect"])

        # Draw centered camera button, greyish style:
        pygame.draw.circle(screen, (70, 70, 80), (center_x, center_y), camera_radius+15, 3)    # dark grey outer ring
        pygame.draw.circle(screen, (110, 110, 120), (center_x, center_y), camera_radius+5, 2)  # mid grey accent
        circle_color = (180, 180, 180) if camera_hover else (130, 130, 130)                    # hover=light grey, else mid grey
        pygame.draw.circle(screen, circle_color, (center_x, center_y), camera_radius, 0)       # main filled circle
        pygame.draw.circle(screen, (220, 220, 220), (center_x, center_y), camera_radius, 4)    # subtle pale grey inner border


        camera_icon_rect = camera_icon.get_rect(center=(center_x, center_y))
        screen.blit(camera_icon, camera_icon_rect)

        pygame.display.flip()


# ---------- Photo Booth Ready Screen ----------
def show_photo_booth_screen():
    pygame.init()
    # Try fullscreen first, fallback to windowed if it fails
    try:
        flags = pygame.FULLSCREEN if FULLSCREEN_MODE else 0
        screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), flags)
    except pygame.error as e:
        print(f"[DISPLAY] Fullscreen failed: {e}")
        print("[DISPLAY] Falling back to windowed mode (1280x720)")
        screen = pygame.display.set_mode((1280, 720), 0)  # Windowed mode with safer resolution
    
    pygame.display.set_caption("Photo Booth")
    clock = pygame.time.Clock()
    
    if os.path.exists(BACKGROUND_IMAGE_PATH):
        bg_img = pygame.image.load(BACKGROUND_IMAGE_PATH)
        bg_img = pygame.transform.scale(bg_img, screen.get_size())
        use_img = True
    else:
        bg_grad = create_gradient_surface(DISPLAY_WIDTH, DISPLAY_HEIGHT, (15, 23, 42), (25, 40, 70))
        use_img = False
    
    font_status = pygame.font.Font(None, 48)
    
    start_time = time.time()
    wait_duration = 3  # Wait 3 seconds then auto-start
    
    while True:
        elapsed = time.time() - start_time
        clock.tick(60)
        
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
        
        if use_img:
            screen.blit(bg_img, (0, 0))
        else:
            screen.blit(bg_grad, (0, 0))
        
        if not SOCKET_READY:
            status = font_status.render("Waiting for camera connection...", True, (255, 100, 100))
        else:
            remaining = wait_duration - elapsed
            status = font_status.render(f"Camera ready! Starting in {max(0, int(remaining))}s...", True, (100, 255, 100))
            
            if elapsed > wait_duration:
                return True
        
        screen.blit(status, status.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2)))
        pygame.display.flip()


# ---------- Display signed photo with auto-skip ----------
def display_signed_photo(signed_photo_path: str, duration: int = 4):
    img = cv2.imread(signed_photo_path)
    if img is None:
        print("[DISPLAY] Error: could not load photo")
        return
    
    img = resize_frame_letterbox(img, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.FULLSCREEN if FULLSCREEN_MODE else 0)
    pygame.display.set_caption("Your Photo")
    
    img_surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
    
    start_t = time.time()
    clock = pygame.time.Clock()
    
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
        
        if time.time() - start_t > duration:
            return
        
        screen.blit(img_surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)


# ---------- QR Code Display (AUTO RETURN) ----------
def display_qr_code_with_button(qr_path):
    """Display QR code and AUTO return after 5 seconds"""
    pygame.init()
    qr_img = pygame.image.load(qr_path)
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.FULLSCREEN if FULLSCREEN_MODE else 0)
    pygame.display.set_caption("QR Code")
    
    btn_w, btn_h = 200, 60
    btn_rect = pygame.Rect((DISPLAY_WIDTH - btn_w) // 2, DISPLAY_HEIGHT - 100, btn_w, btn_h)
    font = pygame.font.Font(None, 44)

    if os.path.exists(BACKGROUND_IMAGE_PATH):
        bg_img = pygame.image.load(BACKGROUND_IMAGE_PATH)
        bg_img = pygame.transform.scale(bg_img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    else:
        bg_img = create_gradient_surface(DISPLAY_WIDTH, DISPLAY_HEIGHT, (15, 23, 42), (25, 40, 70))

    qr_w, qr_h = qr_img.get_size()
    qr_x = (DISPLAY_WIDTH - qr_w) // 2
    qr_y = (DISPLAY_HEIGHT - qr_h) // 2 - 50

    clock = pygame.time.Clock()
    
    start_time = time.time()
    auto_return_duration = 5  # Auto return after 5 seconds

    while True:
        elapsed = time.time() - start_time
        hover = btn_rect.collidepoint(pygame.mouse.get_pos())
        
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if btn_rect.collidepoint(pygame.mouse.get_pos()):
                    return
        
        # AUTO RETURN after 5 seconds
        if elapsed > auto_return_duration:
            print("[QR] Auto-returning to intro screen")
            return
        
        screen.blit(bg_img, (0, 0))
        screen.blit(qr_img, (qr_x, qr_y))
        
        # Show remaining time
        remaining = auto_return_duration - elapsed
        time_font = pygame.font.Font(None, 36)
        time_text = time_font.render(f"Returning in {max(0, int(remaining))}s...", True, (200, 200, 200))
        screen.blit(time_text, time_text.get_rect(center=(DISPLAY_WIDTH // 2, DISPLAY_HEIGHT - 200)))
        
        draw_button(screen, btn_rect, "DONE", font, hover)
        pygame.display.flip()
        clock.tick(60)


# ---------- Main Live Capture Workflow ----------
def live_video_capture_and_workflow(frame_path=FRAME_IMAGE_PATH):
    """Auto countdown + capture + signature workflow"""
    photos_dir = "photos"
    os.makedirs(photos_dir, exist_ok=True)
    counter = max((int(f.split(".")[0]) for f in os.listdir(photos_dir) if f.split(".")[0].isdigit()), default=0)

    countdown_imgs = preload_countdown_images()
    
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.FULLSCREEN if FULLSCREEN_MODE else 0)
    pygame.display.set_caption("Photo Booth - Camera")
    clock = pygame.time.Clock()

    while True:
        # Countdown and capture
        for fname in ("3.png", "2.png", "1.png"):
            start = time.time()
            while time.time() - start < 0.8:
                frame = get_latest_frame()
                if frame is None:
                    continue

                frame = resize_frame_without_stretch(frame, DISPLAY_WIDTH, DISPLAY_HEIGHT)
                frame = cv2.flip(frame, 1)
                if fname in countdown_imgs:
                    h, w = frame.shape[:2]
                    ci = countdown_imgs[fname]
                    ch, cw = ci.shape[:2]
                    x1 = (w - cw) // 2
                    y1 = (h - ch) // 2
                    frame = blend_countdown_overlay(frame, ci, x1, x1 + cw, y1, y1 + ch)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                clock.tick(60)

        # Capture photo
        photo = get_latest_frame()
        if photo is None:
            print("[CAPTURE] No frame available")
            continue

        counter += 1
        photo_path = os.path.join(photos_dir, f"{counter}.jpg")
        cv2.imwrite(photo_path, photo)
        print(f"[CAPTURE] Photo {counter} saved")

        # Frame and sign
        framed = compose_photo_with_frame(counter, photo_path, frame_path)
        sig_img = pygame_signature_with_timeout(timeout=4)

        if sig_img is None:
            sig_img = Image.new("RGBA", (SIGNATURE_RECT_WIDTH, SIGNATURE_RECT_HEIGHT), (0, 0, 0, 0))
            print("[SIG] Using blank signature")

        # NEW: Signature placed at blue circle location
        signed_path = paste_signature_on_frame(counter, framed, sig_img)
        display_signed_photo(signed_path, duration=4)

        # QR Code with AUTO RETURN
        url = get_url_from_excel(counter)
        if url:
            qr_path = generate_qr_code(url, counter)
            display_qr_code_with_button(qr_path)
            # After QR code, auto returns here
            print(f"[WORKFLOW] ✓ Photo {counter} complete - returning to intro\n")
            return  # Return to main loop, which will show intro screen


# ---------- Main ----------
def main():
    global running, SOCKET_READY
    
    print("\n" + "="*60)
    print("🎥 PHOTO BOOTH - ZMQ SOCKET VERSION")
    print("="*60)
    print(f"Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print(f"Socket: {SOCKET_URL}")
    print("="*60 + "\n")
    
    pygame.init()
    
    # Start socket receiver
    print("[MAIN] Starting socket receiver thread...")
    receiver_thread = threading.Thread(target=zmq_frame_receiver, daemon=True)
    receiver_thread.start()
    
    time.sleep(2)
    
    camera_icon_path = "camera_icon.png"
    
    while True:
        print("\n[MAIN] Showing intro screen...")
        result = show_intro_screen(camera_icon_path=camera_icon_path)
        
        if result == 'camera':
            print("[MAIN] Starting photo booth...")
            if show_photo_booth_screen():
                print("[MAIN] Starting live workflow...")
                live_video_capture_and_workflow()
                print("[MAIN] Returning to intro screen...")
                # Loop continues and shows intro again
        elif result is False:
            break
    
    # Cleanup
    running = False
    receiver_thread.join(timeout=2)
    pygame.quit()
    print("[MAIN] Application closed")


if __name__ == "__main__":
    main()
