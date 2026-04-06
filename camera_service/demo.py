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
# Auto-detect screen size, but cap at 1920x1080
import pygame
pygame.init()
info = pygame.display.Info()
DISPLAY_WIDTH = min(info.current_w, 1920)
DISPLAY_HEIGHT = min(info.current_h, 1080)
# Use fullscreen by default; dialogs fall back if unsupported
FULLSCREEN_MODE = True

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

# Global variables for socket frame management
latest_frame = None
frame_lock = threading.Lock()
running = True
frame_ready = False



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
    """Background thread to receive frames from ZMQ socket using poller (non-blocking)"""
    global latest_frame, frame_lock, running
    
    context = zmq.Context()
    frame_socket = context.socket(zmq.SUB)
    
    try:
        print(f"Connecting to ZMQ socket at {SOCKET_URL}...")
        frame_socket.connect(SOCKET_URL)
        frame_socket.subscribe(SOCKET_TOPIC)
        print(f"Successfully subscribed to {SOCKET_TOPIC.decode()} on {SOCKET_URL}")
        
        poller = zmq.Poller()
        poller.register(frame_socket, zmq.POLLIN)
        
        while running:
            events = dict(poller.poll(100))  # timeout 100ms
            if frame_socket in events:
                try:
                    topic, meta_json, img_bytes = frame_socket.recv_multipart(flags=zmq.NOBLOCK)
                    meta = json.loads(meta_json.decode())
                    frame = np.frombuffer(img_bytes, dtype=meta['dtype']).reshape(meta['shape']).copy()
                    
                    with frame_lock:
                        latest_frame = frame
                except Exception as e:
                    print(f"Receive error: {e}")
                    
    except Exception as e:
        print(f"Socket connection error: {e}")
    finally:
        frame_socket.close()
        context.term()
        print("Socket receiver thread closed.")



def get_latest_frame():
    """Get the latest frame from socket in a thread-safe manner"""
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






def detect_red_banner(frame_img: Image.Image):
    rgba = frame_img.convert("RGBA")
    rgb = np.array(rgba)[..., :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, (0, 150, 100), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 150, 100), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    h, w = mask.shape
    mask[: int(h * 0.4), :] = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best_area = 0
    best_bbox = (0, h - 200, w, h)
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if y + bh >= h - 5 and area > best_area:
            best_area = area
            best_bbox = (x, y, x + bw, y + bh)
    return best_bbox



# ---------- Compose photo ----------
def compose_photo_with_frame(counter, photo_path, frame_path=FRAME_IMAGE_PATH, output_folder="framed_photos", typed_signature=TYPED_SIGNATURE_TEXT):
    """Compose photo inside frame WITHOUT signature first - signature added later"""
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{counter}_framed.png")
    
    frame_img = Image.open(frame_path).convert("RGBA")
    photo_img = Image.open(photo_path).convert("RGBA")
    
    # Get aperture bounds from frame transparency
    alpha = np.array(frame_img)[..., 3]
    y_idx, x_idx = np.where(alpha == 0)
    x1, y1, x2, y2 = np.min(x_idx), np.min(y_idx), np.max(x_idx), np.max(y_idx)
    aperture_w, aperture_h = x2 - x1, y2 - y1
    
    # Resize photo to fit aperture
    photo_resized = ImageOps.fit(photo_img, (aperture_w, aperture_h), method=Image.LANCZOS, centering=(0.5, 0.5))
    
    # Create canvas and paste photo at aperture location
    canvas = Image.new("RGBA", frame_img.size, (0, 0, 0, 0))
    canvas.paste(photo_resized, (x1, y1))
    
    # Composite frame on top
    canvas = Image.alpha_composite(canvas, frame_img)
    
    canvas.save(out_path)
    print(f"Framed photo saved as {out_path}")
    return out_path




# ---------- Strip white from signature ----------
def strip_white_pixels(pil_sig: Image.Image):
    rgba = pil_sig.convert("RGBA")
    data = np.array(rgba)
    r,g,b,a = cv2.split(data)
    mask = (r>200) & (g>200) & (b>200)
    a[mask] = 0
    cleaned = cv2.merge([r,g,b,a])
    return Image.fromarray(cleaned, mode="RGBA")



# ---------- Paste signature ----------
def paste_signature_on_frame(counter, framed_photo_path, signature_img, output_folder="signed_photos"):
    """Paste signature on BOTTOM-RIGHT of the PHOTO portion (inside frame)"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the framed photo
    framed = Image.open(framed_photo_path).convert("RGBA")
    
    # Get frame alpha to find aperture bounds
    frame_alpha = np.array(framed)[..., 3]
    y_idx, x_idx = np.where(frame_alpha == 0)
    
    if len(x_idx) == 0 or len(y_idx) == 0:
        print("Error: Could not detect frame aperture")
        return framed_photo_path
    
    x1, y1, x2, y2 = np.min(x_idx), np.min(y_idx), np.max(x_idx), np.max(y_idx)
    aperture_w, aperture_h = x2 - x1, y2 - y1
    
    print(f"Aperture detected: ({x1}, {y1}) to ({x2}, {y2}), size: {aperture_w}x{aperture_h}")
    
    # Clean signature image (make white strokes instead of black)
    sig_clean = strip_white_pixels(signature_img)
    
    # Convert signature black strokes to white
    sig_array = np.array(sig_clean)
    # Find dark pixels (black strokes)
    dark_mask = (sig_array[..., 0] < 100) & (sig_array[..., 1] < 100) & (sig_array[..., 2] < 100) & (sig_array[..., 3] > 50)
    # Make them white
    sig_array[dark_mask, 0] = 255
    sig_array[dark_mask, 1] = 255
    sig_array[dark_mask, 2] = 255
    sig_white = Image.fromarray(sig_array, "RGBA")
    
    # Resize signature (~15% of aperture height)
    sig_height = int(aperture_h * 0.15)
    sig_aspect = sig_white.width / sig_white.height
    sig_width = int(sig_height * sig_aspect)
    sig_resized = sig_white.resize((sig_width, sig_height), Image.LANCZOS)
    
    print(f"Signature resized to: {sig_width}x{sig_height}")
    
    # Calculate position: bottom-right of aperture with margin
    margin = int(sig_height * 0.5)
    sig_x = x1 + aperture_w - sig_width - margin
    sig_y = y1 + aperture_h - sig_height - margin
    
    print(f"Signature position: ({sig_x}, {sig_y})")
    
    # Paste signature directly on framed image at calculated position
    framed.paste(sig_resized, (sig_x, sig_y), sig_resized)
    
    # Save result
    out_path = os.path.join(output_folder, f"{counter}_signed.png")
    framed.save(out_path)
    print(f"Signed photo saved as {out_path}")
    return out_path


# ---------- Pygame Signature pad ----------
def pygame_signature_rect():
    pygame.init()
    if FULLSCREEN_MODE:
        flags = pygame.FULLSCREEN
    else:
        flags = 0
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), flags)
    pygame.display.set_caption("Sign your signature")
    clock = pygame.time.Clock()

    overlay = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0,0,0,180))
    pad_x = (DISPLAY_WIDTH - SIGNATURE_RECT_WIDTH)//2
    pad_y = (DISPLAY_HEIGHT - SIGNATURE_RECT_HEIGHT)//2 - 80
    pad_rect = pygame.Rect(pad_x, pad_y, SIGNATURE_RECT_WIDTH, SIGNATURE_RECT_HEIGHT)
    drawing_surface = pygame.Surface((SIGNATURE_RECT_WIDTH, SIGNATURE_RECT_HEIGHT), pygame.SRCALPHA)
    drawing_surface.fill((255,255,255))

    btn_w, btn_h = 200, 60
    btn_y = pad_rect.bottom + 40
    save_btn = pygame.Rect(DISPLAY_WIDTH//2 - btn_w -20, btn_y, btn_w, btn_h)
    cancel_btn = pygame.Rect(DISPLAY_WIDTH//2 + 20, btn_y, btn_w, btn_h)
    clear_btn = pygame.Rect(DISPLAY_WIDTH//2 - btn_w//2, btn_y + 80, btn_w, btn_h)

    font_btn = pygame.font.Font(None, 44)
    font_title = pygame.font.Font(None, 48)
    drawing = False
    last_pos = None

    while True:
        mouse_pos = pygame.mouse.get_pos()
        save_hover = save_btn.collidepoint(mouse_pos)
        cancel_hover = cancel_btn.collidepoint(mouse_pos)
        clear_hover = clear_btn.collidepoint(mouse_pos)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if save_btn.collidepoint(mouse_pos):
                    out_path = "signature_temp.png"
                    pygame.image.save(drawing_surface, out_path)
                    return Image.open(out_path)
                if cancel_btn.collidepoint(mouse_pos):
                    return None
                if clear_btn.collidepoint(mouse_pos):
                    drawing_surface.fill((255,255,255))
                    continue
                if pad_rect.collidepoint(mouse_pos):
                    drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
        if drawing and pygame.mouse.get_pressed()[0]:
            rel_x = mouse_pos[0]-pad_x
            rel_y = mouse_pos[1]-pad_y
            if 0 <= rel_x < SIGNATURE_RECT_WIDTH and 0 <= rel_y < SIGNATURE_RECT_HEIGHT:
                if last_pos is not None:
                    pygame.draw.line(drawing_surface, DRAW_COLOUR, (last_pos[0]-pad_x, last_pos[1]-pad_y), (rel_x, rel_y), BRUSH_SIZE)
                else:
                    pygame.draw.circle(drawing_surface, DRAW_COLOUR, (rel_x, rel_y), BRUSH_SIZE//2)
                last_pos = mouse_pos
        else:
            last_pos = None

        screen.fill((30,30,30))
        screen.blit(overlay, (0,0))
        title = font_title.render("Please sign below", True, (255,255,255))
        title_rect = title.get_rect(center=(DISPLAY_WIDTH//2, pad_y-60))
        screen.blit(title, title_rect)
        pygame.draw.rect(screen, (255,255,255), pad_rect, border_radius=10)
        pygame.draw.rect(screen, (200,200,200), pad_rect, 3, border_radius=10)
        screen.blit(drawing_surface, (pad_x, pad_y))
        draw_button(screen, save_btn, "SAVE", font_btn, save_hover)
        draw_button(screen, cancel_btn, "CANCEL", font_btn, cancel_hover)
        draw_button(screen, clear_btn, "CLEAR", font_btn, clear_hover)
        pygame.display.flip()
        clock.tick(60)



# ---------- URL Fetch and QR Code ----------
def get_url_from_excel(counter, excel_path=EXCEL_URL_SHEET, sheet_name="Sheet1"):
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Excel file not found: {excel_path}")
        return None
    if counter <= len(df):
        url = df.iloc[counter-2,1]
        if isinstance(url,str) and url.startswith("http"):
            print(f"Fetched URL for photo {counter}: {url}")
            return url
    print(f"No valid URL found for photo {counter}.")
    return None



def generate_qr_code(url, counter, output_folder="qr_codes"):
    os.makedirs(output_folder, exist_ok=True)
    qr = qrcode.make(url)
    qr_path = os.path.join(output_folder, f"qr_code_{counter}.png")
    qr.save(qr_path)
    print(f"QR code saved as {qr_path}")
    return qr_path



# --------- Gradients, Countdown, Letterbox Helpers ---------
def create_gradient_surface(w, h, top, bottom):
    surf = pygame.Surface((w, h))
    for y in range(h):
        ratio = y / h
        r = int(top[0] * (1 - ratio) + bottom[0] * ratio)
        g = int(top[1] * (1 - ratio) + bottom[1] * ratio)
        b = int(top[2] * (1 - ratio) + bottom[2] * ratio)
        pygame.draw.line(surf, (r, g, b), (0, y), (w, y))
    return surf



def preload_countdown_images(folder=COUNTDOWN_FOLDER, scale=1.5):
    """Load countdown images with larger scale for better visibility"""
    images = {}
    for fname in ("3.png", "2.png", "1.png"):
        path = os.path.join(folder, fname)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Scale up for better visibility (577x433 -> ~865x650)
                img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), cv2.INTER_AREA)
                images[fname] = img
                print(f"Loaded countdown image: {fname} - size: {img.shape}")
    return images



def blend_countdown_overlay(frame, overlay, x1,x2,y1,y2):
    if y1<0 or x1<0 or y2>frame.shape[0] or x2>frame.shape[1]:
        return frame
    region = frame[y1:y2, x1:x2].astype(float)
    if overlay.shape[2]==4:
        ov = overlay[..., :3].astype(float)
        mask = overlay[..., 3:] / 255.0
    else:
        ov = overlay.astype(float)
        mask = np.ones((*overlay.shape[:2],1))
    if region.shape != ov.shape:
        return frame
    blended = (1.0 - mask)*region + mask*ov
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame



def resize_frame_letterbox(frame, target_w, target_h):
    h, w = frame.shape[:2]
    aspect = w/h
    target_aspect = target_w/target_h
    if aspect > target_aspect:
        new_w = target_w
        new_h = int(target_w/aspect)
    else:
        new_h = target_h
        new_w = int(target_h*aspect)
    resized = cv2.resize(frame, (new_w,new_h))
    canvas = np.zeros((target_h,target_w,3), dtype=np.uint8)
    y_off = (target_h-new_h)//2
    x_off = (target_w-new_w)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas



# ---------- Display signed photo with button ----------
def display_signed_photo_with_button(signed_photo_path: str, duration: int = 4):
    img = cv2.imread(signed_photo_path)
    if img is None:
        print("Error: could not load signed photo")
        return
    img = resize_frame_letterbox(img, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pygame.init()
    if FULLSCREEN_MODE:
        screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption("Your Signed Photo")
    img_surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
    start_t = time.time()
    clock = pygame.time.Clock()

    while True:
        elapsed = time.time() - start_t
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
        if elapsed > duration:
            return
        screen.blit(img_surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)




# ---------- QR Code Display ----------
def display_qr_code_with_button(qr_path):
    pygame.init()
    qr_img = pygame.image.load(qr_path)
    if FULLSCREEN_MODE:
        screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
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

    while True:
        hover = btn_rect.collidepoint(pygame.mouse.get_pos())
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if btn_rect.collidepoint(pygame.mouse.get_pos()):
                    return
        screen.blit(bg_img, (0, 0))
        screen.blit(qr_img, (qr_x, qr_y))
        draw_button(screen, btn_rect, "DONE", font, hover)
        pygame.display.flip()



# ---------- Main Live Capture Workflow from Socket ----------
def live_video_capture_and_workflow(frame_path=FRAME_IMAGE_PATH):
    photos_dir = "photos"
    os.makedirs(photos_dir, exist_ok=True)
    counter = max((int(f.split(".")[0]) for f in os.listdir(photos_dir) if f.split(".")[0].isdigit()), default=0)

    countdown_imgs = preload_countdown_images()
    pygame.init()
    # Use FULLSCREEN for immersive experience (fallback to windowed if unavailable)
    if FULLSCREEN_MODE:
        screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption("Live Camera Feed")
    clock = pygame.time.Clock()

    # Load background image
    if os.path.exists(BACKGROUND_IMAGE_PATH):
        bg_img = pygame.image.load(BACKGROUND_IMAGE_PATH)
        bg_img = pygame.transform.scale(bg_img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        print(f"Background image loaded: {BACKGROUND_IMAGE_PATH}")
    else:
        bg_img = create_gradient_surface(DISPLAY_WIDTH, DISPLAY_HEIGHT, (15, 23, 42), (25, 40, 70))
        print("Background image not found, using gradient")

    # Define camera preview area (centered for the white rectangle in background)
    # For 1920x1080 display - center the preview nicely
    preview_width = 1100
    preview_height = 650
    preview_x = (DISPLAY_WIDTH - preview_width) // 2
    preview_y = (DISPLAY_HEIGHT - preview_height) // 2  # Perfect center
    camera_preview_rect = pygame.Rect(preview_x, preview_y, preview_width, preview_height)
    print(f"Camera preview area: x={preview_x}, y={preview_y}, w={preview_width}, h={preview_height}")
    
    # Capture button (centered at bottom)
    capture_btn_rect = pygame.Rect(DISPLAY_WIDTH // 2 - 150, DISPLAY_HEIGHT - 150, 300, 70)
    # Back button (bottom-left) - allows returning to main UI without taking a photo
    back_btn_rect = pygame.Rect(50, DISPLAY_HEIGHT - 150, 200, 70)
    font_btn = pygame.font.Font(None, 48)
    
    # Debug: Check countdown images
    print(f"Countdown images loaded: {list(countdown_imgs.keys())}")

    inactivity_timeout = 4  # seconds for signature inactivity skip
    display_signed_duration = 4  # seconds to show signed or framed photo before QR

    while True:
        # --- Show live camera feed with capture button ---
        capture_clicked = False
        go_back = False
        print("\nWAITING FOR USER - Click CAPTURE button to take photo")
        print(f"   Camera preview area: ({preview_x}, {preview_y}) {preview_width}x{preview_height}")
        print(f"   Capture button: {capture_btn_rect}")
        
        while not capture_clicked:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    print("User quit")
                    return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    print("ESC pressed")
                    return
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    # Back button click -> return to main UI (exit demo)
                    if back_btn_rect.collidepoint((mouse_x, mouse_y)):
                        print(f"BACK clicked at ({mouse_x}, {mouse_y}) - minimizing window (staying alive)")
                        try:
                            # Minimize the pygame window so the main UI can resume foreground
                            pygame.display.iconify()
                        except Exception:
                            pass
                        # Try to bring the main KAIRA UI to the foreground (Windows)
                        try:
                            if os.name == 'nt':
                                import ctypes
                                import ctypes.wintypes as wintypes

                                user32 = ctypes.windll.user32
                                EnumWindows = user32.EnumWindows
                                EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

                                found = False

                                @EnumWindowsProc
                                def _enum(hwnd, lParam):
                                    if not user32.IsWindowVisible(hwnd):
                                        return True
                                    length = user32.GetWindowTextLengthW(hwnd)
                                    if length == 0:
                                        return True
                                    buf = ctypes.create_unicode_buffer(length + 1)
                                    user32.GetWindowTextW(hwnd, buf, length + 1)
                                    title = buf.value
                                    # Look for KAIRA window caption
                                    if 'KAIRA' in title:
                                        SW_RESTORE = 9
                                        user32.ShowWindow(hwnd, SW_RESTORE)
                                        try:
                                            user32.SetForegroundWindow(hwnd)
                                        except Exception:
                                            pass
                                        # mark found in outer scope
                                        nonlocal found
                                        found = True
                                        return False
                                    return True

                                EnumWindows(_enum, 0)
                        except Exception as e:
                            print(f"⚠️  Could not bring KAIRA UI to front: {e}")

                        # Signal we should go back to waiting state without exiting process
                        go_back = True
                        # Wait briefly to avoid accidental immediate re-activation
                        time.sleep(0.2)
                        break
                    
                        
                        
                    elif capture_btn_rect.collidepoint((mouse_x, mouse_y)):
                        print(f"CAPTURE clicked at ({mouse_x}, {mouse_y})")
                        capture_clicked = True
                    else:
                        print(f"   Click at ({mouse_x}, {mouse_y}) - not on button")

            # Draw background first
            screen.blit(bg_img, (0, 0))
            
            # Get live frame
            frame = get_latest_frame()
            if frame is not None:
                # Resize frame to fit camera preview area
                frame_resized = resize_frame_without_stretch(frame, camera_preview_rect.width, camera_preview_rect.height)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frame_surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                
                # Draw frame in camera preview area
                screen.blit(frame_surf, camera_preview_rect)
            else:
                # Draw placeholder if no frame
                pygame.draw.rect(screen, (50, 50, 50), camera_preview_rect)
                no_feed_font = pygame.font.Font(None, 36)
                text = no_feed_font.render("Waiting for camera feed...", True, (200, 200, 200))
                text_rect = text.get_rect(center=camera_preview_rect.center)
                screen.blit(text, text_rect)
            
            # Draw white border around preview area for visibility
            pygame.draw.rect(screen, (255, 255, 255), camera_preview_rect, 3)
            
            # Draw capture button
            mouse_pos = pygame.mouse.get_pos()
            hover = capture_btn_rect.collidepoint(mouse_pos)
            draw_button(screen, capture_btn_rect, "CAPTURE", font_btn, hover)
            # Draw back button (left side)
            back_hover = back_btn_rect.collidepoint(mouse_pos)
            draw_button(screen, back_btn_rect, "BACK", font_btn, back_hover)
                
            pygame.display.flip()
            clock.tick(60)
            
        # If user clicked BACK, skip capture workflow and return to waiting loop
        if go_back:
            continue

        # --- Countdown and capture ---
        print("\n=== STARTING COUNTDOWN ===")
        countdown_font = pygame.font.Font(None, 120)  # Large font for countdown text
        for idx, fname in enumerate(["3.png", "2.png", "1.png"]):
            countdown_num = str(3 - idx)  # "3", "2", "1"
            print(f"Countdown: {countdown_num}")
            start = time.time()
            while time.time() - start < 1.0:  # 1 second per countdown image
                # Handle events to prevent freezing
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        return
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        return
                
                fr2 = get_latest_frame()
                if fr2 is None:
                    continue

                # Draw background
                screen.blit(bg_img, (0, 0))

                # Resize frame to fit camera preview
                fr2_resized = resize_frame_without_stretch(fr2, camera_preview_rect.width, camera_preview_rect.height)

                # Overlay countdown on frame
                if fname in countdown_imgs:
                    h, w = fr2_resized.shape[:2]
                    ci = countdown_imgs[fname]
                    ch, cw = ci.shape[:2]
                    x1 = (w - cw) // 2
                    y1 = (h - ch) // 2
                    fr2_resized = blend_countdown_overlay(fr2_resized, ci, x1, x1 + cw, y1, y1 + ch)
                else:
                    print(f"WARNING: Countdown image {fname} not found in loaded images")

                fr2_rgb = cv2.cvtColor(fr2_resized, cv2.COLOR_BGR2RGB)
                surf2 = pygame.surfarray.make_surface(fr2_rgb.swapaxes(0, 1))
                screen.blit(surf2, camera_preview_rect)
                
                # Draw white border around preview during countdown
                pygame.draw.rect(screen, (255, 255, 255), camera_preview_rect, 5)  # Thicker border
                
                # Also draw countdown number as text overlay (backup if PNG not visible)
                countdown_text = countdown_font.render(countdown_num, True, (255, 255, 0))
                countdown_text_rect = countdown_text.get_rect(center=camera_preview_rect.center)
                # Draw text shadow for better visibility
                shadow_text = countdown_font.render(countdown_num, True, (0, 0, 0))
                shadow_rect = countdown_text_rect.copy()
                shadow_rect.x += 4
                shadow_rect.y += 4
                screen.blit(shadow_text, shadow_rect)
                screen.blit(countdown_text, countdown_text_rect)
                
                pygame.display.flip()
                clock.tick(60)

        # Capture photo
        print("SNAP! Capturing photo...")
        photo = get_latest_frame()
        if photo is None:
            print("ERROR: Capture failed: no frame available")
            return  # or break
        # Save captured photo as-is (no auto-rotation)
        counter += 1
        photo_path = os.path.join(photos_dir, f"{counter}.jpg")
        cv2.imwrite(photo_path, photo)
        print(f"Photo saved: {photo_path}")
        
        print("Creating framed photo...")
        framed = compose_photo_with_frame(counter, photo_path, frame_path)
        print(f"Framed photo created: {framed}")

        # Signature capture with inactivity timeout
        sig_img = None

        def pygame_signature_with_timeout(timeout=4):
            """Signature capture with automatic save on inactivity timeout"""
            pygame.init()
            if FULLSCREEN_MODE:
                flags = pygame.FULLSCREEN
            else:
                flags = 0
            screen_sig = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), flags)
            pygame.display.set_caption("Sign your signature")
            clock_sig = pygame.time.Clock()

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

                # Auto-save and return if idle timeout
                if not events and (time.time() - last_event_time) > timeout:
                    try:
                        out_path = "signature_temp.png"
                        pygame.image.save(drawing_surface, out_path)
                        return Image.open(out_path)
                    except Exception as e:
                        print(f"Error saving signature on timeout: {e}")
                        return None

                for event in events:
                    last_event_time = time.time()
                    if event.type == pygame.QUIT:
                        try:
                            out_path = "signature_temp.png"
                            pygame.image.save(drawing_surface, out_path)
                            return Image.open(out_path)
                        except Exception as e:
                            print(f"Error saving signature at quit: {e}")
                            return None
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if save_btn.collidepoint(mouse_pos):
                            out_path = "signature_temp.png"
                            pygame.image.save(drawing_surface, out_path)
                            return Image.open(out_path)
                        if cancel_btn.collidepoint(mouse_pos):
                            # Also save current drawing on cancel before skipping
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

                screen_sig.fill((30, 30, 30))
                screen_sig.blit(overlay, (0, 0))
                title = font_title.render("Please sign below", True, (255, 255, 255))
                title_rect = title.get_rect(center=(DISPLAY_WIDTH // 2, pad_y - 60))
                screen_sig.blit(title, title_rect)
                pygame.draw.rect(screen_sig, (255, 255, 255), pad_rect, border_radius=10)
                pygame.draw.rect(screen_sig, (200, 200, 200), pad_rect, 3, border_radius=10)
                screen_sig.blit(drawing_surface, (pad_x, pad_y))
                draw_button(screen_sig, save_btn, "SAVE", font_btn, save_hover)
                draw_button(screen_sig, cancel_btn, "CANCEL", font_btn, cancel_hover)
                draw_button(screen_sig, clear_btn, "CLEAR", font_btn, clear_hover)
                pygame.display.flip()
                clock_sig.tick(60)


        sig_img = pygame_signature_with_timeout()

        if sig_img is None:
            # Create transparent blank signature if timeout happens
            print("No signature input detected; applying blank signature overlay.")
            sig_img = Image.new("RGBA", (SIGNATURE_RECT_WIDTH, SIGNATURE_RECT_HEIGHT), (0, 0, 0, 0))

        signed_path = paste_signature_on_frame(counter, framed, sig_img)
        display_signed_photo_with_button(signed_path)

        # Display signed photo for 4 seconds
        start_show = time.time()
        signed_img_cv = cv2.imread(signed_path)
        signed_img_cv = resize_frame_letterbox(signed_img_cv, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        signed_img_rgb = cv2.cvtColor(signed_img_cv, cv2.COLOR_BGR2RGB)
        signed_surf = pygame.surfarray.make_surface(signed_img_rgb.swapaxes(0, 1))

        while time.time() - start_show < 4:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
            screen.blit(signed_surf, (0, 0))
            pygame.display.flip()
            clock.tick(60)

        # Proceed to QR Code display
        url = get_url_from_excel(counter)
        if url:
            qr_path = generate_qr_code(url, counter)
            display_qr_code_with_button(qr_path)

        print(f"✓ Complete workflow finished for photo {counter}")

# ---------- Main ----------
def main():
    global running
    print("="*70)
    print("STARTING PHOTO BOOTH")
    print("="*70)
    print(f"Display resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} (fullscreen={FULLSCREEN_MODE})")
    print(f"Socket URL: {SOCKET_URL}")
    print(f"Background: {BACKGROUND_IMAGE_PATH}")
    print(f"Frame: {FRAME_IMAGE_PATH}")
    print(f"Countdown folder: {COUNTDOWN_FOLDER}")
    
    pygame.init()
    
    # Start ZMQ frame receiver thread
    print("\nStarting socket frame receiver thread...")
    receiver_thread = threading.Thread(target=zmq_frame_receiver, daemon=True)
    receiver_thread.start()
    
    # Wait a bit for connection to be established
    print("Waiting for connection...")
    time.sleep(2)
    
    # Go directly to live feed
    print("Starting live feed...")
    print("="*70)
    live_video_capture_and_workflow()
    
    # Cleanup
    running = False
    receiver_thread.join(timeout=2)
    print("\n" + "="*70)
    print("Photo Booth closed.")
    print("="*70)
    pygame.quit()



if __name__ == "__main__":
    main()
