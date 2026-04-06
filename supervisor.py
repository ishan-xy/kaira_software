#!/usr/bin/env python3
import subprocess
import time
import sys
import atexit
import os
import threading
import requests
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

uv_path = "uv"

sequential_services = [
    {
        "name": "Camera Server",
        "cmd": [uv_path, "run", str(BASE_DIR / "serve_camera.py")],
        "cwd": BASE_DIR,
        "delay": 5
    },
    {
        "name": "Camera Receive",
        "cmd": [uv_path, "run", str(BASE_DIR / "camera_recv.py")],
        "cwd": BASE_DIR,
        "delay": 3
    },
    {
        "name": "Face Reco CV",
        "cmd": [uv_path, "run", str(BASE_DIR / "face_reco" / "cv.py")],
        "cwd": BASE_DIR,
        "delay": 3
    },
]

parallel_services = [
    {
        "name": "Live API",
        "cmd": [uv_path, "run", str(BASE_DIR / "liveapi.py")],
        "cwd": BASE_DIR
    },
    {
        "name": "Main UI",
        "cmd": [uv_path, "run", str(BASE_DIR / "main.py")],
        "cwd": BASE_DIR
    },
]

processes = {}
log_buffer = []
log_lock = threading.Lock()

def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=3
        )
    except:
        pass

def log(msg):
    print(msg)
    with log_lock:
        log_buffer.append(msg)

def telegram_loop():
    while True:
        time.sleep(2)
        with log_lock:
            if not log_buffer:
                continue
            chunk = "\n".join(log_buffer[:50])
            del log_buffer[:50]
        if len(chunk) > 3500:
            chunk = chunk[-3500:]
        send_telegram(chunk)

threading.Thread(target=telegram_loop, daemon=True).start()

def stream_process_output(name, proc):
    for line in proc.stdout:
        line = line.rstrip()
        log(f"[{name}] {line}")

def start_service(service_config):
    name = service_config["name"]
    cmd = service_config["cmd"]
    cwd = service_config.get("cwd", ".")
    script_path = cmd[2]

    if not os.path.exists(script_path):
        log(f"[Manager] ERROR: Script not found for {name} at {script_path}")
        return None

    log(f"[Manager] Starting {name}...")
    try:
        env = os.environ.copy()
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        processes[name] = p
        threading.Thread(target=stream_process_output, args=(name, p), daemon=True).start()
        return p
    except Exception as e:
        log(f"[Manager] Failed to start {name}: {e}")
        return None

def cleanup():
    log("[Manager] Stopping all services...")
    all_services = [s["name"] for s in parallel_services] + [s["name"] for s in sequential_services]
    for name in reversed(all_services):
        p = processes.get(name)
        if p and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except:
                p.kill()
    log("[Manager] All services stopped.")

atexit.register(cleanup)

def main():
    send_telegram("Supervisor started")

    log(f"[Manager] Using global 'uv' command.")
    log(f"[Manager] Project base directory: {BASE_DIR}")

    if not os.getenv("VIRTUAL_ENV"):
        log("[Manager] WARNING: No virtual environment is activated.")
        log("Please activate your venv first: source .venv/bin/activate")

    log("--- Starting Sequential Services ---")
    for service in sequential_services:
        if start_service(service) is None:
            log(f"[Manager] FATAL: Failed to start required service {service['name']}. Exiting.")
            sys.exit(1)
        delay = service.get("delay", 1)
        log(f"[Manager] Waiting {delay}s for {service['name']} to initialize...")
        time.sleep(delay)

    log("--- Starting Parallel Services ---")
    for service in parallel_services:
        start_service(service)

    log("[Manager] All services are running. Monitoring for crashes...")

    try:
        while True:
            time.sleep(5)
            for service in sequential_services:
                name = service["name"]
                p = processes.get(name)
                if p is None or p.poll() is not None:
                    log(f"[Manager] FATAL: Required service {name} crashed. Shutting down all services.")
                    sys.exit(1)
            for service in parallel_services:
                name = service["name"]
                p = processes.get(name)
                if p is None or p.poll() is not None:
                    if p:
                        log(f"[Manager] Service {name} crashed (exit code {p.returncode}). Restarting...")
                    else:
                        log(f"[Manager] Service {name} failed to start. Retrying...")
                    start_service(service)
    except KeyboardInterrupt:
        log("[Manager] Shutdown signal received.")
        sys.exit(0)

if __name__ == "__main__":
    main()
