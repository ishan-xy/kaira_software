"""
KAIRA Service Launcher
Starts all services in separate terminal windows
Windows Compatible
"""

import subprocess
import sys
import os
import time
import platform

# Service definitions
SERVICES = [
    {
        'name': 'Camera Server',
        'script': 'serve_camera.py',
        'description': 'WebRTC camera streaming server'
    },
    {
        'name': 'Camera Receiver',
        'script': 'camera_recv.py',
        'description': 'Receives camera frames via WebRTC'
    },
    {
        'name': 'Face Recognition',
        'script': 'face_recognition_service.py',
        'description': 'Processes frames for face recognition'
    },
    {
        'name': 'Live API',
        'script': 'liveapi.py',
        'description': 'Gemini Live API integration'
    },
    {
        'name': 'KAIRA Main',
        'script': 'main.py',
        'description': 'Main KAIRA interface (STT + UI)'
    }
]

def get_python_executable():
    """Get the current Python executable path"""
    # Check if we have a virtual environment
    venv_python = os.path.join(os.path.dirname(__file__), '.venv', 'Scripts', 'python.exe')
    if os.path.exists(venv_python):
        print(f"🐍 Using virtual environment Python: {venv_python}")
        return venv_python
    return sys.executable

def launch_service_windows(service):
    """Launch a service in a new Windows terminal"""
    python_exe = get_python_executable()
    script_path = service['script']
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"⚠️  Warning: {script_path} not found!")
        return None
    
    # Use 'start' command to open new cmd window
    # /K keeps window open, /MIN starts minimized (remove for debugging)
    cmd = f'start cmd /K "title {service["name"]} && {python_exe} {script_path}"'
    
    try:
        subprocess.Popen(cmd, shell=True)
        print(f"✅ Launched: {service['name']}")
        return True
    except Exception as e:
        print(f"❌ Failed to launch {service['name']}: {e}")
        return None

def launch_service_unix(service):
    """Launch a service in a new Unix/Mac terminal"""
    python_exe = get_python_executable()
    script_path = service['script']
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"⚠️  Warning: {script_path} not found!")
        return None
    
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            # Use AppleScript to open Terminal
            script = f'''
            tell application "Terminal"
                do script "cd {os.getcwd()} && {python_exe} {script_path}"
                activate
            end tell
            '''
            subprocess.Popen(['osascript', '-e', script])
        else:  # Linux
            # Try different terminal emulators
            terminals = ['gnome-terminal', 'konsole', 'xterm']
            launched = False
            
            for terminal in terminals:
                try:
                    if terminal == 'gnome-terminal':
                        subprocess.Popen([
                            terminal, 
                            '--',
                            python_exe,
                            script_path
                        ])
                    elif terminal == 'konsole':
                        subprocess.Popen([
                            terminal,
                            '-e',
                            python_exe,
                            script_path
                        ])
                    else:  # xterm
                        subprocess.Popen([
                            terminal,
                            '-e',
                            python_exe,
                            script_path
                        ])
                    launched = True
                    break
                except FileNotFoundError:
                    continue
            
            if not launched:
                print(f"⚠️  No supported terminal found for {service['name']}")
                return None
        
        print(f"✅ Launched: {service['name']}")
        return True
    except Exception as e:
        print(f"❌ Failed to launch {service['name']}: {e}")
        return None

def main():
    """Main launcher function"""
    print("=" * 70)
    print("🚀 KAIRA SERVICE LAUNCHER")
    print("=" * 70)
    print()
    
    # Detect platform
    system = platform.system()
    print(f"📍 Detected platform: {system}")
    print()
    
    # Show services to be launched
    print("Services to launch:")
    for i, service in enumerate(SERVICES, 1):
        print(f"  {i}. {service['name']}: {service['description']}")
    print()
    
    # Ask for confirmation
    response = input("Launch all services? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Launch cancelled.")
        return
    
    print()
    print("=" * 70)
    print("Launching services...")
    print("=" * 70)
    print()
    
    # Launch each service with delay
    launch_function = launch_service_windows if system == "Windows" else launch_service_unix
    
    for service in SERVICES:
        launch_function(service)
        time.sleep(1)  # Small delay between launches
    
    print()
    print("=" * 70)
    print("✅ All services launched!")
    print("=" * 70)
    print()
    print("📝 Notes:")
    print("  - Each service runs in its own terminal window")
    print("  - Check individual windows for logs and errors")
    print("  - Close all windows or press Ctrl+C to stop")
    print()
    print("⚠️  If any service failed to start:")
    print("  1. Check if the script file exists")
    print("  2. Verify all dependencies are installed")
    print("  3. Check for port conflicts")
    print()
    print("Press Ctrl+C to exit launcher...")
    
    try:
        # Keep launcher running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n✅ Launcher stopped.")

if __name__ == "__main__":
    main()
