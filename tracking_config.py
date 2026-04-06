"""
Tracking Service Configuration
Connection settings for Jetson Nano and tracking parameters
"""

# --- Jetson Nano Ethernet Configuration ---
JETSON_IP = "192.168.10.2"   # Jetson Nano's Ethernet IP address
JETSON_PORT = 6000           # Communication port
CONNECTION_TIMEOUT = None          # No connection timeout - keep alive until KAIRA stops

# --- Tracking Service API Configuration ---
TRACKING_SERVICE_HOST = "localhost"
TRACKING_SERVICE_PORT = 8003  # FastAPI tracking service port

# --- Movement Commands ---
COMMANDS = {
    "YES": "YES",             # Handshake command
    "FORWARD": "FORWARD",
    "BACKWARD": "BACKWARD",
    "TURN_LEFT": "TURN_LEFT",
    "TURN_RIGHT": "TURN_RIGHT",
    "STOP": "STOP"
}

# --- Tracking Parameters ---
CENTER_TOLERANCE = 80          # Pixels from center to consider "centered"
MIN_BBOX_AREA = 5000          # Minimum area to consider a valid target
OPTIMAL_BBOX_HEIGHT = 200     # Target height for person in frame
HEIGHT_TOLERANCE = 80         # Tolerance for optimal height

# --- Connection Settings ---
MAX_LOST_FRAMES = 60          # Give up after 1 second at 30fps
RECONNECT_ATTEMPTS = 3        # Number of reconnection attempts
RECONNECT_DELAY = 2           # Delay between reconnection attempts (seconds)

# --- Debug Settings ---
DEBUG_MODE = False            # Enable debug logging
SHOW_TRACKING_OVERLAY = True  # Show tracking visualization

print("✅ Tracking configuration loaded")
print(f"📡 Jetson Nano: {JETSON_IP}:{JETSON_PORT}")
print(f"🎯 Tracking Service: {TRACKING_SERVICE_HOST}:{TRACKING_SERVICE_PORT}")


# --- Handshake Function ---
def send_yes_handshake(ip=JETSON_IP, port=JETSON_PORT, timeout=5):
    """
    Send YES handshake command to Jetson Nano
    NOTE: Connection stays open and is NOT closed - only closes when KAIRA stops
    Returns: (success: bool, message: str, socket: socket or None)
    """
    import socket

    try:
        print(f"\n🤝 Attempting handshake with Jetson Nano at {ip}:{port}...")

        # Create socket connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(timeout)

        # Connect to Jetson
        print(f"📡 Connecting to {ip}:{port}...")
        client_socket.connect((ip, port))
        print(f"✅ Connected successfully!")

        # Send YES command
        yes_command = COMMANDS["YES"] + "\n"
        print(f"📤 Sending handshake: '{COMMANDS['YES']}'")
        client_socket.sendall(yes_command.encode())
        print(f"✅ Handshake sent!")

        # Try to receive response (optional, Nano may not send response)
        try:
            client_socket.settimeout(2)  # Short timeout for response
            response = client_socket.recv(1024)
            if response:
                print(f"📥 Response from Jetson: {response.decode().strip()}")
        except socket.timeout:
            print(f"⏱️  No response from Jetson (this is normal)")

        # Set socket to no timeout for persistent connection
        client_socket.settimeout(None)

        # Keep connection open - DO NOT close until KAIRA stops
        print(f"🔗 Connection kept alive - will persist until KAIRA stops\n")

        return True, "Handshake successful - connection alive", client_socket

    except socket.timeout:
        error_msg = f"Connection timeout - Jetson not responding at {ip}:{port}"
        print(f"❌ {error_msg}")
        return False, error_msg, None

    except ConnectionRefusedError:
        error_msg = f"Connection refused - Jetson may not be running server at {ip}:{port}"
        print(f"❌ {error_msg}")
        return False, error_msg, None

    except OSError as e:
        if "WinError 10051" in str(e) or "unreachable" in str(e).lower():
            error_msg = f"Network unreachable - Check Ethernet connection and IP configuration"
            print(f"❌ {error_msg}")
            print(f"💡 Tip: Run 'ipconfig' to verify your Ethernet adapter has IP 192.168.10.1")
        else:
            error_msg = f"Network error: {e}"
            print(f"❌ {error_msg}")
        return False, error_msg, None

    except Exception as e:
        error_msg = f"Handshake failed: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg, None


if __name__ == "__main__":
    """
    When run directly, test the YES handshake with Jetson Nano
    Usage: python tracking_config.py
    """
    print("\n" + "="*60)
    print("🎯 JETSON NANO HANDSHAKE TEST")
    print("="*60)

    # Send handshake
    success, message, client_socket = send_yes_handshake()

    print("="*60)
    if success:
        print("✅ HANDSHAKE TEST PASSED")
        print(f"🔗 Connection is alive and persistent")
        print(f"⚠️  Remember to close socket when done: client_socket.close()")
    else:
        print("❌ HANDSHAKE TEST FAILED")
        print(f"   Reason: {message}")
    print("="*60 + "\n")

    # Cleanup in test mode
    if success and client_socket:
        print("🧹 Cleaning up test connection...")
        client_socket.close()
        print("✅ Test connection closed")
