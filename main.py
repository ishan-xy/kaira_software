# main.py

import sys
import traceback
from kaira_core import KAIRACore
from kaira_ui import KAIRAUI

def main():
    """Main function to initialize and run KAIRA core and UI."""
    core = None
    ui = None
    try:
        # 1. Initialize Core
        core = KAIRACore()
        
        # 2. Initialize UI, passing the core to it
        ui = KAIRAUI(core)
        
        # 3. Start the core services (audio, STT)
        core.start()
        
        # 4. Run the UI (this is the main blocking loop)
        ui.run()

    except KeyboardInterrupt:
        print("\n\n⚡ KAIRA interrupted - Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ UNHANDLED ERROR: {e}")
        traceback.print_exc()
    finally:
        # 5. Stop services
        if core:
            core.stop()
        # UI cleanup is called by its own .run() method,
        # but we call it again just in case of an error.
        if ui:
            ui.cleanup()
        
        print("✅ KAIRA has shut down completely.")
        sys.exit()

if __name__ == "__main__":
    main()