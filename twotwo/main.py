"""TwoTwo - Local AI Assistant

Entry point for the TwoTwo application.
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from core.controller import Controller


def main():
    """Main entry point for TwoTwo."""
    # Create application
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("TwoTwo")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("TwoTwo")
    
    # Prevent app from quitting when windows are hidden
    app.setQuitOnLastWindowClosed(False)
    
    # Enable high DPI scaling
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create and start controller
    controller = Controller(app)
    controller.start()
    
    # Run event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

