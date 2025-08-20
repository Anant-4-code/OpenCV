import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

# Configuration
PORT = 8000
HTML_FILE = "camera_test_landing.html"

def create_server():
    """Create and start a simple HTTP server"""
    
    # Change to the directory containing this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if HTML file exists
    if not os.path.exists(HTML_FILE):
        print(f"❌ Error: {HTML_FILE} not found!")
        print("Make sure the HTML file is in the same directory as this script.")
        return False
    
    # Create server
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"🚀 Server started successfully!")
            print(f"📁 Serving files from: {os.getcwd()}")
            print(f"🌐 Local URL: http://localhost:{PORT}")
            print(f"📱 Camera Test Page: http://localhost:{PORT}/{HTML_FILE}")
            print(f"⏹️  Press Ctrl+C to stop the server")
            print("-" * 50)
            
            # Automatically open the camera test page
            webbrowser.open(f"http://localhost:{PORT}/{HTML_FILE}")
            
            # Start serving
            httpd.serve_forever()
            
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use!")
            print(f"Try using a different port or close other applications using port {PORT}")
        else:
            print(f"❌ Server error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
        return True

if __name__ == "__main__":
    print("🎥 Camera Test Landing Page - Local Server")
    print("=" * 50)
    
    success = create_server()
    
    if success:
        print("✅ Server stopped successfully")
    else:
        print("❌ Failed to start server")
        input("Press Enter to exit...")
