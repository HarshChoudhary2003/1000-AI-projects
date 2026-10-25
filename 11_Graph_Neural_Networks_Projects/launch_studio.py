#!/usr/bin/env python3
"""
Graph Neural Networks Studio - One-Click Launcher
Starts a local lightweight HTTP server and opens the interactive workbench in the default browser.
"""

import http.server
import os
import socketserver
import threading
import time
import webbrowser

PORT = 8088
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def log_message(self, format, *args):
        # Keep terminal output clean
        pass

def open_browser():
    time.sleep(0.8)
    url = f"http://localhost:{PORT}/index.html"
    print(f"\n==================================================================")
    print(f"  🕸️  Graph Neural Networks Interactive Studio")
    print(f"  🚀  Serving at: {url}")
    print(f"  Press Ctrl+C to stop the studio server.")
    print(f"==================================================================\n")
    webbrowser.open(url)

if __name__ == "__main__":
    # Ensure port is reusable
    socketserver.TCPServer.allow_reuse_address = True
    
    # Attempt port or fallback
    for p in range(PORT, PORT + 10):
        try:
            httpd = socketserver.TCPServer(("", p), Handler)
            PORT = p
            break
        except OSError:
            continue

    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStudio server stopped.")
