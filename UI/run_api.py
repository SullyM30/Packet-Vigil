"""
Application entry point.
Run this file to start the Flask + Dash application with the detection engine.

Usage:
    python run_api.py
"""

# import the Flask server instance created in api/app.py
# this already includes routes, dashboard, and optional detection engine
from api.app import server

# import runtime configuration values
# these control where and how the server runs
from api.config import API_HOST, API_PORT, DEBUG_MODE

# only run this block when file is executed directly
# prevents server from starting if imported elsewhere
if __name__ == '__main__':

    # print startup info so user knows where to connect
    print("=" * 60)
    print("Hybrid Anomaly Detection System")
    print("=" * 60)

    # show main server address
    print(f"Starting server on http://{API_HOST}:{API_PORT}")

    # dashboard UI location
    print(f"Dashboard: http://{API_HOST}:{API_PORT}/dashboard/")

    # API endpoint root
    print(f"API: http://{API_HOST}:{API_PORT}/api")
    print("=" * 60)

    # start Flask server
    # debug mode enables auto reload and detailed errors
    # threaded mode allows multiple requests at the same time
    server.run(debug=DEBUG_MODE, host=API_HOST, port=API_PORT, threaded=True)
