"""
Main Flask application for Hybrid IDS Backend API.
Phase 1: API Layer - Dheeraj

This is the backend-only version focusing on REST API endpoints.
Dashboard (Phase 2) and Detection Engine (Phase 3) are handled separately.
"""

from flask import Flask
from flask_cors import CORS
import logging

# load app settings from the shared config file
from api.config import SECRET_KEY, API_HOST, API_PORT, DEBUG_MODE, LOG_FILE, LOG_LEVEL, LOGS_DIR

# create the logs folder before logging starts
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# send logs to both the log file and the terminal
# this helps during local testing and deployment
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# create the main Flask app
app = Flask(__name__)

# store the secret key in the Flask config
app.config['SECRET_KEY'] = SECRET_KEY

# allow requests from the frontend
# this keeps the dashboard and API from running into cross origin issues
CORS(app)

# register all REST API routes with the Flask app
# fail fast here because the backend is not useful without its routes
try:
    from api.api_routes import register_api_routes
    register_api_routes(app)
    logger.info("[App] API routes registered successfully")
except Exception as e:
    logger.error(f"[App] Failed to register API routes: {e}")
    raise

# initialize the data layer before requests start
# this prepares files, caches, or shared state used by the API
try:
    from api.data_manager import DataManager
    DataManager.initialize()
    logger.info("[App] Data layer initialized successfully")
except Exception as e:
    logger.error(f"[App] Failed to initialize data layer: {e}")
    raise

# attach the Dash dashboard to the same Flask server
# the API still runs if dashboard setup fails
try:
    import dash
    import dash_bootstrap_components as dbc
    from api.dash_layout import create_layout
    from api.dash_callbacks import register_callbacks
    
    # mount Dash under /dashboard/
    dash_app = dash.Dash(
        __name__,
        server=app,
        url_base_pathname='/dashboard/',
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    # build layout and register interactive callbacks
    dash_app.layout = create_layout()
    register_callbacks(dash_app)
    logger.info("[App] Dash dashboard mounted at /dashboard/")
except Exception as e:
    logger.error(f"[App] Failed to mount Dash dashboard: {e}")
    # keep the backend alive even if the dashboard fails

# decide whether this app should run its own detection engine
# external mode means another script writes alerts and metrics into the UI data files
from api.config import EXTERNAL_DETECTION_MODE
if not EXTERNAL_DETECTION_MODE:
    try:
        from api.detection_engine import DetectionEngine

        # use the shared engine instance for the app
        engine = DetectionEngine.get_instance()

        # only start the engine if the models load correctly
        if engine.initialize_models():
            engine.start()
            logger.info("[App] Detection engine started automatically")
        else:
            logger.warning("[App] Detection engine could not start (models not loaded)")
    except Exception as e:
        logger.error(f"[App] Failed to start detection engine: {e}")
       # keep the API running even if the engine fails
else:
    logger.info("[App] External detection mode — built-in engine disabled")

# serve a simple landing page with endpoint links and quick start commands
@app.route('/')
def index():
    """Landing page with API documentation"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hybrid IDS Backend API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f8f9fa;
            }
            h1 { 
                color: #333; 
                border-bottom: 3px solid #007bff;
                padding-bottom: 10px;
            }
            .api-box {
                background: #fff;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .endpoint {
                background: #f8f9fa;
                padding: 12px;
                margin: 8px 0;
                border-left: 4px solid #007bff;
                border-radius: 3px;
            }
            .method {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 0.85em;
                margin-right: 10px;
            }
            .get { background-color: #28a745; color: white; }
            .post { background-color: #007bff; color: white; }
            a {
                color: #007bff;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            .info {
                color: #6c757d;
                font-size: 0.9em;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Hybrid IDS Backend API</h1>
        <p>RESTful API for the Hybrid AE→RF Intrusion Detection System</p>
        <p class="info">Phase 1: API Layer - Backend endpoints for metrics, alerts, thresholds, and system health</p>
        <p style="background: #d4edda; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745;">
            <strong>✓ Detection engine running:</strong> Metrics are being generated automatically. Click the links below to view live data!
        </p>

        <div class="api-box">
            <h2>API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong><a href="/api/health">/api/health</a></strong>
                <div class="info">System health check and status information</div>
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong><a href="/api/metrics">/api/metrics</a></strong>
                <div class="info">Model performance metrics (precision, recall, F1, etc.)</div>
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong><a href="/api/alerts?limit=10">/api/alerts</a></strong>
                <div class="info">Recent anomaly events with filtering and pagination</div>
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong><a href="/api/threshold">/api/threshold</a></strong>
                <div class="info">Get current threshold configuration</div>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/threshold</strong>
                <div class="info">Update AE and RF threshold values</div>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/crash</strong>
                <div class="info">Log crash flags and error events</div>
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong><a href="/api/logs?format=json&hours=24">/api/logs</a></strong>
                <div class="info">Export logs as JSON or CSV</div>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/detection/start</strong>
                <div class="info">Start the detection engine background thread</div>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/detection/stop</strong>
                <div class="info">Stop the detection engine background thread</div>
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong><a href="/api/detection/status">/api/detection/status</a></strong>
                <div class="info">Get detection engine status and statistics</div>
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong><a href="/api">/api</a></strong>
                <div class="info">API documentation in JSON format</div>
            </div>
        </div>

        <div class="api-box">
            <h2>Quick Start</h2>
            <p>The detection engine starts automatically! Just access the endpoints to view live data:</p>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto;">
# View detection engine status
curl http://localhost:5000/api/detection/status

# Get real-time metrics (precision, recall, F1, etc.)
curl http://localhost:5000/api/metrics

# Get recent attack alerts
curl "http://localhost:5000/api/alerts?limit=10"

# View system health
curl http://localhost:5000/api/health

# Update detection thresholds
curl -X POST http://localhost:5000/api/threshold \\
  -H "Content-Type: application/json" \\
  -d '{"ae_threshold": 0.025, "rf_threshold": 0.6}'
            </pre>
            <p style="margin-top: 15px; color: #6c757d; font-size: 0.9em;">
                <strong>Tip:</strong> Open the metrics and alerts endpoints in your browser and refresh to see updates!
            </p>
        </div>
    </body>
    </html>
    '''


# expose the Flask app under the name expected by run_api.py
server = app

if __name__ == '__main__':
    # log startup details so you know where the server is running
    logger.info("=" * 60)
    logger.info("Hybrid IDS Backend API - Phase 1")
    logger.info("=" * 60)
    logger.info(f"Starting server on http://{API_HOST}:{API_PORT}")
    logger.info(f"API Documentation: http://{API_HOST}:{API_PORT}/api")
    logger.info(f"Health Check: http://{API_HOST}:{API_PORT}/api/health")
    logger.info("=" * 60)

    # start the Flask development server
    app.run(debug=DEBUG_MODE, host=API_HOST, port=API_PORT, threaded=True)
