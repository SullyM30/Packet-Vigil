#api routes for the model metrics, config, and monitering 
from flask import request, jsonify, Response
from datetime import datetime
import traceback
import psutil



#register all api routes with the flask app
def register_api_routes(app):
    @app.route('/api/metrics', methods=['GET'])
    #returns current metrics
    def get_metrics():
        try:
            from api.data_manager import DataManager
            metrics = DataManager.read_metrics()
            return jsonify(metrics), 200
        except FileNotFoundError:
            return jsonify({"error": "Metrics not yet available"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/threshold', methods=['GET', 'POST'])
    def threshold():

        #GET or POST /api/threshold
        #Get or update threshold config
        try:
            from api.data_manager import DataManager

            if request.method == 'GET':
                config = DataManager.read_config()
                return jsonify({
                    "ae_threshold": config.get('ae_threshold', 0.0198),
                    "rf_threshold": config.get('rf_threshold', 0.5)
                }), 200
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            #validation
            errors = []

            if 'ae_threshold' in data:
                try:
                    ae_thresh = float(data['ae_threshold'])
                    if not (0.0 <= ae_thresh <= 1.0):
                        errors.append("ae_threshold must be between 0.0 and 1.0")
                except (ValueError, TypeError):
                    errors.append("ae_threshold must be a number")

            if 'rf_threshold' in data:
                try:
                    rf_thresh = float(data['rf_threshold'])
                    if not (0.0 <= rf_thresh <= 1.0):
                        errors.append("rf_threshold must be between 0.0 and 1.0")
                except (ValueError, TypeError):
                    errors.append("rf_threshold must be a number")

            if errors:
                return jsonify({"errors": errors}), 400

            #update config
            config = DataManager.read_config()

            if 'ae_threshold' in data:
                config['ae_threshold'] = float(data['ae_threshold'])
            if 'rf_threshold' in data:
                config['rf_threshold'] = float(data['rf_threshold'])

            config['last_modified'] = datetime.utcnow().isoformat() + 'Z'
            config['modified_by'] = 'api'

            DataManager.write_config(config)

            return jsonify({"success": True, "config": config}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/alerts', methods=['GET'])
    def get_alerts():

        #GET /api/alerts
        #returns recent anomaly events with filtering
        try:
            from api.data_manager import DataManager

            #query params
            verdict = request.args.get('verdict')
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            limit = int(request.args.get('limit', 100))
            offset = int(request.args.get('offset', 0))
            sort_by = request.args.get('sort_by', 'timestamp')
            sort_order = request.args.get('sort_order', 'desc')

            #limit and offset
            if limit < 1 or limit > 10000:
                return jsonify({"error": "limit must be between 1 and 10000"}), 400
            if offset < 0:
                return jsonify({"error": "offset must be non-negative"}), 400

            #get alerts
            alerts = DataManager.read_alerts(
                verdict=verdict,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_order=sort_order
            )

            #total count
            total_count = DataManager.count_alerts(
                verdict=verdict,
                start_date=start_date,
                end_date=end_date
            )

            return jsonify({
                "alerts": alerts,
                "total": total_count,
                "limit": limit,
                "offset": offset
            }), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/crash', methods=['POST'])
    def log_crash():

        #POST /api/crash
        #Logs crash/error events

        try:
            from api.data_manager import DataManager

            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            if 'message' not in data:
                return jsonify({"error": "message field required"}), 400

            DataManager.append_crash_log(
                level=data.get('level', 'ERROR'),
                message=data['message'],
                traceback=data.get('traceback')
            )

            return jsonify({"success": True}), 201

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/health', methods=['GET'])
    def health_check():

        #GET /api/health
        #System health check backend 

        try:
            from api.data_manager import DataManager
            from datetime import datetime

            alerts_exists = DataManager.ALERTS_FILE.exists()
            metrics_exists = DataManager.METRICS_FILE.exists()
            config_exists = DataManager.CONFIG_FILE.exists()
            crash_log_exists = DataManager.CRASH_LOG.exists()

            #overall status
            status = "healthy"
            if not config_exists:
                status = "warning"  

            health = {
                "status": status,
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "files": {
                    "alerts_file_exists": alerts_exists,
                    "metrics_file_exists": metrics_exists,
                    "config_file_exists": config_exists,
                    "crash_log_exists": crash_log_exists
                },
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "memory_percent": psutil.virtual_memory().percent,
                "errors_last_hour": DataManager.count_recent_crashes(hours=1)
            }

            status_code = 200 if status == "healthy" else 200  
            return jsonify(health), status_code

        except Exception as e:
            return jsonify({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + 'Z'
            }), 500

    @app.route('/api/logs', methods=['GET'])
    def export_logs():

        #GET /api/logs
        #Export run logs as JSON or CSV

        try:
            from api.data_manager import DataManager

            format_type = request.args.get('format', 'json')
            hours = int(request.args.get('hours', 24))

            if hours < 1 or hours > 168: 
                return jsonify({"error": "hours must be between 1 and 168"}), 400

            if format_type == 'json':
                logs = DataManager.read_logs_json(hours=hours)
                return jsonify(logs), 200
            elif format_type == 'csv':
                csv_data = DataManager.read_logs_csv(hours=hours)
                return Response(
                    csv_data,
                    mimetype='text/csv',
                    headers={"Content-Disposition": f"attachment;filename=logs_{hours}h.csv"}
                )
            else:
                return jsonify({"error": "Invalid format. Use 'json' or 'csv'"}), 400

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/detection/start', methods=['POST'])
    def start_detection():

        #POST /api/detection/start
        #Start the detection engine background thread

        try:
            from api.detection_engine import DetectionEngine

            engine = DetectionEngine.get_instance()

            if engine.is_running():
                return jsonify({
                    "success": False,
                    "message": "Detection engine is already running"
                }), 400

            success = engine.start()

            if success:
                return jsonify({
                    "success": True,
                    "message": "Detection engine started successfully"
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "message": "Failed to start detection engine (models may not be loaded)"
                }), 500

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500

    @app.route('/api/detection/stop', methods=['POST'])
    def stop_detection():
        """
        POST /api/detection/stop
        Stop the detection engine background thread
        """
        try:
            from api.detection_engine import DetectionEngine

            engine = DetectionEngine.get_instance()

            if not engine.is_running():
                return jsonify({
                    "success": False,
                    "message": "Detection engine is not running"
                }), 400

            engine.stop()

            return jsonify({
                "success": True,
                "message": "Detection engine stopped successfully"
            }), 200

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    @app.route('/api/detection/status', methods=['GET'])
    def detection_status():

        #GET /api/detection/status
        #Get detection engine status

        try:
            from api.detection_engine import DetectionEngine

            engine = DetectionEngine.get_instance()

            return jsonify({
                "running": engine.is_running(),
                "models_loaded": engine.models_loaded,
                "uptime_seconds": engine.get_uptime(),
                "total_processed": engine.total_processed,
                "total_alerts": engine.total_alerts,
                "alert_rate_per_second": engine.get_alert_rate(),
                "last_detection_time": engine.last_detection_time
            }), 200

        except Exception as e:
            return jsonify({
                "error": str(e)
            }), 500

    @app.route('/api', methods=['GET'])
    @app.route('/api/', methods=['GET'])
    def api_root():
        #documentation 
        return jsonify({
            "message": "Hybrid Anomaly Detection System API",
            "version": "1.0",
            "endpoints": {
                "GET /api/metrics": "Get model performance metrics",
                "POST /api/threshold": "Update AE and RF thresholds",
                "GET /api/alerts": "Get recent alerts with filtering",
                "POST /api/crash": "Log crash/error events",
                "GET /api/health": "System health check",
                "GET /api/logs": "Export logs as JSON or CSV",
                "POST /api/detection/start": "Start the detection engine",
                "POST /api/detection/stop": "Stop the detection engine",
                "GET /api/detection/status": "Get detection engine status"
            }
        }), 200
