#data manegement for the alerts metrics config and logs

import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os

#I/O operations for alerts, metrics and config 
class DataManager:
    #file lock for safety
    _alerts_lock = threading.Lock()
    _metrics_lock = threading.Lock()
    _config_lock = threading.Lock()
    _crash_lock = threading.Lock()


    BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    DATA_DIR = BASE_DIR / "data"
    ALERTS_FILE = DATA_DIR / "alerts.json"
    METRICS_FILE = DATA_DIR / "metrics.json"
    CONFIG_FILE = DATA_DIR / "config.json"
    CRASH_LOG = DATA_DIR / "crash.log"

    @classmethod
    def initialize(cls):
        #data directory and initialize files 
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)

        if not cls.CONFIG_FILE.exists():
            default_config = {
                'ae_threshold': 0.0198,
                'rf_threshold': 0.5,
                'detection_enabled': True,
                'batch_size': 1000,
                'detection_interval_seconds': 1.0,
                'alert_retention_days': 7,
                'auto_refresh_interval': 5,
                'max_alerts_display': 1000,
                'last_modified': datetime.utcnow().isoformat() + 'Z',
                'modified_by': 'system'
            }
            cls.write_config(default_config)

        #reset for clean dash
        cls.reset_session()

        if not cls.CRASH_LOG.exists():
            cls.CRASH_LOG.touch()

    @classmethod
    def reset_session(cls):

        #clear alerts and resent the metrics 
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)

        #empty the alerts loh
        with cls._alerts_lock:
            cls.ALERTS_FILE.write_text('', encoding='utf-8')

        #clean idle metrics state
        idle_metrics = {
            'last_updated': datetime.utcnow().isoformat() + 'Z',
            'status': 'idle',
            'evaluation_mode': 'real_time',
            'total_flows_processed': 0,
            'total_alerts': 0,
            'ae_flagged': 0,
            'alert_rate_per_second': 0.0,
            'hybrid_metrics': None,
            'resource_usage': {'cpu_percent': 0.0, 'memory_mb': 0.0, 'memory_percent': 0.0},
        }
        cls.write_metrics(idle_metrics)

    @classmethod
    def append_alerts(cls, alerts: List[Dict]):
        with cls._alerts_lock:
            with open(cls.ALERTS_FILE, 'a', encoding='utf-8') as f:
                for alert in alerts:
                    f.write(json.dumps(alert) + '\n')
                f.flush()
                os.fsync(f.fileno())

    @classmethod
    def read_alerts(cls,
                    verdict: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    limit: int = 100,
                    offset: int = 0,
                    sort_by: str = 'timestamp',
                    sort_order: str = 'desc') -> List[Dict]:
        #read alerts with filtering 
        with cls._alerts_lock:
            if not cls.ALERTS_FILE.exists():
                return []

            #read all alerts 
            alerts = []
            with open(cls.ALERTS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            alerts.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            #filter by verdict 
            if verdict:
                alerts = [a for a in alerts if a.get('final_verdict') == verdict]

            #filter by dte range 
            if start_date:
                alerts = [a for a in alerts if a.get('timestamp', '') >= start_date]
            if end_date:
                alerts = [a for a in alerts if a.get('timestamp', '') <= end_date]

            #sort
            reverse = (sort_order == 'desc')
            try:
                alerts.sort(key=lambda x: x.get(sort_by, ''), reverse=reverse)
            except (TypeError, KeyError):
                pass

            return alerts[offset:offset + limit]

    @classmethod
    def count_alerts(cls,
                     verdict: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> int:
        #count alerts matching filter 
        with cls._alerts_lock:
            if not cls.ALERTS_FILE.exists():
                return 0

            count = 0
            with open(cls.ALERTS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            alert = json.loads(line)

                            if verdict and alert.get('final_verdict') != verdict:
                                continue
                            if start_date and alert.get('timestamp', '') < start_date:
                                continue
                            if end_date and alert.get('timestamp', '') > end_date:
                                continue

                            count += 1
                        except json.JSONDecodeError:
                            continue

            return count

    @classmethod
    def read_config(cls) -> Dict:
        with cls._config_lock:
            if not cls.CONFIG_FILE.exists():
                cls.initialize()
            with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)

    @classmethod
    def write_config(cls, config: Dict):
        #write configg with temp file and rename 
        with cls._config_lock:
            temp_file = cls.CONFIG_FILE.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            temp_file.replace(cls.CONFIG_FILE)

    @classmethod
    def read_metrics(cls) -> Dict:
        #read metrics 
        with cls._metrics_lock:
            if not cls.METRICS_FILE.exists():
                cls.initialize()
            with open(cls.METRICS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)

    @classmethod
    def write_metrics(cls, metrics: Dict):
        with cls._metrics_lock:
            temp_file = cls.METRICS_FILE.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            temp_file.replace(cls.METRICS_FILE)

    @classmethod
    def append_crash_log(cls, level: str, message: str, traceback: Optional[str] = None):
        #add to crash log
        with cls._crash_lock:
            with open(cls.CRASH_LOG, 'a', encoding='utf-8') as f:
                timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                f.write(f"[{timestamp}] {level}: {message}\n")
                if traceback:
                    f.write(f"{traceback}\n")

    @classmethod
    def count_recent_crashes(cls, hours: int = 1) -> int:
        #count crashes in last n hours 
        if not cls.CRASH_LOG.exists():
            return 0

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        count = 0

        with cls._crash_lock:
            with open(cls.CRASH_LOG, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('['):
                        try:
                            timestamp_str = line[1:24]
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                            if timestamp >= cutoff and 'ERROR' in line:
                                count += 1
                        except (ValueError, IndexError):
                            pass

        return count

    @classmethod
    def read_logs_json(cls, hours: int = 24) -> List[Dict]:
        #read as json list 
        if not cls.CRASH_LOG.exists():
            return []

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        logs = []

        with cls._crash_lock:
            with open(cls.CRASH_LOG, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('['):
                        try:
                            timestamp_str = line[1:24]
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')

                            if timestamp >= cutoff:
                                #parse log line
                                rest = line[26:]
                                if ':' in rest:
                                    level, message = rest.split(':', 1)
                                    logs.append({
                                        'timestamp': timestamp.isoformat() + 'Z',
                                        'level': level.strip(),
                                        'message': message.strip()
                                    })
                        except (ValueError, IndexError):
                            pass

        return logs

    @classmethod
    def read_logs_csv(cls, hours: int = 24) -> str:
        #read crash log as csv list 
        logs = cls.read_logs_json(hours)

        csv_lines = ['timestamp,level,message']
        for log in logs:
            csv_lines.append(f"{log['timestamp']},{log['level']},{log['message']}")

        return '\n'.join(csv_lines)

