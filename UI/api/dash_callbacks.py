"""
Dash callback functions for interactive dashboard components.
"""

from dash import Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc
from datetime import datetime


def register_callbacks(app):
    """Register all Dash callbacks"""

    from api.dash_layout import create_dashboard_tab, create_inspector_tab, create_settings_tab
    from api.data_manager import DataManager

    # switch visible tab content when user clicks a tab
    @app.callback(
        Output('tabs-content', 'children'),
        Input('tabs', 'value')
    )
    def render_tab_content(active_tab):

        # return layout based on selected tab id
        """Switch tab content based on selected tab"""
        if active_tab == 'dashboard-tab':
            return create_dashboard_tab()
        elif active_tab == 'inspector-tab':
            return create_inspector_tab()
        elif active_tab == 'settings-tab':
            return create_settings_tab()
        return html.Div("Invalid tab")


    # update dashboard metrics every interval tick
    # interval component fires every few seconds
    @app.callback(
        [Output('precision-value', 'children'),
         Output('recall-value', 'children'),
         Output('f1-value', 'children'),
         Output('total-alerts-count', 'children'),
         Output('cpu-usage', 'children'),
         Output('memory-usage', 'children'),
         Output('system-status-badge', 'children'),
         Output('current-ae-threshold', 'children'),
         Output('current-rf-threshold', 'children'),
         Output('alert-rate-status', 'children'),
         Output('system-uptime', 'children'),
         Output('alert-rate-indicator', 'children')],
        Input('interval-component', 'n_intervals')
    )
    def update_dashboard(n):
        """Auto-refresh dashboard metrics every 5 seconds from metrics.json"""

        # read latest metrics and config from disk
        # data manager handles file access and formatting
        try:
            metrics = DataManager.read_metrics()
            config  = DataManager.read_config()

            # hybrid metrics only exist if labels are available
            hybrid    = metrics.get('hybrid_metrics') or {}
            resources = metrics.get('resource_usage') or {}

            # system status drives badge color
            status = metrics.get('status', 'idle')
            status_color = {
                'running':  'success',
                'starting': 'warning',
                'stopped':  'secondary',
                'idle':     'secondary',
            }.get(status, 'secondary')
            status_badge = dbc.Badge(status.title(), color=status_color, className="fs-6")

            # show how many packets processed instead of uptime
            processed = metrics.get('total_flows_processed', 0)
            ae_flagged = metrics.get('ae_flagged', 0)
            uptime_str = f"Processed: {processed:,} pkts | AE flagged: {ae_flagged:,}"

            # only show precision, recall, f1 if ground truth exists
            if hybrid:
                prec_str = f"{hybrid.get('precision', 0):.4f}"
                rec_str  = f"{hybrid.get('recall', 0):.4f}"
                f1_str   = f"{hybrid.get('f1', 0):.4f}"
            else:
                prec_str = rec_str = f1_str = "— (no labels)"

            # total alert count
            total_alerts = metrics.get('total_alerts', 0)

            # system resource usage
            cpu_str = f"{resources.get('cpu_percent', 0):.1f}%"
            mem_str = f"{resources.get('memory_mb', 0):.0f} MB"

            # current thresholds used by detection system
            ae_thresh = config.get('ae_threshold', 0.0198)
            rf_thresh = config.get('rf_threshold', 0.5)

            # alert rate determines system health level
            alert_rate = metrics.get('alert_rate_per_second', 0.0)
            if alert_rate < 1.0:
                rate_status = f"{alert_rate:.2f}/s  (Normal)"
                rate_color  = "success"
            elif alert_rate < 5.0:
                rate_status = f"{alert_rate:.2f}/s  (Elevated)"
                rate_color  = "warning"
            else:
                rate_status = f"{alert_rate:.2f}/s  (Critical)"
                rate_color  = "danger"

            rate_indicator = dbc.Badge(rate_status, color=rate_color, className="fs-6")

             # return values mapped to each UI output
            return (
                prec_str, rec_str, f1_str,
                f"{total_alerts:,}",
                cpu_str, mem_str,
                status_badge,
                f"{ae_thresh:.4f}", f"{rf_thresh:.2f}",
                rate_status,
                uptime_str,
                rate_indicator,
            )

        # fallback if metrics not ready yet
        except Exception:
            waiting = dbc.Badge("Waiting", color="secondary", className="fs-6")
            return (
                "—", "—", "—",
                "0",
                "—", "—",
                waiting,
                "—", "—",
                "Waiting for detection run…",
                "No run yet",
                dbc.Badge("—", color="secondary"),
            )

    # update alerts table with filters, sorting, and order
    @app.callback(
        [Output('alerts-table', 'data'),
         Output('alerts-table', 'page_count'),
         Output('alerts-total-label', 'children')],
        [Input('apply-filters-btn', 'n_clicks'),
         Input('alerts-table', 'page_current'),
         Input('alerts-table', 'sort_by'),
         Input('interval-component', 'n_intervals')],
        [State('verdict-filter', 'value'),
         State('date-range-picker', 'start_date'),
         State('date-range-picker', 'end_date')]
    )
    def update_alerts_table(n_clicks, page_current, sort_by, n_intervals,
                            verdict, start_date, end_date):
        """Update alerts table with filtering, sorting, pagination"""
        try:
            #page setup
            limit  = 50
            offset = (page_current or 0) * limit

            # sorting config
            sort_column = sort_by[0]['column_id'] if sort_by else 'timestamp'
            sort_order  = 'asc' if sort_by and sort_by[0]['direction'] == 'asc' else 'desc'

            # apply verdict filter
            verdict_filter = None if verdict == 'all' else verdict

            # fetch filtered alerts
            alerts = DataManager.read_alerts(
                verdict=verdict_filter,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset,
                sort_by=sort_column,
                sort_order=sort_order,
            )

            #total count for pagination 
            total_count = DataManager.count_alerts(
                verdict=verdict_filter,
                start_date=start_date,
                end_date=end_date,
            )
            page_count = max(1, (total_count + limit - 1) // limit)

            # format timestamps for readability: "2026-03-01 12:34:56"
            for alert in alerts:
                ts = alert.get('timestamp', '')
                if ts:
                    alert['timestamp'] = ts[:19].replace('T', ' ')

            # label showing current page range
            label = f"Showing {offset + 1}–{min(offset + limit, total_count)} of {total_count:,} alerts"
            if total_count == 0:
                label = "No alerts yet — run the detector to populate this table."

            return alerts, page_count, label

        except Exception:
            return [], 1, "Error loading alerts."

    # apply new threshold values from settings tab
    @app.callback(
        Output('threshold-update-message', 'children'),
        Input('apply-threshold-btn', 'n_clicks'),
        [State('ae-threshold-slider', 'value'),
         State('rf-threshold-slider', 'value')],
        prevent_initial_call=True
    )
    def apply_threshold_update(n_clicks, ae_thresh, rf_thresh):
        """Apply threshold updates when button is clicked"""
        if not n_clicks:
            raise PreventUpdate

        try:
            # load current config and update values
            config = DataManager.read_config()
            config['ae_threshold']  = ae_thresh
            config['rf_threshold']  = rf_thresh
            config['last_modified'] = datetime.utcnow().isoformat() + 'Z'
            config['modified_by']   = 'dashboard'
            # save updated config
            DataManager.write_config(config)

            return dbc.Alert(
                f"Thresholds updated — AE: {ae_thresh:.4f}  RF: {rf_thresh:.2f}",
                color="success", dismissable=True, duration=4000
            )
        except Exception as e:
            return dbc.Alert(
                f"Error updating thresholds: {e}",
                color="danger", dismissable=True, duration=4000
            )
        
    # load current thresholds when settings tab opens
    @app.callback(
        [Output('ae-threshold-slider', 'value'),
         Output('rf-threshold-slider', 'value')],
        Input('tabs', 'value')
    )
    def load_current_thresholds(active_tab):
        """Load current thresholds into sliders when Settings tab is opened"""
        if active_tab == 'settings-tab':
            try:
                config = DataManager.read_config()
                return config.get('ae_threshold', 0.0198), config.get('rf_threshold', 0.5)
            except Exception:
                return 0.0198, 0.5
        raise PreventUpdate

    # show message when user clicks restart button
    # actual restart must be done manually outside dashboard
    @app.callback(
        Output('threshold-update-message', 'children', allow_duplicate=True),
        Input('restart-detection-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def restart_detection(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return dbc.Alert(
            "To restart detection, re-run replay_detector.py or live_detector_linux.py.",
            color="info", dismissable=True, duration=6000
        )
