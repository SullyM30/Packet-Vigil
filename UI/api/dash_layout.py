#dashboard layout for 3 tabs, Dashboard, Anomaly Inspector, and Settings 
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


def create_layout():
    #creaste main dashboard with layers 
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Hybrid Anomaly Detection System",
                            className="text-primary fw-bold"), width=12)
        ], className="mb-3 mt-3"),

        dcc.Tabs(id="tabs", value='dashboard-tab', children=[
            dcc.Tab(label='Dashboard',         value='dashboard-tab'),
            dcc.Tab(label='Anomaly Inspector', value='inspector-tab'),
            dcc.Tab(label='Settings',          value='settings-tab'),
        ]),

        html.Div(id='tabs-content', className="mt-4"),

        #autorefresh every 5 seconds 
        dcc.Interval(id='interval-component', interval=5 * 1000, n_intervals=0),
        dcc.Store(id='config-store'),
    ], fluid=True)


#dashboard tab

def create_dashboard_tab():
    #live metrics dashboard, data fed from metrics.json through ui bridge
    return dbc.Container([

        dbc.Row([
            #system status
            dbc.Col(dbc.Card([
                dbc.CardHeader("System Status"),
                dbc.CardBody([
                    html.Div(id='system-status-badge'),
                    html.Small(id='system-uptime', className="text-muted mt-2 d-block"),
                ])
            ]), width=3),

            #performance metrics
            dbc.Col(dbc.Card([
                dbc.CardHeader("Performance Metrics"),
                dbc.CardBody([
                    html.Div([html.Span("Precision: "), html.Strong(id='precision-value')]),
                    html.Div([html.Span("Recall: "),    html.Strong(id='recall-value')]),
                    html.Div([html.Span("F1 Score: "),  html.Strong(id='f1-value')]),
                ])
            ]), width=3),

            #alerts
            dbc.Col(dbc.Card([
                dbc.CardHeader("Alerts Detected"),
                dbc.CardBody([
                    html.H2(id='total-alerts-count', className="text-danger fw-bold"),
                    html.Small(id='alert-rate-status', className="text-muted"),
                ])
            ]), width=3),

            #resource usage 
            dbc.Col(dbc.Card([
                dbc.CardHeader("Resource Usage (ML process)"),
                dbc.CardBody([
                    html.Div([html.Span("CPU: "),    html.Strong(id='cpu-usage')]),
                    html.Div([html.Span("Memory: "), html.Strong(id='memory-usage')]),
                ])
            ]), width=3),
        ], className="mb-4"),

        dbc.Row([
            #thresholds
            dbc.Col(dbc.Card([
                dbc.CardHeader("Active Thresholds"),
                dbc.CardBody([
                    html.Div([html.Span("AE Threshold: "), html.Strong(id='current-ae-threshold')]),
                    html.Div([html.Span("RF Threshold: "), html.Strong(id='current-rf-threshold')]),
                ])
            ]), width=6),

            #alert rate 
            dbc.Col(dbc.Card([
                dbc.CardHeader("Alert Rate"),
                dbc.CardBody([
                    html.Div(id='alert-rate-indicator'),
                ])
            ]), width=6),
        ]),

    ], fluid=True)


#anomaly inspector tab

def create_inspector_tab():
    #alert table 
    return dbc.Container([

        #filter row
        dbc.Row([
            dbc.Col([
                html.Label("Verdict", className="fw-semibold"),
                dcc.Dropdown(
                    id='verdict-filter',
                    options=[
                        {'label': 'All',    'value': 'all'},
                        {'label': 'Attack', 'value': 'attack'},
                        {'label': 'Benign', 'value': 'benign'},
                    ],
                    value='all',
                    clearable=False,
                )
            ], width=2),

            dbc.Col([
                html.Label("Date Range", className="fw-semibold"),
                dcc.DatePickerRange(
                    id='date-range-picker',
                    start_date_placeholder_text="Start",
                    end_date_placeholder_text="End",
                )
            ], width=5),

            dbc.Col([
                html.Label(html.Br()),
                dbc.Button("Apply Filters", id='apply-filters-btn',
                           color="primary", className="w-100"),
            ], width=2),

            dbc.Col([
                html.Label(html.Br()),
                html.Small(id='alerts-total-label', className="text-muted d-block pt-2"),
            ], width=3),
        ], className="mb-3 align-items-end"),

        #alerts table
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='alerts-table',
                    columns=[
                        {'name': 'Time',         'id': 'timestamp'},
                        {'name': 'Flow ID',       'id': 'flow_id'},
                        {'name': 'Src IP',        'id': 'src_ip'},
                        {'name': 'Src Port',      'id': 'src_port'},
                        {'name': 'Dst IP',        'id': 'dst_ip'},
                        {'name': 'Dst Port',      'id': 'dst_port'},
                        {'name': 'AE Score',      'id': 'ae_score',
                         'type': 'numeric', 'format': {'specifier': '.6f'}},
                        {'name': 'RF Prob',       'id': 'rf_prob',
                         'type': 'numeric', 'format': {'specifier': '.4f'}},
                        {'name': 'Verdict',       'id': 'final_verdict'},
                    ],
                    data=[],

                    page_size=50,
                    page_action='custom',
                    page_current=0,
                    page_count=0,

                    sort_action='custom',
                    sort_mode='single',
                    sort_by=[],

                    #row color 
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{final_verdict} = "attack"'},
                            'backgroundColor': '#fff0f0',
                            'color': '#cc0000',
                            'fontWeight': 'bold',
                        }
                    ],

                    #header style
                    style_cell={
                        'textAlign': 'left',
                        'padding': '8px 12px',
                        'fontSize': '13px',
                        'fontFamily': 'monospace',
                        'whiteSpace': 'normal',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                    },
                    style_header={
                        'backgroundColor': '#2c3e50',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'fontSize': '13px',
                        'padding': '10px 12px',
                    },

                    style_cell_conditional=[
                        {'if': {'column_id': 'timestamp'},    'width': '145px', 'minWidth': '145px'},
                        {'if': {'column_id': 'flow_id'},      'width': '90px',  'minWidth': '90px'},
                        {'if': {'column_id': 'src_ip'},       'width': '120px', 'minWidth': '120px'},
                        {'if': {'column_id': 'src_port'},     'width': '75px',  'minWidth': '75px'},
                        {'if': {'column_id': 'dst_ip'},       'width': '120px', 'minWidth': '120px'},
                        {'if': {'column_id': 'dst_port'},     'width': '75px',  'minWidth': '75px'},
                        {'if': {'column_id': 'ae_score'},     'width': '95px',  'minWidth': '95px'},
                        {'if': {'column_id': 'rf_prob'},      'width': '80px',  'minWidth': '80px'},
                        {'if': {'column_id': 'final_verdict'},'width': '80px',  'minWidth': '80px'},
                    ],

                    style_table={'overflowX': 'auto', 'borderRadius': '4px'},
                    style_as_list_view=True,
                )
            ], width=12)
        ])

    ], fluid=True)


#settings tab

def create_settings_tab():
    #threshold config 
    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Threshold Configuration"),
                dbc.CardBody([
                    html.Label("Autoencoder Reconstruction Error Threshold:",
                               className="fw-semibold"),
                    dcc.Slider(
                        id='ae-threshold-slider',
                        min=0.0, max=0.1, step=0.001, value=0.0198,
                        marks={i / 100: f'{i / 100:.2f}' for i in range(0, 11, 2)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Br(),

                    html.Label("Random Forest Probability Cutoff:",
                               className="fw-semibold"),
                    dcc.Slider(
                        id='rf-threshold-slider',
                        min=0.0, max=1.0, step=0.05, value=0.5,
                        marks={i / 10: f'{i / 10:.1f}' for i in range(0, 11, 2)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Br(),

                    dbc.Row([
                        dbc.Col(dbc.Button("Apply Thresholds", id='apply-threshold-btn',
                                           color="success", className="w-100"), width=6),
                        dbc.Col(dbc.Button("Restart Detection", id='restart-detection-btn',
                                           color="warning", className="w-100"), width=6),
                    ]),

                    html.Div(id='threshold-update-message', className="mt-3"),
                ])
            ]), width=8)
        ])
    ], fluid=True)





