import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any
from src.config_utils import load_config

class AttritionDashboard:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the dashboard."""
        self.config = load_config(config_path)
        self.app = dash.Dash(__name__)
        self.predictions = None
        
    def update_data(self, predictions: pd.DataFrame):
        """Update the dashboard with new predictions data."""
        self.predictions = predictions
        
    def create_dashboard_layout(self):
        """Create the dashboard layout."""
        if self.predictions is None:
            self.app.layout = html.Div([
                html.H1("Attrition Intelligence Dashboard"),
                html.P("No data available. Please run the prediction model first.")
            ])
            return
            
        self.app.layout = html.Div([
            html.H1("Attrition Intelligence Dashboard"),
            
            # Risk Distribution
            html.Div([
                html.H2("Attrition Risk Distribution"),
                dcc.Graph(
                    figure=px.histogram(
                        self.predictions,
                        x='attrition_probability',
                        nbins=20,
                        title='Distribution of Attrition Risk Scores'
                    )
                )
            ]),
            
            # High Risk Employees
            html.Div([
                html.H2("High Risk Employees"),
                dash_table.DataTable(
                    data=self.predictions[self.predictions['high_risk'] == 1].to_dict('records'),
                    columns=[
                        {'name': 'Employee ID', 'id': 'employee_id'},
                        {'name': 'Risk Score', 'id': 'attrition_probability'}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px'
                    }
                )
            ])
        ])
        
    def run_server(self):
        """Run the dashboard server."""
        self.create_dashboard_layout()
        print(f"\nStarting dashboard at http://localhost:{self.config['dashboard']['port']}")
        self.app.run(
            host='localhost',
            port=self.config['dashboard']['port'],
            debug=self.config['dashboard']['debug']
        ) 