import dash
from dash import dcc, html, dash_table
import plotly.graph_objs as go
import sqlite3
import pandas as pd
from dash.dependencies import Input, Output

DB_PATH = r"C:\Users\Lukasz\Desktop\stock_project\data\stocks.db"

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

app = dash.Dash(__name__)
app.title = "Stock Prediction Dashboard"


# ---------------------------
# LAYOUT
# ---------------------------

app.layout = html.Div([
    html.H1("📈 Dashboard — Przewidywanie cen akcji", style={"textAlign": "center"}),

    html.Div([
        html.Label("Wybierz model:"),
        dcc.Dropdown(
            id="model-select",
            options=[
                {"label": "Random Forest (RF)", "value": "rf"},
                {"label": "LSTM", "value": "lstm"},
                {"label": "Oba modele", "value": "both"}
            ],
            value="both",
            clearable=False
        )
    ], style={"width": "25%", "display": "inline-block"}),

    html.Div([
        html.Label("Zakres dat:"),
        dcc.DatePickerRange(
            id="date-range",
            start_date=df["date"].min(),
            end_date=df["date"].max(),
            min_date_allowed=df["date"].min(),
            max_date_allowed=df["date"].max()
        )
    ], style={"width": "40%", "display": "inline-block", "marginLeft": "50px"}),

    html.Br(),
    html.H2("📊 Wykres predykcji"),
    dcc.Graph(id="prediction-chart"),

    html.Br(),
    html.H2("📋 Tabela danych"),
    dash_table.DataTable(
        id="data-table",
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left"},
    )
])


# ---------------------------
# CALLBACK — aktualizacja wykresu + tabeli
# ---------------------------

@app.callback(
    [Output("prediction-chart", "figure"),
     Output("data-table", "data")],
    [Input("model-select", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date")]
)
def update_dashboard(selected_model, start_date, end_date):
    filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    fig = go.Figure()

    if selected_model in ("rf", "both"):
        fig.add_trace(go.Scatter(
            x=filtered["date"],
            y=filtered["rf"],
            mode="lines+markers",
            name="Random Forest"
        ))

    if selected_model in ("lstm", "both"):
        fig.add_trace(go.Scatter(
            x=filtered["date"],
            y=filtered["lstm"],
            mode="lines+markers",
            name="LSTM"
        ))

    fig.update_layout(
        title="Predykcje cen w czasie",
        xaxis_title="Data",
        yaxis_title="Cena",
        legend_title="Model",
        template="plotly_white"
    )

    return fig, filtered.to_dict("records")


# ---------------------------
# START SERVER
# ---------------------------

if __name__ == "__main__":
    app.run(debug=True)

