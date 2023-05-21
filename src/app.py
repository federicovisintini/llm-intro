import dash_bootstrap_components as dbc
from dash import Dash, html, Output, Input

from src.llm import llm

app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

app.layout = dbc.Container([
    html.Br(),
    dbc.Row(
        html.H1("Singularity chatbot"), id='title_row'
    ),
    html.Br(),
    dbc.Row([
        html.H5("Insert you input here:"),
        dbc.Input(id="input", placeholder="Type something...", type="text", debounce=True),
    ], id='input_row'),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H5("This is your output:"),
            html.P(id="output_user"),
        ], width=6, id="output_user_col"),
        dbc.Col([
            html.H5("These are the LLM thoughts:"),
            html.P(id="internal_thoughts"),
        ], id="internal_thoughts_col"),
    ], id='output_row'),
])


@app.callback(Output("output_user", "children"), [Input("input", "value")])
def output_text(value):
    if not value:
        return ""
    return llm(value)


@app.callback(Output("internal_thoughts", "children"), [Input("input", "value")])
def thoughts_text(value):
    if not value:
        return ""
    return llm(value + " Think out loud")


if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1')
