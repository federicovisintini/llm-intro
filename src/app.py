import re
import dash_bootstrap_components as dbc
from dash import Dash, html, Output, Input

from src.llm import chain

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


@app.callback([Output("output_user", "children"), Output("internal_thoughts", "children")], [Input("input", "value")])
def output_text(value):
    if not value:
        return "", ""

    print("Calling API")
    result = chain.run(value)

    output_user = extract_user_output(result)
    internal_thoughts = result

    return output_user, internal_thoughts


def extract_user_output(text):
    """ Defined as the text inside "" """
    m = re.search(r"///.*", text)
    if not m:
        return ""
    return m.group().replace('///', '').strip()


if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1')
