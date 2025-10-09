import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
import plotly.graph_objs as go

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Robust Regression Playground"),
    html.Label("Number of Outliers:"),
    dcc.Slider(id='outliers', min=0, max=10, step=1, value=3),
    dcc.Graph(id='regression-plot')
])

@app.callback(
    Output('regression-plot', 'figure'),
    Input('outliers', 'value'))
def update_plot(n_outliers):
    # Data
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 2 * X + 1 + np.random.normal(0, 1, size=X.shape)

    if n_outliers > 0:
        X_outliers = np.random.uniform(0, 10, n_outliers)
        y_outliers = np.random.uniform(-30, 30, n_outliers)
        X_all = np.concatenate([X, X_outliers])
        y_all = np.concatenate([y, y_outliers])
    else:
        X_all = X
        y_all = y

    X_all = X_all.reshape(-1, 1)

    # Fit models
    ols = LinearRegression().fit(X_all, y_all)
    huber = HuberRegressor().fit(X_all, y_all)
    x_pred = np.linspace(0, 10, 100).reshape(-1, 1)

    # Predictions
    y_ols = ols.predict(x_pred)
    y_huber = huber.predict(x_pred)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_all.flatten(), y=y_all, mode='markers', name='Data'))
    fig.add_trace(go.Scatter(x=x_pred.flatten(), y=y_ols, mode='lines', name='OLS'))
    fig.add_trace(go.Scatter(x=x_pred.flatten(), y=y_huber, mode='lines', name='Huber'))

    fig.update_layout(title="OLS vs Huber Regression",
                      xaxis_title="X",
                      yaxis_title="y")
    return fig

server = app.server

if __name__ == '__main__':
    app.run(debug=True)