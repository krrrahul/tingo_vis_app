from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output
import plotly.graph_objs as go
import datetime
import pandas as pd
from config import *

import plotly.graph_objects as go
import pandas as pd



external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']

app = dash.Dash('stock-data',external_scripts=external_js,external_stylesheets=external_css)
color_code = {'goog':['rgb(255, 0, 0)','red'],'aapl':['rgb(0,0,255)','blue'],'amzn':['rgb(0,100,0)','darkgreen'],'fb':['rgb(255, 255, 0)','yellow'],'twtr':['rgb(255,192,203)','pink']}

features = ['goog','aapl','amzn','fb','twtr']
start = pd.to_datetime('15-06-2016').date()
end = pd.to_datetime('15-06-2021').date()
df_temp = pd.read_csv('./data/stock.csv')


def plot_graph(df_temp=df_temp, company_list=features, start=start, end=end, col="high"):
    fig = go.Figure()
    data = []
    df_temp = df_temp.loc[(df_temp['date'] >= start) & (df_temp['date'] <= end)]
    for idx, comp in enumerate(company_list):
        df = df_temp.loc[df_temp['symbol'] == comp]
        #print(color_code[comp])
        mean = df_temp[col].loc[df_temp['symbol'] == comp].mean()
        fig.add_trace(go.Scatter(
            name=comp,
            opacity=0.5,
            marker_color=color_code[comp][0],
            mode="markers+lines", x=df["date"], y=df[col],
            marker_symbol="star",
        ))
        for i in range(len(df)):
            if i == 0 or i == len(df):
                fig.add_shape(type="line", x0=df['date'].iloc[i], y0=mean, x1=df['date'].iloc[i], y1=df[col].iloc[i],
                              line=dict(
                                  color=color_code[comp][1],
                                  width=2
                              ),
                              opacity=0.5,
                              fillcolor=color_code[comp][0])
            else:
                fig.add_trace(go.Scatter(
                    x=[df["date"].iloc[i - 1], df["date"].iloc[i - 1], df["date"].iloc[i], df["date"].iloc[i]],
                    y=[mean, df[col].iloc[i - 1], df[col].iloc[i], mean],
                    mode='lines',
                    opacity=0.5,
                    marker_color=color_code[comp][0],
                    fill="toself",
                    fillcolor=color_code[comp][0]
                ))
    fig.update_shapes(opacity=0.5)
    fig.update_xaxes(showgrid=True, ticklabelmode="period")
    fig.update_layout(width=2000,height=700,)
    return fig

app.layout = html.Div([
    html.Div([
        html.H4('Stock Data [Google,Facebook,Twitter,Amazon,Apple] between 16-05-2016 to 16-05-2021',style={'float': 'left',}),
        ]),
    dcc.Dropdown(id='weather-data-name',
                 options=[{'label': s, 'value': s}
                          for s in features],
                 value=features,
                 multi=True
                 ),
    dcc.DatePickerRange(
        id='date-picker-range',
        min_date_allowed=pd.to_datetime('15-05-2016').date(),
        max_date_allowed=pd.to_datetime('15-06-2021').date(),
        initial_visible_month=pd.to_datetime('15-05-2016').date(),
        start_date  = pd.to_datetime('15-05-2016').date(),
        end_date=pd.to_datetime('15-06-2021').date()
    ),
    dcc.Graph(id='graphs', className='row'),
    ], className="container",style={'width':'98%','margin-left':10,'margin-right':10,'max-width':5000})


@app.callback(
    Output('graphs','figure'),
    [Input('weather-data-name', 'value'),Input('date-picker-range', 'start_date'),Input('date-picker-range', 'end_date')]
    )
def update_graph(data_names,start_date,end_date):
    graphs =[]

    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    df = pd.read_csv('./data/stock.csv')
    df['date'] = pd.to_datetime(df_temp['date'])
    df['date'] = df['date'].apply(lambda x: x.date())
    if len(data_names)>2:
        class_choice = 'col s12 m6 l4'
    elif len(data_names) == 2:
        class_choice = 'col s12 m6 l6'
    else:
        class_choice = 'col s12'
    fig = plot_graph(df_temp=df,company_list=data_names,start=start_date, end=end_date)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)