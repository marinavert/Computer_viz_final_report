# Import libraries
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
from math import sqrt

# Load the original dataset
df = pd.read_csv("/Users/marinavert/Downloads/SeoulBikeData.csv", encoding='unicode_escape', sep=",") # Change path to make it work
df_dendo = df.drop(["Date", "Hour", "Seasons", "Holiday", "Functioning Day"], axis=1)

# Spectral Graph Analysis Dendogram PCP 
names= ["Rented Bike Count", "Temperature(°C)","Humidity(%)","Wind speed (m/s)", "Visibility (10m)", "Dew point temperature(°C)", \
    "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)"]
names_no_bike = [ "Temperature(°C)","Humidity(%)","Wind speed (m/s)", "Visibility (10m)", "Dew point temperature(°C)", \
    "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)"]

### Step 1 : Find the Correlation between the different features
correlation = [[0 for _ in range(len(names))] for _ in range(len(names))]
for i in range(len(names)):
    for j in range(i, len(names)):
        if i == j:
            continue
        avg_i = df_dendo[names[i]].mean()
        avg_j = df_dendo[names[j]].mean()
        sum_x_y = 0
        sum_sqr_x = 0
        sum_sqr_y = 0
        for k in range(len(df_dendo[names[i]])):
            x_i = df_dendo[names[i]][k]
            x_j = df_dendo[names[j]][k]
            sum_x_y += (x_i - avg_i)*(x_j - avg_j)
            sum_sqr_x += (x_i - avg_i)**2
            sum_sqr_y += (x_j - avg_j)**2
        correlation[i][j] = sum_x_y/(sqrt(sum_sqr_x)*sqrt(sum_sqr_y))
        correlation[j][i] = correlation[i][j]

for i in range(len(correlation)):
    for j in range(len(correlation)):
        if abs(correlation[i][j]) < 0.1:
            correlation[i][j] = 0

a = np.copy(correlation)
d = np.zeros((len(a), len(a)))
sum_corr = np.sum(a, axis = 0)

### Step 2 : Spectral Graph Analysis
for i in range(len(d)):
    d[i, i] = sum_corr[i]
l = d-a
eig, vec = np.linalg.eig(l)
eig_min, eig_idx = int(max(eig)+2), 0
for i in range(len(eig)):
    if eig[i] < eig_min and eig[i] > 0.0001:
        eig_min = eig[i]
        eig_idx = i
vect = [[vec[eig_idx][i], [i]] for i in range(len(a))]
vect.sort()
vect_idx = [vect[j][1] for j in range(len(vect))]

### Step 3 : Make the dendogram tree
for _ in range(8):
    mini = abs(vect[0][0] - vect[1][0])
    idx = 0
    for i in range(len(vect)-1):
        diff = abs(vect[i][0] - vect[i+1][0])
        if diff < mini and diff > 0:
            mini = diff
            idx = i
    merge = vect[idx][1]+vect[idx+1][1]
    vect[idx][1] = merge
    vect[idx][0] = 0
    for i in merge:
        vect[idx][0] += vec[eig_idx][i]
    vect[idx][0] /= len(merge)
    del vect[idx+1]
    print()
    print(vect)
    print(_)

# Get the groups from the dendogram separation
groups = [[8,2,6], [4,1], [5,7], [3]]
group_names = []
for g in groups :
    for i in range(1,len(g)):
        group_names.append(names[g[i]])
group_names

### Step 4 : Axis contraction
normalised = df_dendo.copy()
for name in names_no_bike:
    maximum = max(normalised[name])
    for i in range(len(df_dendo[name])):
        normalised[name][i] /= maximum
for group in groups :
    for i in range(len(normalised[names[0]])):
        total = 0
        for k in group :
            total += normalised[names[k]][i]
        normalised[group[0]] = total/len(group)
    print(group[0])
for name in group_names:
    del normalised[name]
normalised["Seasons"] = df["Seasons"]

# ------------------------------------------- #

# Create the Dash app
app = Dash()

# Set up the app layout
season_dropdown = dcc.Dropdown(options=df['Seasons'].unique(),
                            value='Spring')

# Scatter matrix to see all the features
scatter = px.scatter_matrix(df, color="Seasons",width=1200, height=1000)

app.layout = html.Div(children=[
    # 1 : Scatter matrix
    html.Div(children=[
        html.H2(children='Scatter matrix'),
        dcc.Graph(id='scatter', figure = scatter)
    ]),

    # 2 : Select the season you want to focus on
    html.Div(children=[
        html.H2(children='Select the season you are interested in'),
        season_dropdown
    ]),

    # 3 : Number of Rented bike per hour 
    html.Div(children=[
        dcc.Graph(id='price-graph')
    ]),

    # 4 : PCP with all data 
    html.Div(children=[
        dcc.Graph(id="pcp"),
    ]),

    # 5 : PCP with only 5D after dendogram
    html.Div(children=[
        dcc.Graph(id="reduced_pcp"),
    ])
])


# Set up the callback function
@app.callback(
    Output(component_id='price-graph', component_property='figure'),
    Output(component_id='pcp', component_property='figure'),
    Output(component_id='reduced_pcp', component_property='figure'),
    Input(component_id=season_dropdown, component_property='value')
)
def update_graph(selected_season):
    df_seasons = df[df['Seasons'] == selected_season]
    df_dendo = normalised[normalised["Seasons"] == selected_season]
    df_dendo = df_dendo.rename(columns = {'Rented Bike Count':'Bike','Wind speed (m/s)':'Wind, Humid, Solar','Visibility (10m)':'Visib, Temp', \
    'Dew point temperature(°C)':'Dew_Point, Rain', 'Snowfall (cm)':'Snow'})
    print(df_seasons.columns)
    line_fig = px.line(df_seasons,
                       x='Hour', y='Rented Bike Count',
                       color='Date',
                       title=f'Number of rented bikes per hour in {selected_season}')
    pcp = px.parallel_coordinates(df_seasons, title=f'Parallel coordinates plot for all weather information (10D) in {selected_season}')
    reduced_pcp = px.parallel_coordinates(df_dendo[['Wind, Humid, Solar', 'Visib, Temp', 'Dew_Point, Rain','Snow', 'Bike']], title=f'Reduced Parallel coordinates plot (5D) for all weather information in {selected_season}')
    return line_fig, pcp, reduced_pcp


# Run local server
if __name__ == '__main__':
    app.run_server(debug=True)