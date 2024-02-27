import dash
from dash import dcc, html, dash_table
import dash_bio as dashbio
import os
from utils import get_lists
from callbacks import register_callbacks
import webbrowser

Results_folder="Results"
TRANSLATION={
     "STFT":"P",
    "C_2D":"C",
    "DC_2D":"D",
    "MP_2D":"M",
    "AP_2D":"A",
    "GAP_2D":"g",
    "GMP_2D":"G",
    "R_2D":"L",
    "BN_2D":"B",
    "IN_2D":"I",
    "DO":"O",
    "D":"F",
    "RES_2D":"R",
    "BOT_2D":"T",
    "FLAT":"E"
}
TRANSLATION_COLUMNS= [{'name': col, 'id': col} for col in TRANSLATION.keys()]
# Load DataFrame from CSV file
app = dash.Dash(__name__, suppress_callback_exceptions=True, assets_folder="assets")

##########################################
df_list, alignment_dicts,results_dicts,df_results_dict,val_acc_dict, parameters, best_individuals, best_sim_df,sequences_best, bad_design_dfs, dfs_good_cases_best_layers={},{},{},{},{},{},{},{},{},{},{}
tests=[]
################################################################
files=[{dir[19:-7]:f"{Results_folder}/{dir}"} for dir in os.listdir(Results_folder) if ".evonas" in dir]

for study in files:
    name=list(study.items())[0][0]
    path=list(study.items())[0][1]
    df_list[name], alignment_dicts[name],  results_dicts[name], df_results_dict[name], val_acc_dict[name], parameters[name], best_individuals[name],best_sim_df[name], sequences_best[name], bad_design_dfs[name], dfs_good_cases_best_layers[name]=get_lists(path,name)
    tests.append(name)
####################################################################
register_callbacks(app, df_list, alignment_dicts, results_dicts, df_results_dict, val_acc_dict, parameters, best_individuals, best_sim_df, sequences_best, bad_design_dfs, dfs_good_cases_best_layers)
###################################################################

app.layout = html.Div([
    html.H1(f'EvoNAS results',  className="MainTitle"),
    html.Br(),
    html.Button('Show/Hide Params', id='button-params',className="ButtonTable", n_clicks=0),
    html.Div(id='table-container', className="TableContainer", children=[
        dash_table.DataTable(
            id='table-params',
            columns=[
                {'name': 'Keys', 'id': 'Keys'},
                {'name': 'Values', 'id': 'Values'}
            ]
        )
    ]),
    html.Br(),
    dcc.Dropdown(['name', 'fitness', 'val_acc', 'inference_time','memory_footprint_h5'], 'fitness', id='sort', searchable=True),
    html.Br(),       

    ################### Dropdowns median fitness and diversity ###################
    html.Div([
    html.Div(className="DropdownContainer",children=[
        html.Label(id="run_time"),
        html.Br(),
        html.Label('First test:'),
        dcc.Dropdown(
            id='dropdown',
            options=tests,
            searchable=True,
            value=tests[0],
            className="Dropdown"
        ),
    ]),
    html.Div(className="DropdownContainer", children=[
        html.Label(id="run_time2"),
        html.Br(),
        html.Label('Second test:'),
        dcc.Dropdown(
            id='dropdown-2',
            options=tests,
            searchable=True,
            value=tests[0],
            className="Dropdown"
        ),
    ])], className="DropdownsBox"),
    ###################
    html.Br(),
    ################## Standard deviation plot acc ###########################
    html.Div(id='std-median-plot-acc', className="PaperPlots"),
    ################## Standard deviation plot ###########################
    html.Div(id='std-median-plot', className="PaperPlots"),
    ################## scatter plot generations ###########################
    html.Div(id='box-plots-total', className="PaperPlots"),
    ################### Similarity distributions ###########################
    html.Br(),
    dcc.Tabs(id='tabs-example-0', value='tab-0', children=[
        dcc.Tab(label='Distributions', value='tab-0'),
        dcc.Tab(label='Pareto Front', value='tab-1')
    ]),
    html.Div(id='tabs-content-0'),

    ############################### Similarity of best individuals ##################
    html.Div(id='heatmaps-best'),
    ################### Sequences best individuals #######################################
    html.Div(children=[dashbio.AlignmentChart(
        id="sequences-best",
        data="ABCD",
        tilewidth=50,
        height=900,
            )]),
    ######################## Table bad design changes ############
    html.Button('Show/Hide wrong design', id='button-design-wrong', className="ButtonTable", n_clicks=0),
    html.Div(id='table-design-wrong-container', className="TableContainer"),
    
    ######################## Table good design changes ############
    html.Button('Show/Hide good design', id='button-design-good', className="ButtonTable", n_clicks=0),
    html.Div(id='table-design-good-container', className="TableContainer", children=[
        dash_table.DataTable(
            data=[TRANSLATION],
            columns=TRANSLATION_COLUMNS,
            style_table={"width":"40%"},  # Center the table
            style_cell={'textAlign': 'center'},
        ),
        dcc.Graph(id="plot-good-design", style={"width":"50%", "height":"400","justify-content": "center"}),
        dash_table.DataTable(
            id='table-design-good'
        )
    ]),
    
    ###############################################################
    html.Br(),
    dcc.Slider(min=1, step=1, value=1,marks=None, id='slider',updatemode="drag",tooltip={"placement": "bottom", "always_visible": True} ),

    dcc.Tabs(id='tabs-example-1', value='tab-1', children=[
        dcc.Tab(label='Heatmap', value='tab-1'),
        dcc.Tab(label='Distribution', value='tab-2'),
        dcc.Tab(label='Clustergram', value='tab-0')
    ]),
    html.Div(id='tabs-content-1'),
    html.H2('Select number of models', className="MainTitle"),
    dcc.Input(
            id="number", type="number",
            debounce=False, placeholder="Number of models",min=0, step=10, className="InputNumber"),
    dcc.Input(
            id="number-thres", type="number",
            debounce=False, placeholder="Number of models",min=-1,max=1, value=0, step=0.05,className="InputNumber"),
    html.H2('Iteration', className="MainTitle"),

    dcc.Tabs(id='tabs-example-2', value='tab-6', children=[
        dcc.Tab(label='Sequence aligned', value='tab-6'),
    ]),
    html.Div(id='tabs-content-2'),

    ################### Dropdowns ###################
    html.Div([
    html.Div(className="DropdownContainer",children=[
        html.Label('With diversity:'),
        dcc.Dropdown(
            id='dropdown-3',
            options=tests,
            searchable=True,
            value=[],
            multi=True,
            className="Dropdown"
        ),
    ]),
    html.Div(className="DropdownContainer", children=[
        html.Label('Without diversity:'),
        dcc.Dropdown(
            id='dropdown-4',
            options=tests,
            searchable=True,
            value=[],
            multi=True,
            className="Dropdown"
        ),
    ])], className="DropdownsBox"),
    ###################
    html.Br(),
    ################## Standard deviation plot acc ###########################
    html.Div(id='std-median-plot-acc-multi', className="PaperPlots"),
    ################## Standard deviation plot ###########################
    html.Div(id='std-median-plot-multi', className="PaperPlots"),
    ################## Max fitness plot #################################
    html.Div(id='max-fitness-plot-multi', className="PaperPlots") 

])
def main():
    webbrowser.open("http://127.0.0.1:8040")   
    app.run_server(debug=False, port=8040)


if __name__ == '__main__':
    main()