from dash import dcc, html, Input, Output,State, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
import dash_bio as dashbio
from plotly.subplots import make_subplots



def register_callbacks(app, df_list, alignment_dicts, results_dicts, df_results_dict, val_acc_dict, parameters, best_individuals, best_sim_df, sequences_best, bad_design_dfs, dfs_good_cases_best_layers):
    ########################### Select good design cases table and show it #######################
    @app.callback(
        Output('table-design-good', 'data'),
        Output('table-design-good', 'columns'),
        Output('table-design-good-container', 'style'),
        Output('plot-good-design', 'figure'),
        Input('button-design-good', 'n_clicks'),
        Input('dropdown', 'value'),
        suppress_callback_exceptions=True
    )
    def update_table_design_cases_best(n_clicks, selected_option):
        # Check if the button was clicked and return the appropriate data and style
        if n_clicks%2==0:
            return dash.no_update,dash.no_update, {'display': 'none'}, dash.no_update
        else:
            fig = px.bar(dfs_good_cases_best_layers[selected_option].sort_values("weight", ascending=False).head(15), y='layers', x='weight',hover_data=['appearances', 'median_repetitions', 'typical_positions'], text_auto='.2s',
                title="Most repeated blocks and layers in best individuals", orientation='h', template="plotly_white", height=500)

            return dfs_good_cases_best_layers[selected_option].to_dict('records'),[{'name': col, 'id': col} for col in dfs_good_cases_best_layers[selected_option].columns], {'display': True}, fig


    ########################### Select wrong design cases table and show it #######################
    @app.callback(
        Output('table-design-wrong-container', 'style'),
        Output('table-design-wrong-container', 'children'),
        Input('button-design-wrong', 'n_clicks'),
        Input('dropdown', 'value'),
        suppress_callback_exceptions=True
    )
    def update_table_design_cases(n_clicks, selected_option):
        # Check if the button was clicked and return the appropriate data and style
        if n_clicks%2==0:
            return {'display': 'none'},[]
        else:
            data_columns=bad_design_dfs[selected_option].columns
            children=[
                dash_table.DataTable(
                    id='table-design-wrong',
                    data=bad_design_dfs[selected_option].to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in data_columns],
                    ),]
            for i in range(10):
                #data_row=bad_design_dfs[selected_option][["Generation","Individual_1","Individual_2","Fitness difference","Similarity","Substitutions","Insertions","Deletions"]].loc[i].to_dict()
                data_row=bad_design_dfs[selected_option].loc[i].to_dict()

                subchildren=[
                html.Br(),
                dash_table.DataTable(
                    id='table-design-wrong',
                    data=[data_row],
                    columns=[{'name': col, 'id': col} for col in data_columns if col not in ["Sequence_1","Sequence_2"]],
                    style_table={'overflowX': 'auto', 'font_size': '12px', 'padding': '1px'}
                    ),
                #html.Br(),
                dashbio.AlignmentChart(
                    id="two_alignment",
                    data=alignment_dicts[selected_option][data_row["Generation"]-1][f"('{data_row['Individual_1']}', '{data_row['Individual_2']}')"],
                    #data=">example\nABCD\n",
                    showconservation=False,
                    colorscale='lesk',
                    showconsensus=False,
                    showgap=False,
                    numtiles=int(max(1.2*len(data_row["Sequence_1"]),1.2*len("Sequence_2"))),
                    #height=150,
                    textsize=15,
                    tilewidth=40,

                    )
                ]

                children.extend(subchildren)

            return  {'display': True}, children
    ########################### Total running time ################################
    @app.callback(
            Output("run_time","children"),
            Output("run_time2","children"),
            Input('dropdown', 'value'),
            Input('dropdown-2', 'value'))
    def sort_values(study, study2):
            return "Run time: "+str(best_individuals[study]["elapsed time(s)"].sum()/3600),"Run time: "+str(best_individuals[study2]["elapsed time(s)"].sum()/3600)
    ########################### Select params table and show it #######################
    @app.callback(
        Output('table-params', 'data'),
        Output('table-container', 'style'),
        Output('number', 'value'),
        Output('number', 'max'),
        Input('button-params', 'n_clicks'),
        Input('dropdown', 'value'),
        suppress_callback_exceptions=True

    )
    def update_table(n_clicks, selected_option):
        # Check if the button was clicked and return the appropriate data and style
        if n_clicks%2==1:
            return [], {'display': 'none'}, dash.no_update, dash.no_update
        else:
            selected_data = parameters[selected_option]
            n_population=parameters[selected_option]["population_size"]
            selected_df = pd.DataFrame(list(selected_data.items()), columns=['Keys', 'Values'])
            selected_df.Values=selected_df.Values.astype("str")
            return selected_df.to_dict('records'), {'display': True}, n_population, n_population
    ########################## Sequences best individuals ###########################
    @app.callback(
            Output("sequences-best","data"),
            Input("dropdown", "value"))
    def sequences_values(study):
            
            return sequences_best[study]
    ########################### Sort values by variable ################################
    @app.callback(
            Output("slider","value"),
            Input("sort", "value"))
    def sort_values(sort):
            for folder in df_results_dict.keys():
                for gen in range(len(df_list[folder])):
                    desired_order=df_results_dict[folder][gen].sort_values(by=[sort,"fitness","val_acc","inference_time", "memory_footprint_h5"], ascending=[False, False, False, True, True]).name.tolist()
                    df_list[folder][gen]=df_list[folder][gen].reindex(desired_order,columns=desired_order)
            return 1

    ########################### Median plots accuracy with shadows #########################
    @app.callback(
        Output('std-median-plot-acc', 'children'),
        [Input("dropdown", "value"),
        Input("dropdown-2","value")],
        suppress_callback_exceptions=True
    )
    def std_plot_median_acc(drop, drop2):
        # Sample data
        gen=max(len(df_list[drop]),len(df_list[drop2]))
        x_values = list(np.arange(1,gen+1,1))
        y_values_line1=[]
        std_dev_line1=[]
        for i in range(len(df_list[drop])):
            matrix = df_results_dict[drop][i]
            matrix=matrix[((matrix.inference_time!=np.inf) & (matrix.fitness!=-10001.0)) ]["fitness"].values
            y_values_line1.append(np.median(matrix))
            std_dev_line1.append(np.round(np.std(matrix),3))
        #print(std_dev_line1)
        #y_values_line1 = [np.median(mat) for mat in df_list[drop]]
        #std_dev_line1 = [1, 2, 1.5, 1, 3]
        y_values_line2=[]
        std_dev_line2=[]
        for i in range(len(df_list[drop2])):
            matrix = df_results_dict[drop2][i]
            matrix=matrix[((matrix.inference_time!=np.inf) & (matrix.fitness!=-10001.0)) ]["fitness"].values
            y_values_line2.append(np.median(matrix))
            std_dev_line2.append(np.round(np.std(matrix),3))

        #print(std_dev_line2)
        #y_values_line2 = [8, 12, 10, 14, 18]
        #std_dev_line2 = [0.5, 1, 0.8, 0.7, 1.5]

        # Create a line plot
        fig = go.Figure()

        # Add the first line trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_line1,
            mode='lines',
            name=f'{drop} median fitness',
            line=dict(color='rgb(255,0,0)')

        ))

        # Add the shaded area for standard deviation for Line 1
        fig.add_trace(go.Scatter(
            x=x_values + x_values[::-1],
            y=[y - std_dev for y, std_dev in zip(y_values_line1, std_dev_line1)] +
            [y + std_dev for y, std_dev in zip(y_values_line1[::-1], std_dev_line1[::-1])],
            fill='toself',
            fillcolor='rgba(217,121,123,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Std dev {drop}'
        ))

        # Add the second line trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_line2,
            mode='lines',
            name=f'{drop2} median fitness',
            line=dict(color='rgb(144,143,161)')
        ))

        # Add the shaded area for standard deviation for Line 2
        fig.add_trace(go.Scatter(
            x=x_values + x_values[::-1],
            y=[y - std_dev for y, std_dev in zip(y_values_line2, std_dev_line2)] +
            [y + std_dev for y, std_dev in zip(y_values_line2[::-1], std_dev_line2[::-1])],
            fill='toself',
            fillcolor='rgba(152,151,166,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Std dev {drop2}'
        ))

        # Update layout for better visualization
        fig.update_layout(
            title='Population fitness per generation',
            xaxis_title='Generation',
            yaxis_title='Fitness',
            template="presentation", 
            showlegend=True
        )
        fig.update_layout(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
        return dcc.Graph(figure=fig, style={"width":"100%", "height":"100%"})

    ########################### Median plots similarity with shadows #########################
    @app.callback(
        Output('std-median-plot', 'children'),
        [Input("dropdown", "value"),
        Input("dropdown-2","value")],
        suppress_callback_exceptions=True
    )
    def std_plot_median(drop, drop2):
        # Sample data
        gen=max(len(df_list[drop]),len(df_list[drop2]))
        x_values = list(np.arange(1,gen+1,1))
        
        y_values_line1=[]
        std_dev_line1=[]
        for i in range(len(df_list[drop])):
            matrix = df_list[drop][i].to_numpy()
            upper_semidiagonal_mask = np.triu(np.ones(matrix.shape), k=1)
            upper_semidiagonal = matrix[upper_semidiagonal_mask == 1]
            y_values_line1.append(np.median(upper_semidiagonal))
            std_dev_line1.append(np.round(np.std(upper_semidiagonal),2))
        print(std_dev_line1)

        #y_values_line1 = [np.median(mat) for mat in df_list[drop]]
        #std_dev_line1 = [1, 2, 1.5, 1, 3]
        y_values_line2=[]
        std_dev_line2=[]
        for i in range(len(df_list[drop2])):
            matrix = df_list[drop2][i].to_numpy()
            upper_semidiagonal_mask = np.triu(np.ones(matrix.shape), k=1)
            upper_semidiagonal = matrix[upper_semidiagonal_mask == 1]
            y_values_line2.append(np.median(upper_semidiagonal))
            std_dev_line2.append(np.round(np.std(upper_semidiagonal),2))

        len(std_dev_line2)
        #y_values_line2 = [8, 12, 10, 14, 18]
        #std_dev_line2 = [0.5, 1, 0.8, 0.7, 1.5]

        # Create a line plot
        fig = go.Figure()

        # Add the first line trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_line1,
            mode='lines',
            name=f'{drop} median similarity',
            line=dict(color='rgb(255,0,0)')
        ))

        # Add the shaded area for standard deviation for Line 1
        fig.add_trace(go.Scatter(
            x=x_values + x_values[::-1],
            y=[y - std_dev for y, std_dev in zip(y_values_line1, std_dev_line1)] +
            [y + std_dev for y, std_dev in zip(y_values_line1[::-1], std_dev_line1[::-1])],
            fill='toself',
            fillcolor='rgba(217,121,123,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Std dev {drop}'
        ))

        # Add the second line trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_line2,
            mode='lines',
            name=f'{drop2} median similarity',
            line=dict(color='rgb(144,143,161)')
            
        ))

        # Add the shaded area for standard deviation for Line 2
        fig.add_trace(go.Scatter(
            x=x_values + x_values[::-1],
            y=[y - std_dev for y, std_dev in zip(y_values_line2, std_dev_line2)] +
            [y + std_dev for y, std_dev in zip(y_values_line2[::-1], std_dev_line2[::-1])],
            fill='toself',
            fillcolor='rgba(152,151,166,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Std dev {drop2}'
        ))

        # Update layout for better visualization
        fig.update_layout(
            title='Model similarity per generation',
            xaxis_title='Generation',
            yaxis_title='similarity',
            template="presentation", 
            showlegend=True
        )
        fig.update_layout(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
        return dcc.Graph(figure=fig, style={"width":"100%", "height":"100%"})
    ########################### Box plots comparison ###############################
    @app.callback(
        Output('box-plots-total', 'children'),
        [Input("dropdown", "value"),
        Input("dropdown-2","value"),
        Input("sort","value")],
        suppress_callback_exceptions=True
    )
    def box_plot_total(drop, drop2, sort):
        df_total=pd.concat(df_results_dict[drop])
        df_total=df_total[df_total.inference_time!=np.inf]
        df_total["Generation"]=df_total["Generation"].astype(str)  
        fig_box = px.box(df_total, x='Generation',points="suspectedoutliers", y=sort, title=f"{sort} distribution Over Generations")
        fig_box.update_traces(marker_color='blue',name=f'{drop} distribution',showlegend=True,visible='legendonly',opacity=0.7)
        # Create a boxplot for study2
        df_total2=pd.concat(df_results_dict[drop2])
        df_total2=df_total2[df_total2.inference_time!=np.inf]
        df_total2["Generation"]=df_total2["Generation"].astype(str)  
        fig_box2 = px.box(df_total2, x='Generation',points="suspectedoutliers", y=sort, title=f"{sort} distribution Over Generations")
        fig_box2.update_traces(marker_color='#EF6262',name=f'{drop2} distribution',showlegend=True,visible='legendonly',opacity=0.7)

        df_total["Generation"]=df_total["Generation"].astype(int)  
        df_total2["Generation"]=df_total2["Generation"].astype(int)  
        # Create a lineplot of max accuracy for each study
        fig_line = px.line(df_total[["Generation", sort]].groupby("Generation").max().reset_index(), x='Generation', y=sort
                        , title=f"Max {sort} Comparison",markers=True)
        fig_line.update_traces(line=dict(color='#0D1282'), name=f'{drop} max {sort}', showlegend=True, visible='legendonly')
        # Max of study 2
        fig_line2 = px.line(df_total2[["Generation", sort]].groupby("Generation").max().reset_index(), x='Generation', y=sort
                        , title=f"Max {sort} Comparison", markers=True)
        fig_line2.update_traces(line=dict(color='#900C3F'), name=f'{drop2} max {sort}', showlegend=True, visible='legendonly')

        fig_line3 = px.line(best_individuals[drop], x='Generation', y=sort
                        , title=f"Best models {sort} Comparison", markers=True, hover_data="Name")
        fig_line3.update_traces(line=dict(color='#900C3F'), name=f'{drop} best model {sort}', showlegend=True)

        fig_line4 = px.line(best_individuals[drop2], x='Generation', y=sort
                        , title=f"Best models {sort} Comparison", markers=True, hover_data="Name")
        fig_line4.update_traces(line=dict(color='#0D1282'), name=f'{drop2} best model {sort}', showlegend=True)

        # Create subplots with one row and two columns
        fig_combined = make_subplots(rows=1, cols=1, subplot_titles=[f'{sort} Distribution Over Generations'])

        # Add box plots to the first subplot
        for trace in fig_box.data:
            fig_combined.add_trace(trace, row=1, col=1)

        for trace in fig_box2.data:
            fig_combined.add_trace(trace, row=1, col=1)

        # Add line plot to the second subplot
        for trace in fig_line.data:
            fig_combined.add_trace(trace, row=1, col=1)
        for trace in fig_line2.data:
            fig_combined.add_trace(trace, row=1, col=1)
        for trace in fig_line3.data:
            fig_combined.add_trace(trace, row=1, col=1)
        for trace in fig_line4.data:
            fig_combined.add_trace(trace, row=1, col=1)
        # Update layout
        fig_combined.update_layout(template="presentation", showlegend=True)
        fig_combined.update_layout(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
        if sort=="inference_time":
            fig_combined.update_yaxes(range=[0, 10])
        elif sort=="val_acc":
            fig_combined.update_yaxes(range=[0, 1])
        elif sort=="fitness":
            fig_combined.update_yaxes(range=[0, 1])
        elif sort=="memory_footprint_h5":
            fig_combined.update_yaxes(range=[0, 5000000])
        fig_combined.update_xaxes(type="category")
        fig_combined.update_layout()

        return dcc.Graph(figure=fig_combined, style={"width":"100%", "height":"100%"})
    ########################### Distribution and pareto front #######################

    @app.callback(
        Output('tabs-content-0', 'children'),
        Input('tabs-example-0', 'value'),
        Input("dropdown", "value"),
        suppress_callback_exceptions=True
    )
    def filter_heatmap(tab, drop):
        if tab=="tab-0":
                box_traces = []
                dfs=[]
                for i in range(0, len(df_list[drop])):
                    matrix = df_list[drop][i].to_numpy()
                    #upper_semidiagonal_mask = np.triu(np.ones(matrix.shape), k=1)
                    #upper_semidiagonal = matrix[upper_semidiagonal_mask == 1]
                    #dfs.append(upper_semidiagonal)
                    dfs.append(matrix.flatten())

                # Loop through each generation DataFrame and create a box trace
                c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 25)]
                for i, gen_df in enumerate(dfs):
                    box_trace = go.Box(y=gen_df,  name=i+1, boxpoints="outliers", marker_color=c[i], whiskerwidth=1,)
                    box_traces.append(box_trace)

    #############################
                #dfs=[]
                #for i in range(0, len(df_list[drop])):
                #    matrix = df_list[drop][i].values.flatten()
                    #upper_semidiagonal_mask = np.triu(np.ones(matrix.shape), k=1)
                    #upper_semidiagonal = matrix[upper_semidiagonal_mask == 1]
                    #print(upper_semidiagonal.shape())
                #    dfs.append( matrix)

                # Loop through each generation DataFrame and create a box trace
                #c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 25)]
                #for i, gen_df in enumerate(dfs):
                #    box_trace = go.Box(y=gen_df,  name=i+1, boxpoints="outliers", marker_color=c[i], whiskerwidth=1,)
                #    box_traces.append(box_trace)

    ###############################

                layout = go.Layout(
                    title=f"Similarity value for {len(df_list[drop])} Generations",
                    xaxis=dict(title="Generation"),
                    yaxis=dict(title="Value", showgrid=True,range=[0.1, 0.9]
                                ),
                    paper_bgcolor='rgb(243, 243, 243)',
                    plot_bgcolor='rgb(243, 243, 243)',
                    showlegend=True,
                    height=700
                
                )
                fig = go.Figure(data=box_traces, layout=layout)
                return dcc.Graph(figure=fig)
        elif tab=="tab-1":
            #print(df_results_dict[drop][0].head())
            result_df = pd.concat(df_results_dict[drop], ignore_index=False)
            print(result_df.head())
            print(result_df.columns)
            result_df["fitness"] = result_df["fitness"].clip(lower=-1)
            result_df["fitness"] = result_df["fitness"].astype("float")
            result_df["val_acc"] = result_df["val_acc"].astype("float")
            # Create a scatter plot with colors based on the 'Color' column
            fig = px.scatter(result_df, x='memory_footprint_h5', y='inference_time', hover_data=['name', "fitness"], color='val_acc', color_continuous_scale='Viridis', symbol="Generation")

            # Customize hover text
            hover_text = []
            #for i, row in result_df.iterrows():
            #    hover_text.append(f"{row['Generation']}<br> : {row['val_acc']}")

            #fig.update_traces(text=hover_text, hoverinfo='text+x+y')
            fig.update_layout(
                xaxis=dict(range=[0, 6000000]),
                yaxis=dict(range=[0, 8])
            )
            return dcc.Graph(figure=fig)
    ########################### Best models heatmaps ##############################
    @app.callback(
        Output('heatmaps-best', 'children'),
        Input("dropdown", "value"),
        Input("dropdown-2", "value"),
        suppress_callback_exceptions=True
    )
    def filter_heatmap(drop, drop2):
            fig = px.imshow(best_sim_df[drop],text_auto=False, zmin=0, zmax=1)
            fig.update_layout(
                hovermode='closest',
                height=700,  # Set the height of the heatmap (in pixels)
                width=900  # Set the width of the heatmap (in pixels)
            )
            fig2 = px.imshow(best_sim_df[drop2],text_auto=False, zmin=0, zmax=1)
            fig2.update_layout(
                hovermode='closest',
                height=700,  # Set the height of the heatmap (in pixels)
                width=900  # Set the width of the heatmap (in pixels)
            )
            return [dcc.Graph(figure=fig),dcc.Graph(figure=fig2)]
    ############################# Heatmap, Distribution, Clustergram #########################################################
    @app.callback(
        Output('tabs-content-1', 'children'),
        Input('tabs-example-1', 'value'),
        Input("number", "value"),
        Input("slider","value"),
        Input("dropdown", "value"),
        Input("number-thres", "value"),
        suppress_callback_exceptions=True
    )
    def filter_heatmap(tab,cols, slider, drop, thres):

        if tab=="tab-1":
            #sep=len(df_list[slider].columns)/cols
            if thres<0:
                new_df = df_list[drop][slider-1].iloc[:cols,:cols].copy()
                new_df[new_df >= -1*thres] = 1
                if cols>20:
                    fig = px.imshow(new_df,text_auto=False, zmin=0, zmax=1)
                else:
                    fig = px.imshow(new_df,text_auto=True, zmin=0, zmax=1)
            elif thres==0:
                if cols>20:
                    fig = px.imshow(df_list[drop][slider-1].iloc[:cols,:cols],text_auto=False, zmin=0, zmax=1)
                    #fig = go.Figure(go.Heatmap(z=df_list[drop][slider-1].iloc[:cols,:cols],zmin=0, zmax=1, text=False,hovertext=hovertext))
                else:
                    fig = px.imshow(df_list[drop][slider-1].iloc[:cols,:cols],text_auto=True, zmin=0, zmax=1)
                    #fig = go.Figure(go.Heatmap(z=df_list[drop][slider-1].iloc[:cols,:cols],zmin=0, zmax=1, hovertext=hovertext))
                #fig=px.imshow(df.iloc[:5,:5])
            else:
                new_df = df_list[drop][slider-1].iloc[:cols,:cols].copy()
                new_df[new_df <= thres] = 0
                if cols>20:
                    fig = px.imshow(new_df,text_auto=False, zmin=0, zmax=1)
                else:
                    fig = px.imshow(new_df,text_auto=True, zmin=0, zmax=1)

            fig.update_layout(
                hovermode='closest',
                #height=500+1*(cols-10),  # Set the height of the heatmap (in pixels)
                #width=500+1*(cols-10),   # Set the width of the heatmap (in pixels)
                height=600,  # Set the height of the heatmap (in pixels)
                width=600,  
            )

            fig.update_xaxes(showticklabels=False, title_text=f'Generation {slider}', title_font=dict(size=25))
            fig.update_yaxes(showticklabels=False, title='')
            fig.update_layout( margin=dict(l=10, r=10, t=10, b=10),  coloraxis_colorbar=dict(len=0.75))
                #color_axis = fig['data'][0]['coloraxis']
                #new_fig = go.Figure(layout={'coloraxis': color_axis})
                #new_fig.write_image(f'images/axis_{slider}.svg')
            #pio.write_svg(fig, f'heatmap_{slider}.svg')
            #fig.write_image(f'images/heatmap_{slider}.svg')

            return dcc.Graph(id="graph",figure=fig)
        elif tab=="tab-2":
            sep=len(df_list[drop][slider-1].columns)/cols
            matrix = df_list[drop][slider-1].iloc[:cols,:cols].to_numpy()
            upper_semidiagonal_mask = np.triu(np.ones(matrix.shape), k=1)

            # Apply the mask to get the upper semidiagonal elements
            upper_semidiagonal = matrix[upper_semidiagonal_mask == 1]
            fig = go.Figure(data=go.Violin(y=upper_semidiagonal,points="outliers", box_visible=True, line_color='black',
                                        meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,x0='Models'))

            # Add lines for 2 standard deviations from Q1 and Q3
            fig.add_shape(
                type="line",
                x0=-0.5, y0=0.4, x1=0.5, y1=0.4,
                line=dict(color="red", width=2),

            )

            fig.update_layout(
                #title='Boxes area distribution',
                yaxis_title='Similarity score',
                xaxis_title='',
                template="presentation",
                width=800+1*(cols-10),  
                height=600+1*(cols-10),
                yaxis=dict(
            range=[0, 1]  # Set your desired lower and upper limits here
        )
            )
            return dcc.Graph(figure=fig)
        elif tab=="tab-0":
            columns = list(df_list[drop][slider-1].columns.values[:cols])
            rows = list(df_list[drop][slider-1].index[:cols])

            clustergram, traces = dashbio.Clustergram(
                data=df_list[drop][slider-1].iloc[:cols,:cols].values,
                row_labels=rows,
                #generate_curves_dict=True,
                return_computed_traces=True,
                column_labels=columns,
                #color_threshold={
                #    'row': 250,
                #    'col': 700
                #},
                display_ratio=0.01,
                height=700+1*(cols-10),
                width=700+1*(cols-10),
                #height=500,
                #width=500,
                hidden_labels=columns,
                color_list={
                    'row': ['#636EFA', '#00CC96', '#19D3F3'],
                    'col': ['#AB63FA', '#EF553B'],
                    'bg': '#506784'
                },
                center_values=False
            ) 
            #clustergram.data[-1].z = np.where(clustergram.data[-1].z > abs(thres), clustergram.data[-1].z, 0)
            global cluster_rows
            cluster_rows=[]
            for row in traces["row_ids"]:
                cluster_rows.append(rows[row])
            global cluster_columns
            cluster_columns=[]
            for col in traces["column_ids"]:
                cluster_columns.append(columns[col])

            return [ html.Div([
        html.Div(dcc.Graph(id="clustergram",figure=clustergram), style={'display': 'inline-block', 'width': '50%'}),  # 'six columns' means half of the row width
        ])]
        else:
                box_traces = []
                dfs=[]
                for i in range(0, len(df_list[drop])):
                    matrix = df_list[drop][i].to_numpy()
                    upper_semidiagonal_mask = np.triu(np.ones(matrix.shape), k=1)
                    upper_semidiagonal = matrix[upper_semidiagonal_mask == 1]
                    dfs.append(upper_semidiagonal)

                # Loop through each generation DataFrame and create a box trace
                c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 25)]
                for i, gen_df in enumerate(dfs):
                    box_trace = go.Box(y=gen_df, name=i+1, boxpoints="outliers", marker_color=c[i], whiskerwidth=1,)
                    box_traces.append(box_trace)

                layout = go.Layout(
                    title=f"Similarity value for {len(df_list[drop])} Generations",
                    xaxis=dict(title="Generation"),
                    yaxis=dict(title="Value", showgrid=True,range=[0.1, 0.9]
                                ),
                    paper_bgcolor='rgb(243, 243, 243)',
                    plot_bgcolor='rgb(243, 243, 243)',
                    showlegend=False,
                    height=700
                
                )
                fig = go.Figure(data=box_traces, layout=layout)
                return dcc.Graph(figure=fig)

    @app.callback(
            Output("slider", "max"),
            Input("dropdown", "value"))
    def n_generation(drop):
        return len(df_list[drop])

    #################### Table + sequences alignment + Val accuracy ################
    @app.callback(
        Output('tabs-content-2', 'children'),
        Input('tabs-example-2', 'value'),
        Input("number", "value"),
        Input("slider","value"),
        Input("dropdown", "value"),
        suppress_callback_exceptions=True,allow_duplicate=True
    )
    def representations(tab,cols, slider, drop):
            return [dash_table.DataTable(
                        id="results_table",
                        columns=[
                            {'name': 'Attribute', 'id': 'Attribute'},
                            {'name': 'Value', 'id': 'Value'},
                        ],
                        data=[
                            {'Attribute': key, 'Value': value} for key, value in {"memory_footprint_h5": 77440}.items()
                        ],
                    ),
                    dashbio.AlignmentChart(
                                id="two_alignment",
                                #data=alignments_sequences[('adamant_panda','adamant_trogon')],
                                data=">example\nABCD\n",
                                showconservation=True,
                                colorscale='lesk',
                                showconsensus=False,
                                showgap=True,
                                height=500,
                                textsize=18,
                                tilewidth=40),
                    
                    dcc.Graph(id='accuracy-plot')
                    ]
    ################ Display val accuracy ############
    @app.callback(
        Output('accuracy-plot', 'figure'),
        Input("graph", "clickData"),
        State("slider", "value"),
        State("dropdown", "value"),
        suppress_callback_exceptions=True,prevent_initial_call=True
    )
    def display_alignment( click_data, slider, drop):
        if click_data:
            name_a = click_data["points"][0]["x"]
            name_b = click_data["points"][0]["y"]
            val_acc_a=val_acc_dict[drop][slider-1][name_a]
            val_acc_b=val_acc_dict[drop][slider-1][name_b]
            epochs=np.arange(1,parameters[drop]["nb_epochs"])
            model_a=go.Scatter(x=epochs, y=val_acc_a, mode='lines+markers', name=name_a)
            model_b=go.Scatter(x=epochs, y=val_acc_b, mode='lines+markers', name=name_b)
            layout=go.Layout(title='Validation Acc vs Epochs', xaxis=dict(title="Epoch"), yaxis=dict(title='Validation accuracy'))
            return {'data':[model_a, model_b],'layout':layout}
        return dash.no_update
    ################ Display sequences alignment ##############################
    @app.callback(
        Output("two_alignment", "data",allow_duplicate=True),
        Input("graph", "clickData"),  # Input clickData for capturing click event
        State("slider", "value"),
        State("dropdown", "value"),
        suppress_callback_exceptions=True,prevent_initial_call=True
    )
    def display_alignment( click_data, slider, drop):
        if click_data:
            x_val = click_data["points"][0]["x"]
            y_val = click_data["points"][0]["y"]
            #print(alignment_dicts[drop][slider-1].keys())
            return  alignment_dicts[drop][slider-1][f"('{x_val}', '{y_val}')"]
        return ">example\nABCD\n"

    @app.callback(
        Output("two_alignment", "data",allow_duplicate=True),
        Input("clustergram", "clickData"),  # Input clickData for capturing click event
        State("slider", "value"),
        State("dropdown", "value"),
        suppress_callback_exceptions=True,prevent_initial_call=True
    )
    def display_alignment( click_data, slider, drop):
        if click_data:
    #        print(click_data["points"][0]["x"])
    #        print(click_data["points"][0]["y"])
            #print(click_data["points"][0])
            x_val = cluster_columns[int(click_data["points"][0]["x"]/10)]
            y_val = cluster_rows[int(click_data["points"][0]["y"]/10)]

    #        print(x_val)
    #        print(y_val)
            #print(alignment_dicts[drop][slider-1].keys())
            return  alignment_dicts[drop][slider-1][f"('{x_val}', '{y_val}')"]
        return ">example\nABCD\n"

    #################### Fill table when selecting models ##########################
    @app.callback(
        Output("results_table", "data",allow_duplicate=True),
        Output("results_table", "columns",allow_duplicate=True),
        Input("graph", "clickData"),  # Input clickData for capturing click event
        State("slider", "value"),
        State("dropdown", "value"),
        suppress_callback_exceptions=True,prevent_initial_call=True
    )
    def display_results_hist( click_data, slider, drop):
        if click_data:
            #print(click_data)
            #x_val = cluster_columns[int(click_data["points"][0]["x"]/10)]
            #y_val = cluster_rows[int(click_data["points"][0]["y"]/10)]
            x_val = click_data["points"][0]["x"]
            y_val = click_data["points"][0]["y"]
            chrom1=results_dicts[drop][slider-1][x_val]
            #print("###############")
            #print(chrom1)
            chrom2=results_dicts[drop][slider-1][y_val]
            #print("###################")
            #   print(chrom2)
            combined_dict = {}
            # Merge the dictionaries while accumulating values in lists
            for key, value in chrom1.items():
                combined_dict.setdefault(key, []).append(str(value))
            for key, value in chrom2.items():
                combined_dict.setdefault(key, []).append(str(value))
            df = pd.DataFrame.from_dict(combined_dict, orient='index').reset_index()
            
            df.columns=["parameter", str(x_val), str(y_val)] if x_val!=y_val else ["parameter", str(x_val), str(y_val)+"2"]
            return df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]
        columns=[
            {'name': 'Attribute', 'id': 'Attribute'},
            {'name': 'Value', 'id': 'Value'},
        ]
        data=[
            {'Attribute': key, 'Value': value} for key, value in {"memory_footprint_h5": 77440}.items()
        ]
        return data, columns

    @app.callback(
        Output("results_table", "data",allow_duplicate=True),
        Output("results_table", "columns",allow_duplicate=True),
        Input("clustergram", "clickData"),  # Input clickData for capturing click event
        State("slider", "value"),
        State("dropdown", "value"),
        suppress_callback_exceptions=True,prevent_initial_call=True
    )
    def display_results_cluster( click_data, slider, drop):
        if click_data:
            print(click_data)
            x_val = cluster_columns[int(click_data["points"][0]["x"]/10)]
            y_val = cluster_rows[int(click_data["points"][0]["y"]/10)]
            chrom1=results_dicts[drop][slider-1][x_val]
            chrom2=results_dicts[drop][slider-1][y_val]
            combined_dict = {}
            # Merge the dictionaries while accumulating values in lists
            for key, value in chrom1.items():
                combined_dict.setdefault(key, []).append(str(value))
            for key, value in chrom2.items():
                combined_dict.setdefault(key, []).append(str(value))
            df = pd.DataFrame.from_dict(combined_dict, orient='index').reset_index()
            
            df.columns=["parameter", str(x_val), str(y_val)] if x_val!=y_val else ["parameter", str(x_val), str(y_val)+"2"]
            return df.to_dict('records'), [{'name': col, 'id': col} for col in df.columns]
        columns=[
            {'name': 'Attribute', 'id': 'Attribute'},
            {'name': 'Value', 'id': 'Value'},
        ]
        data=[
            {'Attribute': key, 'Value': value} for key, value in {"memory_footprint_h5": 77440}.items()
        ]
        return data, columns


    ########################### Median plots accuracy with shadows multiple selection #########################
    @app.callback(
        Output('std-median-plot-acc-multi', 'children'),
        [Input("dropdown-3", "value"),
        Input("dropdown-4","value")],
        suppress_callback_exceptions=True
    )
    def std_plot_median_acc_multi(drops, drops2):
        # Sample data
        y_values_line1=[]
        #std_dev_line1=[]
        max_values_line1=[]
        min_values_line1=[]
        if drops:
            gens = len(df_list[drops[0]])
            x_values = list(np.arange(1,gens+1,1))
            for gen in x_values[:]:
                total_matrix=[]
                for drop in drops:
                    matrix = df_results_dict[drop][gen-1]
                    matrix=matrix[((matrix.inference_time!=np.inf) & (matrix.fitness!=-10001.0)) ]["fitness"].values
                    #total_matrix.extend(list(matrix))
                    total_matrix.append(np.max(matrix))
                y_values_line1.append(np.mean(total_matrix))
                max_values_line1.append(np.max(total_matrix))
                min_values_line1.append(np.min(total_matrix))
                #std_dev_line1.append(np.round(np.std(total_matrix),3))

        y_values_line2=[]
        #std_dev_line2=[]
        max_values_line2=[]
        min_values_line2=[]
        if drops2:
            gens = len(df_list[drops2[0]])
            x_values = list(np.arange(1,gens+1,1))
            for gen in x_values:
                total_matrix=[]
                for drop in drops2:
                    matrix = df_results_dict[drop][gen-1]
                    matrix=matrix[((matrix.inference_time!=np.inf) & (matrix.fitness!=-10001.0)) ]["fitness"].values
                    #total_matrix.extend(list(matrix))
                    total_matrix.append(np.max(matrix))
                y_values_line2.append(np.median(total_matrix))
                max_values_line2.append(np.max(total_matrix))
                min_values_line2.append(np.min(total_matrix))
                #std_dev_line2.append(np.round(np.std(total_matrix),3))

        # Create a line plot
        fig = go.Figure()

        # Add the first line trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_line1,
            mode='lines',
            name=f'Diversity control median fitness',
            line=dict(color='rgb(255,0,0)')

        ))

        # Add the shaded area for standard deviation for Line 1
        fig.add_trace(go.Scatter(
            x=x_values + x_values[::-1],
            #y=[y - std_dev for y, std_dev in zip(y_values_line1, std_dev_line1)] +
            #[y + std_dev for y, std_dev in zip(y_values_line1[::-1], std_dev_line1[::-1])],
            y=max_values_line1+min_values_line1[::-1],
            fill='toself',
            fillcolor='rgba(217,121,123,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Std dev diversity'
        ))

        # Add the second line trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_line2,
            mode='lines',
            name=f'No diversity median fitness',
            line=dict(color='rgb(144,143,161)')
        ))

        # Add the shaded area for standard deviation for Line 2
        fig.add_trace(go.Scatter(
            x=x_values + x_values[::-1],
            #y=[y - std_dev for y, std_dev in zip(y_values_line2, std_dev_line2)] +
            #[y + std_dev for y, std_dev in zip(y_values_line2[::-1], std_dev_line2[::-1])],
            y=max_values_line2+min_values_line2[::-1],
            fill='toself',
            fillcolor='rgba(152,151,166,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Std dev no diversity'
        ))

        # Update layout for better visualization
        fig.update_layout(
            title='Population fitness per generation',
            xaxis_title='Generation',
            yaxis_title='Fitness',
            template="presentation", 
            showlegend=True
        )
        fig.update_layout(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
        return dcc.Graph(figure=fig, style={"width":"100%", "height":"100%"})

    ########################### Median plots similarity with shadows multiple selection #########################
    @app.callback(
        Output('std-median-plot-multi', 'children'),
        [Input("dropdown-3", "value"),
        Input("dropdown-4","value")],
        suppress_callback_exceptions=True
    )
    def std_plot_median_multi(drops, drops2):
        # Sample data
        if ((not drops) and (not drops2)):
            x_values=[]

        y_values_line1=[]
        #std_dev_line1=[]
        max_values_line1=[]
        min_values_line1=[]
        if drops:
            gens = len(df_list[drops[0]])
            x_values = list(np.arange(1,gens+1,1))

            for gen in x_values[:]:
                total_matrix=[]
                for drop in drops:
                    matrix = df_list[drop][gen-1].to_numpy()
                    upper_semidiagonal_mask = np.triu(np.ones(matrix.shape), k=1)
                    upper_semidiagonal = matrix[upper_semidiagonal_mask == 1]
                    #total_matrix.extend(list(upper_semidiagonal))
                    total_matrix.append(np.median(upper_semidiagonal))
                y_values_line1.append(np.median(total_matrix))
                max_values_line1.append(np.max(total_matrix))
                min_values_line1.append(np.min(total_matrix))
        #y_values_line1 = [np.median(mat) for mat in df_list[drop]]
        #std_dev_line1 = [1, 2, 1.5, 1, 3]
        y_values_line2=[]
        #std_dev_line1=[]
        max_values_line2=[]
        min_values_line2=[]
        if drops2:
            gens = len(df_list[drops2[0]])
            x_values = list(np.arange(1,gens+1,1))

            for gen in x_values[:]:
                total_matrix=[]
                for drop in drops2:
                    matrix = df_list[drop][gen-1].to_numpy()
                    upper_semidiagonal_mask = np.triu(np.ones(matrix.shape), k=1)
                    upper_semidiagonal = matrix[upper_semidiagonal_mask == 1]
                    #total_matrix.extend(list(upper_semidiagonal))
                    total_matrix.append(np.median(upper_semidiagonal))
                y_values_line2.append(np.mean(total_matrix))
                max_values_line2.append(np.max(total_matrix))
                min_values_line2.append(np.min(total_matrix))

        # Create a line plot
        fig = go.Figure()

        # Add the first line trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_line1,
            mode='lines',
            name=f'Diversity median similarity',
            line=dict(color='rgb(255,0,0)')
        ))

        # Add the shaded area for standard deviation for Line 1
        fig.add_trace(go.Scatter(
            x=x_values + x_values[::-1],
            #y=[y - std_dev for y, std_dev in zip(y_values_line1, std_dev_line1)] +
            #[y + std_dev for y, std_dev in zip(y_values_line1[::-1], std_dev_line1[::-1])],
            y=max_values_line1+min_values_line1[::-1],
            fill='toself',
            fillcolor='rgba(217,121,123,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Std dev diversity'
        ))

        # Add the second line trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values_line2,
            mode='lines',
            name=f'No diversity median similarity',
            line=dict(color='rgb(144,143,161)')
            
        ))

        # Add the shaded area for standard deviation for Line 2
        fig.add_trace(go.Scatter(
            x=x_values + x_values[::-1],
            #y=[y - std_dev for y, std_dev in zip(y_values_line2, std_dev_line2)] +
            #[y + std_dev for y, std_dev in zip(y_values_line2[::-1], std_dev_line2[::-1])],
            y=max_values_line2+min_values_line2[::-1],
            fill='toself',
            fillcolor='rgba(152,151,166,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Std dev No diversity'
        ))

        # Update layout for better visualization
        fig.update_layout(
            title='Model similarity per generation',
            xaxis_title='Generation',
            yaxis_title='similarity',
            template="presentation", 
            showlegend=True
        )
        fig.update_layout(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
        return dcc.Graph(figure=fig, style={"width":"100%", "height":"100%"})

    ########################### Max fitness plots similarity with shadows multiple selection #########################
    @app.callback(
        Output('max-fitness-plot-multi', 'children'),
        [Input("dropdown-3", "value"),
        Input("dropdown-4","value")],
        suppress_callback_exceptions=True
    )
    def max_fitness_multi(drops, drops2):
        # Sample data
        if ((not drops) and (not drops2)):
            x_values=[]

        max_values_line1=[]
        if drops:
            gens = len(df_list[drops[0]])
            x_values = list(np.arange(1,gens+1,1))

            for gen in x_values[:]:
                total_matrix=[]
                for drop in drops:
                    matrix = df_results_dict[drop][gen-1]
                    matrix=matrix[((matrix.inference_time!=np.inf) & (matrix.fitness!=-10001.0)) ]["fitness"].values
                    #total_matrix.extend(list(matrix))
                    total_matrix.append(np.max(matrix))
                max_values_line1.append(np.max(total_matrix))

        max_values_line2=[]
        if drops2:
            gens = len(df_list[drops2[0]])
            x_values = list(np.arange(1,gens+1,1))

            for gen in x_values[:]:
                total_matrix=[]
                for drop in drops2:
                    matrix = df_results_dict[drop][gen-1]
                    matrix=matrix[((matrix.inference_time!=np.inf) & (matrix.fitness!=-10001.0)) ]["fitness"].values
                    #total_matrix.extend(list(matrix))
                    total_matrix.append(np.max(matrix))
                max_values_line2.append(np.max(total_matrix))

        # Create a line plot
        fig = go.Figure()

        # Add the first line trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=max_values_line1,
            mode='lines',
            name=f'Diversity Max fitness',
            line=dict(color='rgb(255,0,0)')
        ))

        # Add the second line trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=max_values_line2,
            mode='lines',
            name=f'No diversity Max fitness',
            line=dict(color='rgb(144,143,161)')
            
        ))

        # Update layout for better visualization
        fig.update_layout(
            title='Max fitness per generation',
            xaxis_title='Generation',
            yaxis_title='fitness',
            template="presentation", 
            showlegend=True
        )
        fig.update_layout(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
        return dcc.Graph(figure=fig, style={"width":"100%", "height":"100%"})

