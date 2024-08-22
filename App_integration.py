import dash
from dash import dcc, html
import plotly.express as px
import scanpy as sc
import rapids_singlecell as rsc
import anndata as ad
import pandas as pd
import argparse
import os
import dash.dependencies as dd

def read_and_concat_h5ad_files(file_path):
    adata_list = []
    
    for filename in os.listdir(file_path):
        if filename.endswith(".h5ad"):
            filepath = os.path.join(file_path, filename)
            adata = sc.read_h5ad(filepath)
            adata_list.append(adata)
    
    concat_adata = ad.concat(adata_list,
                             join='outer', 
                             label='batch', 
                             keys=[os.path.splitext(f)[0] for f in os.listdir(file_path) if f.endswith('.h5ad')])
    
    concat_adata.obs_names_make_unique()
    
    return concat_adata

def generate_umap_plot(adata, color_by):
    umap_df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
    umap_df[color_by] = adata.obs[color_by].values
    
    cluster_colors = px.colors.qualitative.Plotly
    color_map = {val: cluster_colors[i % len(cluster_colors)] for i, val in enumerate(sorted(umap_df[color_by].unique()))}
    
    umap_plot = px.scatter(umap_df, x='UMAP1', y='UMAP2', color=color_by, color_discrete_map=color_map,
                           title=f'UMAP of Concatenated Data by {color_by}', template='plotly_white')
    
    umap_plot.update_layout(legend_title=color_by, legend_traceorder='reversed')
    
    return umap_plot

def main():
    parser = argparse.ArgumentParser(description='Data integration using Rapids single-cell.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input .h5ad files')
    
    args = parser.parse_args()
    
    global concat_adata
    concat_adata = read_and_concat_h5ad_files(args.file_path)

    # Define initial plots
    initial_umap_plot_batch = generate_umap_plot(concat_adata, 'batch')
    
    app = dash.Dash(__name__)
    
    app.layout = html.Div(children=[
        html.H1(children='Data Integration', style={'text-align': 'center'}),
    
        html.H2("UMAP merged data", style={'margin-top': '30px'}),
        html.Div(children=[
            dcc.Graph(id='umap-plot-batch', figure=initial_umap_plot_batch, style={'height': '600px', 'width': '800px'}),
            html.Button('Apply Data Integration', id='integrate-button', n_clicks=0),
            html.Div(id='additional-plots-container', children=[])
        ], style={'margin-bottom': '20px', 'text-align': 'center'})
    ])

    @app.callback(
        dd.Output('additional-plots-container', 'children'),
        [dd.Input('integrate-button', 'n_clicks')]
    )
    def update_plots(n_clicks):
        if n_clicks > 0:
            rsc.get.anndata_to_GPU(concat_adata)
            rsc.pp.harmony_integrate(concat_adata, 'batch')
            rsc.pp.neighbors(concat_adata, use_rep='X_pca')
            rsc.tl.umap(concat_adata)
            rsc.tl.leiden(concat_adata)
            
            # Generate new plots
            umap_plot_batch = generate_umap_plot(concat_adata, 'batch')
            umap_plot_leiden = generate_umap_plot(concat_adata, 'leiden')
            
            additional_plots = html.Div(children=[
                dcc.Graph(figure=umap_plot_batch,style={'height': '600px', 'width': '800px'}),
                dcc.Graph(figure=umap_plot_leiden,style={'height': '600px', 'width': '800px'})
            ], style={'display': 'flex', 'justify-content': 'space-between'})
            
            if 'Cell_type' in concat_adata.obs.columns:
                umap_plot_cell_type = generate_umap_plot(concat_adata, 'Cell_type')
                additional_plots.append(dcc.Graph(figure=umap_plot_cell_type,style={'height': '600px', 'width': '800px'}))
            
            return additional_plots
        
        return dash.no_update

    app.run_server(debug=True)

if __name__ == "__main__":
    main()
