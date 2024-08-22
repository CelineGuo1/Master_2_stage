import dash
from dash import dcc, html, Input, Output, State, callback
from dash import callback_context
import plotly.express as px
import scanpy as sc
from sklearn.metrics import silhouette_score
import pandas as pd
import cupy as cp 
import numpy as np 

import rapids_singlecell as rsc

import argparse
import os
import io

np.random.seed(123)
sc.settings.seed = 123

def process_data(file_path, subset_size=None):
    if os.path.isdir(file_path):
        adata = sc.read_10x_mtx(file_path, var_names="gene_symbols", cache=True)
        gene_family_prefix = "MT-"
    elif file_path.endswith('.h5ad'):
        adata = sc.read(file_path)
        gene_family_prefix = "mt-"
    elif file_path.endswith('.h5'):
        adata = sc.read_10x_h5(file_path)
        gene_family_prefix = "MT-"
    else:
        raise ValueError("Unsupported file format. Please provide a .h5ad/.h5 file or a directory containing .mtx files.")

    adata.var_names_make_unique()
    rsc.get.anndata_to_GPU(adata)
    
    if subset_size is None:
        subset_size = adata.shape[0]
    
    if subset_size:
        adata = adata[:subset_size, :].copy()
    
    rsc.pp.flag_gene_family(adata, 
                            gene_family_name="mt", 
                            gene_family_prefix=gene_family_prefix
                           )
    rsc.pp.calculate_qc_metrics(adata, 
                                qc_vars=["mt"], 
                                log1p=False
                               )
    rsc.pp.filter_cells(adata, qc_var='mt', min_count=200)
    rsc.pp.filter_genes(adata, min_count=3)

    df = adata.obs.copy()
    df['n_genes_by_counts'] = adata.obs['n_genes_by_counts']
    df['total_counts'] = adata.obs['total_counts']
    df['pct_counts_mt'] = adata.obs['pct_counts_mt']
    
    return adata, df
    

def create_violin_plots(df, n_genes_by_counts, pct_counts_mt):
    filtered_df = df[df['n_genes_by_counts'] < n_genes_by_counts]
    filtered_df = filtered_df[filtered_df['pct_counts_mt'] < pct_counts_mt]

    fig1 = px.violin(filtered_df, y='n_genes_by_counts', box=True, points=False, 
                     title='n_genes_by_counts', width=400, height=400
                    )
    fig1.update_traces(jitter=0.2, marker=dict(size=2, opacity=0.5), 
                       meanline_visible=True, line=dict(color='black', width=1)
                      )

    fig2 = px.violin(filtered_df, y='total_counts', box=True, points=False, 
                     title='total_counts', width=400, height=400
                    )
    fig2.update_traces(jitter=0.2, marker=dict(size=2, opacity=0.5), 
                       meanline_visible=True, line=dict(color='black', width=1)
                      )

    fig3 = px.violin(filtered_df, y='pct_counts_mt', box=True, points=False, 
                     title='pct_counts_mt', width=400, height=400
                    )
    fig3.update_traces(jitter=0.2, marker=dict(size=2, opacity=0.5), 
                       meanline_visible=True, line=dict(color='black', width=1)
                      )

    return fig1, fig2, fig3
    

def convert_df_to_tsv(df):
    buffer = io.StringIO()
    df.to_csv(buffer, sep='\t', index=False)
    buffer.seek(0)
    tsv_string = buffer.getvalue()
    buffer.close()
    return tsv_string

def save_anndata_to_h5ad(adata, file_path):
    adata.write(file_path)
    
    
def main():
    parser = argparse.ArgumentParser(description='Single cell analysis using Rapids single-cell.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input .h5ad or .mtx file')
    parser.add_argument('--subset_size', type=int, required = False, help='Number of cells to subset for analysis')
    
    args = parser.parse_args()
    
    adata, df = process_data(args.file_path, args.subset_size)

    # Créer les graphiques de violon initiaux
    fig1_initial = px.violin(df, y='n_genes_by_counts', box=True, points=False, 
                             title='n_genes_by_counts', width=400, height=400
                            )
    fig1_initial.update_traces(jitter=0.2, marker=dict(size=2, opacity=0.5), 
                               meanline_visible=True, line=dict(color='black', width=1)
                              )

    fig2_initial = px.violin(df, y='total_counts', box=True, points=False, 
                             title='total_counts', width=400, height=400
                            )
    fig2_initial.update_traces(jitter=0.2, marker=dict(size=2, opacity=0.5), 
                               meanline_visible=True, line=dict(color='black', width=1)
                              )

    fig3_initial = px.violin(df, y='pct_counts_mt', box=True, points=False, 
                             title='pct_counts_mt', width=400, height=400
                            )
    fig3_initial.update_traces(jitter=0.2, marker=dict(size=2, opacity=0.5), 
                               meanline_visible=True, line=dict(color='black', width=1)
                              )        

    app = dash.Dash(__name__)
    
    hvg_adata_filtered = None
    
    # Définir la mise en page de l'application
    app.layout = html.Div(children=[
        html.H1(children='Single Cell Analysis', style={'text-align': 'center'}),
    
        html.H2("Quality Control", style={'margin-top': '30px'}),
        html.Div(children=[
            dcc.Graph(figure=fig1_initial, style={'display': 'inline-block', 'width': '33%'}),
            dcc.Graph(figure=fig2_initial, style={'display': 'inline-block', 'width': '33%'}),
            dcc.Graph(figure=fig3_initial, style={'display': 'inline-block', 'width': '33%'})
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
    
        html.Div(children=[
            html.Label('n_genes_by_counts'),
            dcc.Input(id='input-n_genes_by_counts', type='number', placeholder='n_genes_by_counts', min=1),
            html.Label('pct_counts_mt'),
            dcc.Input(id='input-pct_counts_mt', type='number', placeholder='pct_counts_mt', min=1),
            html.Button(id='submit-button', n_clicks=0, children='Filter'),
        ], style={'margin-bottom': '20px', 'text-align': 'center'}),
    
        html.Div(id='output-violin-plots'),
        html.Div(id='cell-count-info'),
    
        html.H2("Highly Variable Genes (HVG) Selection"),
        html.Div(children=[
            html.Label('min_mean'),
            dcc.Input(id='input-min_mean', type='number', placeholder='min_mean', min=0),
            html.Label('max_mean'),
            dcc.Input(id='input-max_mean', type='number', placeholder='max_mean', min=0),
            html.Label('min_disp'),
            dcc.Input(id='input-min_disp', type='number', placeholder='min_disp', min=0),
            html.Button(id='hvg-button', n_clicks=0, children='Select HVGs'),
        ], style={'margin-bottom': '20px', 'text-align': 'center'}),
    
        html.Div(id='hvg-info'),
        dcc.Graph(id='hvg-scatter-plot', style={'height': '600px', 'width': '800px'}),
    
        html.H3("PCA and Elbow Plot"),
        html.Div(children=[
            dcc.Graph(id='pca-plot', style={'display': 'inline-block', 'height': '600px', 'width': '800px'}),
            dcc.Graph(id='elbow-plot', style={'display': 'inline-block', 'height': '600px', 'width': '800px'})
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
    
        html.H2("Leiden Clustering and UMAP"), 
        html.Div(children=[
            html.Label('N_neighbors'),
            dcc.Input(id='input-n_neighbors', type='number', placeholder='n_neighbors', min=1, step=1),
            html.Label('N_pcs'),
            dcc.Input(id='input-n_pcs', type='number', placeholder='n_pcs', min=1, step=1),
            html.Label('Leiden Resolution'),
            dcc.Input(id='input-leiden-resolution', type='number', placeholder='resolution', min=0.1, step=0.1),
            html.Button(id='leiden-button', n_clicks=0, children='Apply Leiden Clustering')
        ], style={'margin-bottom': '20px', 'text-align': 'center'}),
        
        html.Div(id='leiden-info'),
        html.Div(id='umap-plots'),
        html.Div(id='marker-genes'),
        html.Div(children=[
        html.Button(id='marker-button', n_clicks=0, children='Download Data and marker genes'),
        html.Label('Marker'),
        dcc.Input(id='input-marker', type='text', placeholder='marker'),
        html.Button(id='display-marker-button', n_clicks=0, children='Display Marker UMAP')
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),

        html.A(
            '',
            id='download-link',
            download="genes_markers.tsv",
            href="",
            target="_blank"
        ),
        html.Div(id='display-marker-umap')
    ])
  
    @app.callback(
        [Output('output-violin-plots', 'children'),
         Output('cell-count-info', 'children')],
        [Input('submit-button', 'n_clicks')],
        [State('input-n_genes_by_counts', 'value'),
         State('input-pct_counts_mt', 'value')]
    )
    
    def update_violin_plots(n_clicks, n_genes_by_counts, pct_counts_mt):
        if n_clicks > 0:
            initial_cell_count = len(df)
            
            filtered_df = df[df['n_genes_by_counts'] < n_genes_by_counts]
            filtered_df = filtered_df[filtered_df['pct_counts_mt'] < pct_counts_mt]
            
            filtered_cell_count = len(filtered_df)
    
            # Normaliser les données avec les paramètres de l'utilisateur
            adata_filtered = adata[adata.obs['n_genes_by_counts'] < n_genes_by_counts]
            adata_filtered = adata_filtered[adata_filtered.obs['pct_counts_mt'] < pct_counts_mt]
            rsc.pp.normalize_total(adata_filtered, target_sum=1e4)
    
            fig1, fig2, fig3 = create_violin_plots(filtered_df, n_genes_by_counts, pct_counts_mt)
    
            return (html.Div([
                        html.Div(dcc.Graph(figure=fig1), style={'display': 'inline-block', 'width': '33%'}),
                        html.Div(dcc.Graph(figure=fig2), style={'display': 'inline-block', 'width': '33%'}),
                        html.Div(dcc.Graph(figure=fig3), style={'display': 'inline-block', 'width': '33%'})
                    ]),
                    html.Div([
                        html.Div(f"Nombre de cellules au départ : {initial_cell_count}"),
                        html.Div(f"Nombre de cellules conservées : {filtered_cell_count}")
                    ]))
        else:
            return [], []
    
    # Callback pour la sélection des HVG et affichage des résultats
    @app.callback(
        [Output('hvg-info', 'children'),
         Output('hvg-scatter-plot', 'figure'),
         Output('pca-plot', 'figure'),
         Output('elbow-plot', 'figure')],
        [Input('hvg-button', 'n_clicks')],
        [State('input-n_genes_by_counts', 'value'),
         State('input-pct_counts_mt', 'value'),
         State('input-min_mean', 'value'),
         State('input-max_mean', 'value'),
         State('input-min_disp', 'value')]
    )
        
    def select_hvg(n_clicks, n_genes_by_counts, pct_counts_mt, min_mean, max_mean, min_disp):
        global hvg_adata_filtered
        if n_clicks > 0:
            # Filtrer et normaliser les données avec les paramètres de l'utilisateur
            adata_filtered = adata[adata.obs['n_genes_by_counts'] < n_genes_by_counts]
            adata_filtered = adata_filtered[adata_filtered.obs['pct_counts_mt'] < pct_counts_mt]
            rsc.pp.normalize_total(adata_filtered, target_sum=1e4)
            rsc.pp.log1p(adata_filtered)
            
            # Sélection des gènes hautement variables
            rsc.pp.highly_variable_genes(adata_filtered, 
                                         min_mean=min_mean, 
                                         max_mean=max_mean, 
                                         min_disp=min_disp,
                                         n_top_genes = None
                                        )
            
            # Calculer le nombre de gènes hautement variables
            hvg = adata_filtered.var[adata_filtered.var['highly_variable']]
            n_top_genes = hvg.shape[0]
    
            # Limiter à 5000 si n_top_genes dépasse cette valeur
            if n_top_genes > 5000:
                n_top_genes = 5000
            
            rsc.pp.highly_variable_genes(adata_filtered, 
                                         min_mean=min_mean, 
                                         max_mean=max_mean, 
                                         min_disp=min_disp,
                                         n_top_genes=n_top_genes
                                        )
            
            hvg = adata_filtered.var[adata_filtered.var['highly_variable']]
            other_genes = adata_filtered.var[~adata_filtered.var['highly_variable']]
            num_hvg = hvg.shape[0]
    
            hvg_df = pd.DataFrame({
                'mean': hvg['means'],
                'dispersion': hvg['dispersions'],
                'type': 'HVG'
            })
            other_genes_df = pd.DataFrame({
                'mean': other_genes['means'],
                'dispersion': other_genes['dispersions'],
                'type': 'Other'
            })
            combined_df = pd.concat([hvg_df, other_genes_df])
            
            hvg_scatter = px.scatter(combined_df, x='mean', y='dispersion', color='type', title='Highly Variable Genes', 
                                     labels={'mean': 'Mean expression', 'dispersion': 'Dispersion'},
                                     color_discrete_map={'HVG': 'red', 'Other': 'blue'})
            hvg_scatter.update_traces(marker=dict(size=5, opacity=0.5), line=dict(color='black', width=1))
    
            # Filtrage des gènes HVG et prétraitement
            adata_filtered = adata_filtered[:, adata_filtered.var.highly_variable]
            rsc.pp.regress_out(adata_filtered, ["total_counts", "pct_counts_mt"])
            rsc.pp.scale(adata_filtered, max_value=10)
    
            # Calcul de l'ACP
            rsc.pp.pca(adata_filtered, n_comps = 50)
            pca_df = pd.DataFrame(
                adata_filtered.obsm['X_pca'], 
                columns=[f'PC{i+1}' for i in range(adata_filtered.obsm['X_pca'].shape[1])]
            )
            pca_df['cell'] = adata_filtered.obs_names
    
            pca_plot = px.scatter(pca_df, x='PC1', y='PC2', title='PCA Plot')
            pca_plot.update_traces(marker=dict(size=5, opacity=0.5), line=dict(color='black', width=1))
    
            # Calcul des variances expliquées pour le Elbow plot
            var_ratio = adata_filtered.uns['pca']['variance']
            elbow_df = pd.DataFrame({'PC': [f'PC{i+1}' for i in range(len(var_ratio))], 'Variance Explained': var_ratio})
            elbow_plot = px.line(elbow_df, x='PC', y='Variance Explained', title='Elbow Plot')
            elbow_plot.update_traces(marker=dict(size=5, opacity=0.5), line=dict(color='black', width=1))

            hvg_adata_filtered = adata_filtered
    
            return (
                html.Div(f"Nombre de gènes hautement variables sélectionnés : {num_hvg}"),
                hvg_scatter,
                pca_plot,
                elbow_plot
            )
        else:
            return html.Div("Aucune sélection de gènes n'a encore été effectuée."), {}, {}, {}
    
    
    @app.callback(
    [Output('leiden-info', 'children'),
     Output('umap-plots', 'children'),
     Output('marker-genes', 'children'),
     Output('download-link', 'href'),
     Output('display-marker-umap', 'children')],
    [Input('leiden-button', 'n_clicks'),
     Input('marker-button', 'n_clicks'),
     Input('display-marker-button', 'n_clicks')],
    [State('input-leiden-resolution', 'value'),
     State('input-n_neighbors', 'value'),
     State('input-n_pcs', 'value'),
     State('input-marker', 'value')]
)
    def apply_leiden_clustering(n_clicks_leiden, n_clicks_marker, n_clicks_display_marker, leiden_resolution, n_neighbors, n_pcs, marker):
        global hvg_adata_filtered
    
        ctx = callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
        if button_id == 'leiden-button' and n_clicks_leiden > 0:
            rsc.pp.neighbors(hvg_adata_filtered, 
                             n_neighbors=n_neighbors, 
                             n_pcs=n_pcs)
            rsc.tl.umap(hvg_adata_filtered)
            rsc.tl.leiden(hvg_adata_filtered, 
                          resolution=leiden_resolution, 
                          random_state=0, 
                          n_iterations=100)
            
            silhouette_avg = silhouette_score(hvg_adata_filtered.obsm['X_pca'], hvg_adata_filtered.obs['leiden'])
            
            umap_df = pd.DataFrame(hvg_adata_filtered.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
            umap_df['Leiden_clusters'] = hvg_adata_filtered.obs['leiden'].astype(str)
            
            num_clusters = hvg_adata_filtered.obs['leiden'].nunique()
            
            cluster_colors = px.colors.qualitative.Alphabet
            color_map = {cluster: cluster_colors[i % len(cluster_colors)] for i, cluster in enumerate(sorted(umap_df['Leiden_clusters'].unique()))}
            
            umap_plot = px.scatter(umap_df, x='UMAP1', y='UMAP2', color=hvg_adata_filtered.obs['leiden'], color_discrete_map=color_map,
                                   title=f'UMAP avec clustering Leiden (Résolution={leiden_resolution})', template='plotly_white')
            
            umap_plot.update_layout(legend_title='Clusters', legend_traceorder='reversed')
            
            plots = [dcc.Graph(figure=umap_plot, style={'height': '600px', 'width': '800px'})]
    
            rsc.get.anndata_to_CPU(hvg_adata_filtered)
            sc.tl.rank_genes_groups(hvg_adata_filtered, "leiden", method="wilcoxon")
            result = hvg_adata_filtered.uns["rank_genes_groups"]
            groups = result["names"].dtype.names
            genes_markers = pd.DataFrame(
                {
                    group + "_" + key[:1]: result[key][group]
                    for group in groups
                    for key in ["names", "pvals"]
                }
            )
            
            marker_genes_output = html.Div([
                html.H2("Find Marker Genes"),
                html.Table([
                    html.Thead(
                        html.Tr([html.Th(col) for col in genes_markers.columns])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(genes_markers.iloc[i][col]) for col in genes_markers.columns
                        ]) for i in range(min(10, len(genes_markers)))
                    ]),
                ])
            ])
            
            return (
                html.Div([
                    html.Div(f"Nombre de Clusters : {num_clusters}"),
                    html.Div(f"Silhouette Score : {silhouette_avg:.3f}")
                ]),
                html.Div(plots),
                marker_genes_output,
                "",
                ""
            )
    
        elif button_id in ['marker-button', 'display-marker-button'] and (n_clicks_marker > 0 or n_clicks_display_marker > 0):
            rsc.pp.neighbors(hvg_adata_filtered, 
                             n_neighbors=n_neighbors, 
                             n_pcs=n_pcs)
            rsc.tl.umap(hvg_adata_filtered)
            rsc.tl.leiden(hvg_adata_filtered, 
                          resolution=leiden_resolution, 
                          random_state=0, 
                          n_iterations=100)
            rsc.get.anndata_to_CPU(hvg_adata_filtered)
            sc.tl.rank_genes_groups(hvg_adata_filtered, 
                                    "leiden", 
                                    method="wilcoxon")
            
            result = hvg_adata_filtered.uns["rank_genes_groups"]
            groups = result["names"].dtype.names
            genes_markers = pd.DataFrame(
                {
                    group + "_" + key[:1]: result[key][group]
                    for group in groups
                    for key in ["names", "pvals"]
                }
            )
            
            marker_genes_output = html.Div([
                html.H2("Find Marker Genes"),
                html.Table([
                    html.Thead(
                        html.Tr([html.Th(col) for col in genes_markers.columns])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(genes_markers.iloc[i][col]) for col in genes_markers.columns
                        ]) for i in range(min(10, len(genes_markers)))
                    ]),
                ])
            ])
            
            tsv_string = convert_df_to_tsv(genes_markers)
            filename = "genes_markers.tsv"
            with open(filename, "w") as file:
                file.write(tsv_string)
    
            file_path = 'filtered_data.h5ad'
            save_anndata_to_h5ad(hvg_adata_filtered, file_path)
            
            download_link = f"/download/{filename}"
            
            if button_id == 'display-marker-button' and marker:
                if marker in hvg_adata_filtered.var_names:
                    marker_values = hvg_adata_filtered[:, marker].X
                else:
                    available_markers = list(hvg_adata_filtered.var_names)
                    first_five_markers = available_markers[:5]
                    return (
                        "",
                        "",
                        "",
                        "",
                        html.Div([
                            html.Div(f"Erreur : Le marqueur '{marker}' n'existe pas. \
                            Les premiers marqueurs disponibles sont : {first_five_markers}")
                        ])
                    )
                
                umap_df = pd.DataFrame(hvg_adata_filtered.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
                umap_df['Marker'] = marker_values.flatten()
                
                umap_plot_marker = px.scatter(umap_df, x='UMAP1', y='UMAP2', color='Marker', title=f'UMAP avec clustering du marker {marker}',
                                              template='plotly_white', color_continuous_scale='Viridis')
                
                umap_plot_marker.update_layout(legend_title=marker, legend_traceorder='reversed')
                
                return (
                    "", 
                    "", 
                    marker_genes_output, 
                    download_link, 
                    dcc.Graph(figure=umap_plot_marker, style={'height': '600px', 'width': '800px'})
                )
            
            return "", "", marker_genes_output, download_link, ""
        
        return "", "", "", "", ""
            
    app.run_server(debug=True, port = '8053')

if __name__ == '__main__':
    main()