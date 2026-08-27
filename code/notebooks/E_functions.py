import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast

def generate_thesis_visualizations(csv_path, clip_outliers=True, area_col='area_meters_sq', quantile=0.99):
    """
    Loads the unified metrics CSV and generates insightful analytical plots.
    """
    print("📊 Loading data and prepping geometry...")
    df = pd.read_csv(csv_path)
    
    # Set dynamic labels based on your column choice
    if area_col == 'area_px':
        area_label = 'Ground Truth Area (pixels)'
    elif area_col == 'overlap_pixels':
        area_label = 'Prediction Overlap (pixels)'
    elif area_col == 'area_meters_sq':
        area_label = 'Ground Truth Area (m²)'

    # 1. Pre-processing
    df['is_artifact'] = df['is_artifact'].astype(bool)
    df['is_cut'] = df['is_cut'].astype(bool)
    
    def get_center_coords(bbox_str):
        if pd.isna(bbox_str): return pd.Series([np.nan, np.nan])
        try:
            bbox = ast.literal_eval(bbox_str)
            return pd.Series([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])
        except:
            return pd.Series([np.nan, np.nan])
            
    df[['center_x', 'center_y']] = df['bbox'].apply(get_center_coords)
    
    # Calculate 99th percentile for Area to prevent extreme outliers from squishing the linear plots
    max_area_plot = df[area_col].quantile(quantile) if clip_outliers else df[area_col].max()
    max_perim_plot = df['perimeter_meters'].quantile(quantile) if clip_outliers else df['perimeter_meters'].max()    
    sns.set_theme(style="whitegrid", palette="muted")
    
    # ==========================================
    # GRAPH 1: Area vs Number of Vertices
    # ==========================================
    print("📈 Generating Graph 1: Area vs Vertices...")
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(data=df, x=area_col, y='num_vertices', 
                    alpha=0.5, edgecolor=None, color='#2b8cbe')
    
    plt.title('Roof Complexity: Area vs. Number of Vertices', fontsize=14, fontweight='bold')
    plt.xlabel(area_label, fontsize=12)
    plt.ylabel('Number of Vertices', fontsize=12)
    plt.xscale('log')

    if clip_outliers:
        plt.xlim(right=max_area_plot)
        
    plt.tight_layout()
    plt.savefig('graph1_area_vs_vertices.png', dpi=300)
    plt.show()

    # ==========================================
    # GRAPH 2A: Impact of Artifacts on IoU
    # ==========================================
    print("📈 Generating Graph 2A: Artifacts vs IoU...")
    plt.figure(figsize=(8, 6))
    
    sns.violinplot(data=df, x='is_artifact', y='iou_score', 
                   palette=['#9ecae1', '#fc9272'], split=True, inner="quartile")
    plt.title('Impact of Artifacts on IoU Score', fontsize=14, fontweight='bold')
    plt.xlabel('Contains Artifact?', fontsize=12)
    plt.ylabel('IoU Score', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('graph2a_iou_vs_artifacts.png', dpi=300)
    plt.show()

    # ==========================================
    # GRAPH 2B: Impact of Cut Edges on IoU
    # ==========================================
    print("📈 Generating Graph 2B: Cut Edges vs IoU...")
    plt.figure(figsize=(8, 6))
    
    sns.violinplot(data=df, x='is_cut', y='iou_score', 
                   palette=['#a1d99b', '#bcbddc'], split=True, inner="quartile")
    plt.title('Impact of Tile Borders on IoU Score', fontsize=14, fontweight='bold')
    plt.xlabel('Cut by Tile Edge?', fontsize=12)
    plt.ylabel('IoU Score', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('graph2b_iou_vs_cuts.png', dpi=300)
    plt.show()

    # ==========================================
    # GRAPH 2C: Correlation: Area vs IoU
    # ==========================================
    print("📈 Generating Graph 2C: Area vs IoU...")
    plt.figure(figsize=(10, 6))
    
    # We use a regplot to overlay a linear regression trendline
    sns.regplot(data=df, x=area_col, y='iou_score', 
                scatter_kws={'alpha':0.3, 's':15}, line_kws={'color':'red', 'linewidth':2})
    
    plt.title(f'Correlation: Roof Area vs. Model Accuracy (IoU) (Quantile: {quantile})', fontsize=14, fontweight='bold')
    plt.xlabel(area_label, fontsize=12)
    plt.ylabel('IoU Score', fontsize=12)
    plt.xscale('log')
    
    # ✅ Limit the x-axis to the 99th percentile
    # We only set the 'right' limit because a log scale cannot start at 0
    if clip_outliers:
        plt.xlim(right=max_area_plot)
        
    plt.tight_layout()
    plt.savefig(f'graph2c_area_vs_iou_{quantile}.png', dpi=300)
    plt.show()

    # ==========================================
    # GRAPH 3: Spatial Heatmap of IoU Scores
    # ==========================================
    print("📈 Generating Graph 3: Spatial IoU Heatmap...")
    plt.figure(figsize=(10, 8))
    
    hm_df = df.dropna(subset=['center_x', 'center_y', 'iou_score'])
    
    hb = plt.hexbin(hm_df['center_x'], hm_df['center_y'], C=hm_df['iou_score'], 
                    gridsize=25, cmap='RdYlGn', reduce_C_function=np.mean, 
                    edgecolors='white', linewidths=0.2)
    
    cb = plt.colorbar(hb, label='Average IoU Score')
    plt.title('Spatial Heatmap of Average IoU Scores', fontsize=14, fontweight='bold')
    plt.xlabel('Geospatial X Coordinate', fontsize=12)
    plt.ylabel('Geospatial Y Coordinate', fontsize=12)
    
    plt.axis('equal') 
    plt.tight_layout()
    plt.savefig('graph3_spatial_heatmap.png', dpi=300)
    plt.show()
    print("✅ All visualizations generated and saved successfully!")

    # ==========================================
    # GRAPH 4: Enhanced Scatter: Complexity vs Performance
    # ==========================================
    # If you see a cluster of red dots (low IoU) floating at the top 
    # right of the graph, you can explicitly state: "SAM 2 struggles 
    # significantly with roofs that are both massive and highly polygonal, 
    # whereas small, simple polygons (bottom left) consistently achieve high IoU."

    print("📈 Generating Enhanced Graph 4: Complexity vs Performance...")
    plt.figure(figsize=(12, 8))
    
    # x = Area, y = Vertices
    # hue = IoU Score (Red to Green)
    # style = Is Artifact (Circles vs X's)
    sns.scatterplot(data=df, x=area_col, y='num_vertices', 
                    hue='iou_score', palette='RdYlGn', 
                    style='is_artifact', markers={False: 'o', True: 'X'},
                    alpha=0.7, s=50, edgecolor='w', linewidth=0.5)
    
    plt.title(f'Does Geometric Complexity Ruin Model Performance? (Quantile:{quantile})', fontsize=14, fontweight='bold')
    plt.xlabel(area_label, fontsize=12)
    plt.ylabel('Number of Vertices', fontsize=12)
    
    # Move the legend outside the plot so it doesn't cover data
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if clip_outliers:
        plt.xlim(0, max_area_plot)
        
    plt.tight_layout()
    plt.savefig(f'graph4_enhanced_scatter_{quantile}.png', dpi=300)
    plt.show()


    # ==========================================
    # GRAPH 5: Faceted Regression: Area vs IoU
    # ==========================================
    # This proves rigorous analytical thinking. It allows you to say: 
    # "When isolating whole roofs (is_cut=False), Area has a positive 
    # correlation with IoU. However, when a roof is cut by a tile edge, 
    # the model fails predictably regardless of area."
    print("📈 Generating Faceted Regression: Area vs IoU...")
    
    # col='is_cut' creates two side-by-side graphs
    # hue='is_artifact' colors the regression lines differently within those graphs
    g = sns.lmplot(data=df, x=area_col, y='iou_score', 
                   col='is_cut', hue='is_artifact', 
                   palette=['#2b8cbe', '#de2d26'], 
                   scatter_kws={'alpha': 0.3, 's': 15}, 
                   height=6, aspect=1.2)
    
    g.fig.suptitle(f'How Tile Borders and Artifacts Alter the Area/IoU Relationship (Quantile: {quantile})', 
                   fontsize=16, fontweight='bold', y=1.05)
                   
    g.set_axis_labels(area_label, "IoU Score")
    if clip_outliers:
        g.set(xlim=(0, max_area_plot))
        
    plt.savefig(f'graph5_faceted_regression_{quantile}.png', dpi=300)
    plt.show()


    # ==========================================
    # GRAPH 6: Joint Density Plot: Perimeter
    # ==========================================
    print("📈 Generating Joint Density Plot...")
    
    # kind='hex' shows density in the middle, marginal histograms on the sides
    g = sns.jointplot(data=df, x='perimeter_meters', y='iou_score', kind="kde", space=0)
    
    g.fig.suptitle(f'Density Analysis: Perimeter vs Accuracy (Quantile: {quantile})', fontweight='bold', y=1.03)
    g.set_axis_labels('Perimeter (m)', 'IoU Score')
    
    if clip_outliers:
        g.ax_joint.set_xlim(0, max_perim_plot)

    plt.savefig(f'graph6_jointplot_density_{quantile}.png', dpi=300)
    plt.show()

    # ==========================================
    # GRAPH 7: Pairplot Matrix
    # ==========================================
    # see how every numerical variable correlates with every other numerical variable
    print("📈 Generating Pairplot Matrix...")
    
    # Select only the crucial numeric columns to keep the plot readable
    columns_to_plot = [area_col, 'perimeter_meters', 'num_vertices', 'iou_score']
    
    # Plot the matrix, colored by whether it was successfully matched at all
    g = sns.pairplot(df, vars=columns_to_plot, hue='is_artifact', 
                     palette='husl', plot_kws={'alpha': 0.5, 's': 10},
                     corner=True) # corner=True hides the redundant top-right half of the grid
                     
    g.fig.suptitle('Multi-Collinearity and Performance Matrix', fontweight='bold', y=1.02)
    plt.savefig('graph7_pairplot_matrix.png', dpi=300)
    plt.show()

    # ==========================================
    # GRAPH 8: NEW Overlap Validation Plot
    # ==========================================
    print("📈 Generating Graph 8: Absolute Overlap vs IoU...")
    plt.figure(figsize=(10, 6))
    
    # Filter out false negatives (misses) so we only see actual predictions
    overlap_df = df[df['overlap_pixels'] > 0]
    
    sns.scatterplot(data=overlap_df, x='overlap_pixels', y='iou_score', 
                    alpha=0.4, color='#8856a7', s=20)
    
    plt.title('Prediction Confidence: Absolute Overlap vs. IoU Score', fontsize=14, fontweight='bold')
    plt.xlabel('Prediction Overlap (pixels)', fontsize=12)
    plt.ylabel('IoU Score', fontsize=12)
    
    # Add a horizontal line at 0.5 (standard IoU threshold for success)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='0.5 Success Threshold')
    plt.legend()
    
    if clip_outliers:
        max_overlap_plot = overlap_df['overlap_pixels'].quantile(quantile)
        plt.xlim(0, max_overlap_plot)
        
    plt.tight_layout()
    plt.savefig(f'graph8_overlap_vs_iou_{quantile}.png', dpi=300)
    plt.close()

# To run it:
# generate_thesis_visualizations('path/to/your/cantidio_sampaio_unified_metrics.csv')