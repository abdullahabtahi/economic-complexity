import vizro.actions as va
import vizro.models as vm
import vizro.plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from vizro import Vizro
from vizro.managers import data_manager
from vizro.models.types import capture
from vizro.figures import kpi_card
from vizro.tables import dash_ag_grid
import pandas as pd
import numpy as np
import scipy.linalg as la

#### DATA SETUP ####
countries=['Vale of Arryn','The North','The Riverlands', 'Dorne', 'The Stormland','The Westerlands','The Reach']
activities=[f'P{j}' for j in range(9)]
Mcp=np.array([[1, 0, 1, 0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1, 1]])

def compute_diversification(mcp): return mcp.sum(axis=1)
def compute_ubiquity(mcp): return mcp.sum(axis=0)

diversification = compute_diversification(Mcp)
ubiquity = compute_ubiquity(Mcp)

Kc0_inv = np.diag(1.0 / diversification)
Kp0_inv = np.diag(1.0 / ubiquity)
Mcc = Kc0_inv @ Mcp @ Kp0_inv @ Mcp.T
Mpp = Kp0_inv @ Mcp.T @ Kc0_inv @ Mcp

evals_c, evecs_c = la.eig(Mcc)
idx_c = evals_c.argsort()[::-1]
evecs_c = evecs_c[:, idx_c]
eci_raw = np.real(evecs_c[:, 1])
if np.corrcoef(diversification, eci_raw)[0, 1] < 0: eci_raw = -eci_raw
eci = (eci_raw - np.mean(eci_raw)) / np.std(eci_raw)

evals_p, evecs_p = la.eig(Mpp)
idx_p = evals_p.argsort()[::-1]
evecs_p = evecs_p[:, idx_p]
pci_raw = np.real(evecs_p[:, 1])
if np.corrcoef(ubiquity, pci_raw)[0, 1] > 0: pci_raw = -pci_raw
pci = (pci_raw - np.mean(pci_raw)) / np.std(pci_raw)

def compute_fitness_complexity(mcp, max_iter=1000, tol=1e-6):
    Fc = np.ones(mcp.shape[0])
    Qp = np.ones(mcp.shape[1])
    for i in range(max_iter):
        Fc_old = Fc.copy()
        Qp_old = Qp.copy()
        Fc_tilde = np.sum(mcp * Qp_old, axis=1)
        Qp_tilde = 1.0 / np.sum(mcp / (Fc_old[:, np.newaxis] + 1e-12), axis=0)
        Fc = Fc_tilde / np.mean(Fc_tilde)
        Qp = Qp_tilde / np.mean(Qp_tilde)
        if np.mean(np.abs(Fc - Fc_old)) < tol and np.mean(np.abs(Qp - Qp_old)) < tol: break
    return Fc, Qp

Fc, Qp = compute_fitness_complexity(Mcp)

def compute_proximity(mcp, ubiq):
    co = mcp.T @ mcp
    max_u = np.maximum.outer(ubiq, ubiq)
    prox = co / max_u
    np.fill_diagonal(prox, 1.0)
    return prox

proximity_matrix = compute_proximity(Mcp, ubiquity)

def compute_density(mcp, prox):
    p_no_self = prox.copy()
    np.fill_diagonal(p_no_self, 0.0)
    den = (mcp @ p_no_self) / p_no_self.sum(axis=0)
    den[mcp == 1] = np.nan
    return den

density_matrix = compute_density(Mcp, proximity_matrix)

df_countries = pd.DataFrame({
    'Kingdom': countries, 'Diversification': diversification,
    'ECI': eci.round(3), 'Fitness': Fc.round(3)
})

df_products = pd.DataFrame({
    'Product': activities, 'Ubiquity': ubiquity,
    'PCI': pci.round(3), 'Activity_Complexity': Qp.round(3)
})

df_raw = pd.DataFrame(Mcp, index=countries, columns=activities).reset_index()
df_raw.rename(columns={'index': 'Kingdom'}, inplace=True)

df_proximity = pd.DataFrame(proximity_matrix, index=activities, columns=activities).reset_index()
df_proximity.rename(columns={'index': 'Product'}, inplace=True)

df_density = pd.DataFrame(density_matrix, index=countries, columns=activities).reset_index()
df_density.rename(columns={'index': 'Kingdom'}, inplace=True)

data_manager["countries"] = df_countries
data_manager["products"] = df_products
data_manager["raw_matrix"] = df_raw

# Melted versions for heatmaps
df_prox_melted = df_proximity.melt(id_vars='Product', var_name='Target', value_name='Proximity')
df_den_melted = df_density.melt(id_vars='Kingdom', var_name='Product', value_name='Density')

order_c = np.argsort(diversification)[::-1]
order_p = np.argsort(pci)[::-1]
sorted_countries = [countries[i] for i in order_c]
sorted_products = [activities[i] for i in order_p]

df_prox_melted['Product'] = pd.Categorical(df_prox_melted['Product'], categories=sorted_products, ordered=True)
df_prox_melted['Target'] = pd.Categorical(df_prox_melted['Target'], categories=sorted_products, ordered=True)
df_den_melted['Kingdom'] = pd.Categorical(df_den_melted['Kingdom'], categories=sorted_countries, ordered=True)
df_den_melted['Product'] = pd.Categorical(df_den_melted['Product'], categories=sorted_products, ordered=True)

data_manager["proximity"] = df_prox_melted
data_manager["density"] = df_den_melted

# Network Graph setup
def build_relatedness_network(proximity: np.ndarray, products: list, extra_percentile: float = 75) -> nx.Graph:
    n = len(products)
    complete = nx.Graph()
    complete.add_nodes_from(products)
    for i in range(n):
        for j in range(i + 1, n):
            complete.add_edge(products[i], products[j], weight=proximity[i, j])

    graph = nx.maximum_spanning_tree(complete, weight="weight")

    off_diagonal = proximity[np.triu_indices(n, k=1)]
    threshold = np.percentile(off_diagonal, extra_percentile)
    for u, v, data in complete.edges(data=True):
        if data["weight"] >= threshold and not graph.has_edge(u, v):
            graph.add_edge(u, v, weight=data["weight"])

    return graph

relatedness_graph = build_relatedness_network(proximity_matrix, activities)
pos = nx.spring_layout(relatedness_graph, seed=42, k=0.9, weight="weight")

df_nodes = pd.DataFrame({
    'Product': activities,
    'x': [pos[p][0] for p in activities],
    'y': [pos[p][1] for p in activities],
    'PCI': pci
})

edge_x = []
edge_y = []
for u, v in relatedness_graph.edges():
    edge_x.extend([pos[u][0], pos[v][0], None])
    edge_y.extend([pos[u][1], pos[v][1], None])
df_edges = pd.DataFrame({'x': edge_x, 'y': edge_y})

data_manager["network_nodes"] = df_nodes

data_manager["kpi_kingdoms"] = pd.DataFrame([{"Value": len(countries)}])
data_manager["kpi_products"] = pd.DataFrame([{"Value": len(activities)}])
data_manager["kpi_div"] = pd.DataFrame([{"Value": max(diversification)}])

#### CUSTOM CHARTS ####
@capture("graph")
def custom_scatter_eci(data_frame, x, y, **kwargs):
    import plotly.express as px_standard
    fig = px_standard.scatter(data_frame, x=x, y=y, text='Kingdom', size_max=15, log_y=True, **kwargs)
    fig.update_traces(textposition='middle right')
    return fig

@capture("graph")
def custom_scatter_pci(data_frame, x, y, **kwargs):
    import plotly.express as px_standard
    fig = px_standard.scatter(data_frame, x=x, y=y, text='Product', size_max=15, log_y=True, **kwargs)
    fig.update_traces(textposition='middle right')
    return fig

@capture("graph")
def custom_heatmap(data_frame, x, y, color, **kwargs):
    import plotly.express as px_standard
    df_pivot = data_frame.pivot(index=y, columns=x, values=color)
    fig = px_standard.imshow(df_pivot, aspect="auto", **kwargs)
    return fig

@capture("graph")
def custom_network_graph(data_frame, **kwargs):
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=df_edges['x'], y=df_edges['y'],
        line=dict(width=2, color='#cccccc'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=data_frame['x'], y=data_frame['y'],
        mode='markers+text',
        text=data_frame['Product'],
        textposition="middle center",
        textfont=dict(color="black", size=10, family="Arial Black"),
        marker=dict(
            showscale=True,
            colorscale='RdYlBu_r',
            color=data_frame['PCI'],
            size=40,
            colorbar=dict(
                thickness=15,
                title=dict(text='PCI', side='right'),
                xanchor='left'
            ),
            line=dict(width=1.5, color='black')
        )
    ))
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        **kwargs
    )
    return fig


#### DASHBOARD PAGES ####
page_overview = vm.Page(
    title="Overview",
    layout=vm.Grid(grid=[
        [0, 1, 2],
        [3, 3, 3],
        [4, 5, 5]
    ]),
    components=[
        vm.Figure(id="kpi1", figure=kpi_card(data_frame="kpi_kingdoms", value_column="Value", value_format="{value}", title="Total Kingdoms", icon="public")),
        vm.Figure(id="kpi2", figure=kpi_card(data_frame="kpi_products", value_column="Value", value_format="{value}", title="Total Products", icon="inventory")),
        vm.Figure(id="kpi3", figure=kpi_card(data_frame="kpi_div", value_column="Value", value_format="{value}", title="Max Diversification", icon="trending_up")),
        vm.AgGrid(title="Raw Matrix (Mcp)", figure=dash_ag_grid(data_frame="raw_matrix")),
        vm.Graph(
            header="Click a kingdom to filter the dashboard",
            figure=px.bar("countries", x="Diversification", y="Kingdom", orientation='h'),
            actions=va.set_control(control="kingdom_filter", value="y")
        ),
        vm.Graph(figure=px.bar("products", x="Product", y="Ubiquity")),
    ],
    controls=[
        vm.Filter(id="kingdom_filter", column="Kingdom", selector=vm.Dropdown(multi=True))
    ]
)

page_complexity = vm.Page(
    title="Complexity Metrics",
    layout=vm.Grid(grid=[[0, 1]]),
    components=[
        vm.Graph(figure=custom_scatter_eci("countries", x="ECI", y="Fitness", title="Linear ECI vs Non-Linear Fitness")),
        vm.Graph(figure=custom_scatter_pci("products", x="PCI", y="Activity_Complexity", title="Linear PCI vs Non-Linear Activity Complexity")),
    ]
)

page_relatedness = vm.Page(
    title="Relatedness & Diversification",
    layout=vm.Grid(grid=[[0, 1]]),
    components=[
        vm.Graph(id="network", figure=custom_network_graph("network_nodes", title="The Product Space: Network of Related Activities")),
        vm.Graph(id="den", figure=custom_heatmap("density", x="Product", y="Kingdom", color="Density", title="Density Heatmap")),
    ]
)

page_strategy = vm.Page(
    title="Strategic Recommendations",
    layout=vm.Grid(grid=[[0]]),
    components=[
        vm.Card(
            text="""
### Economic Complexity Insights & Diversification Paths

Based on the **Harvard Growth Lab's** economic complexity framework, we can map out clear, actionable diversification strategies for the imaginary kingdoms. Growth is constrained by a kingdom's existing capabilities, so future investments must balance the complexity of new products against their relatedness (Density) to the kingdom's current industries.

#### 1. High Complexity, High Density (The Leaders)
Kingdoms like **The Reach** and **The Westerlands** already produce a highly diversified set of products. 
* **Recommendation:** Focus on frontier innovation. Transition into the few remaining high-PCI (Product Complexity Index) activities that they don't already dominate. 
* **Action:** Shift from basic industrial subsidies to aggressive R&D investments and specialized labor to maintain their competitive moat.

#### 2. Medium Complexity (The Transitioners)
Kingdoms like **The North** and **The Riverlands** show strong potential but lack the absolute diversification of the leaders.
* **Recommendation:** Look at the **Proximity Matrix**. They must target "bridge" products that are highly related to their current capabilities (high Density score) but that they do not yet produce.
* **Action:** Subsidize adjacent industries to naturally bridge the gap in their industrial capabilities and unlock new sectors of the economy.

#### 3. Low Complexity (The Laggards)
Kingdoms like **Dorne** rely heavily on a few ubiquitous products.
* **Recommendation:** Avoid attempting to jump directly into highly complex products, as the "distance" (lack of existing capabilities) is too great and risks market failure. 
* **Action:** Focus on capturing "low-hanging fruit", products with lower PCI but high relatedness to their existing capabilities. Build the industrial base incrementally.
            """
        )
    ]
)

dashboard = vm.Dashboard(
    title="Economic Complexity",
    pages=[page_overview, page_complexity, page_relatedness, page_strategy]
)

app = Vizro().build(dashboard)
server = app.dash.server

if __name__ == "__main__":
    app.run(port=8051, debug=True)
