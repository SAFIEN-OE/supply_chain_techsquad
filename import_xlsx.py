import pandas as pd
import json
from graph_api_rewrite import Node, Graph, Edge

filename = "IN_Supply_Chain_Model.xlsx"

g = pd.read_excel(filename, sheet_name = ["Nodes", "Edges"], engine='openpyxl')

nodes_df = g['Nodes']
edges_df = g['Edges']

nodes = []
edges = []

for i, n in nodes_df.iterrows():
    nodes.append(Node(id = n['Node ID'], name = n['Node Name'], type = ['node', n['Type']], throughput=n['Throughput'], storage_capacity=n['Storage Capacity'], supply=n['Supply'], demand=n['Demand'],
                            current_storage=n['Current Storage'], resupply=(n['Resupply'] == 'yes'), location=(float(n['Latitude']), float(n['Longitude']))))

for i, e in edges_df.iterrows():
    edges.append(Edge(id=e['Edge ID'], start=e['Start Node'], end=e['End Node'], name=e['Edge Name'], type=['edge', e['Type']], capacity=e['Capacity']))

g = Graph(edges=edges, nodes=nodes)

g.cut_edges(start_id = 39, end_id = 26)
g.cut_edges(start_id = 42, end_id = 40)

g.export_json("IN_Supply_Chain_Model.json")