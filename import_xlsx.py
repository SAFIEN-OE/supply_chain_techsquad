import pandas as pd
import json
from graph_api_rewrite import Node, Graph, Edge, Risk

filename = "IN_Supply_Chain_Model.xlsx"

g = pd.read_excel(filename, sheet_name = ["Nodes", "Edges", "Risks - Location", "Risks - Type", "Risks - List"], engine='openpyxl')

nodes_df = g['Nodes']
edges_df = g['Edges']
lrisks_df = g['Risks - Location']
trisks_df = g['Risks - Type']
lirisks_df = g['Risks - List']

nodes = []
edges = []
risks = []

for i, n in nodes_df.iterrows():
    nodes.append(Node(id = n['Node ID'], name = n['Node Name'], type = ['node', n['Type']], throughput=n['Throughput'], storage_capacity=n['Storage Capacity'], supply=n['Supply'], demand=n['Demand'],
                            current_storage=n['Current Storage'], resupply=(n['Resupply'] == 'yes'), location=(float(n['Latitude']), float(n['Longitude'])),
                            geometry=Graph.geometry_from_points(shape='point', points =[float(n['Latitude']), float(n['Longitude'])])))

for i, e in edges_df.iterrows():
    p1 = next((n for n in nodes if n.get_id() == e['Start Node']))
    p2 = next((n for n in nodes if n.get_id() == e['End Node']))
    edges.append(Edge(id=e['Edge ID'], start=e['Start Node'], end=e['End Node'], name=e['Edge Name'], type=['edge', e['Type']], capacity=e['Capacity'],
                            geometry=Graph.geometry_from_points(shape='line', points=[p1.get_location(), p2.get_location()])))

for i, lrisk in lrisks_df.iterrows():
    if lrisk['Box or Circle'] == 'Box':
        location = [(lrisk['Box 1 Lat'], lrisk['Box 1 Lon']), (lrisk['Box 2 Lat'], lrisk['Box 2 Lon'])]
        location.sort()
        location = [*location[0]] + [*location[1]]
        geometry = Graph.geometry_from_points(shape='box', points=location)
    else:
        location = [(lrisk['Circle Lat'], lrisk['Circle Lon']), (lrisk['Inner Distance'], lrisk['Outer Distance'])]
        geometry = Graph.geometry_from_points(shape='circle', points=location[0], outer_distance=location[1][1])
    risks.append(Risk(id=lrisk['Risk ID'], name=lrisk['Risk Name'], description=lrisk['Description'],
                        type=['risk', 'location', lrisk['Type']], shape = lrisk['Box or Circle'], 
                        probability=lrisk['Probability'], impact=lrisk['Impact'], location = location,
                        affected_objects = ['node', 'edge'], geometry=geometry))
                        
for i, trisk in trisks_df.iterrows():
    risks.append(Risk(id=trisk['Risk ID'], name=trisk['Risk Name'], type=('risk', 'type', trisk['Risk Type']),
                        affected_objects=[trisk['Edge or Node'].lower()],
                        target_types=trisk['Target Type'],
                        probability=trisk['Probability'], impact=trisk['Impact']))
                        
for i, lirisk in lirisks_df.iterrows():
    risks.append(Risk(id=lirisk['Risk ID'], name=lirisk['Risk Name'], description=lirisk['Risk Description'],
                        type=('risk', 'list', lirisk['Risk Type']),  affected_objects=[trisk['Edge or Node'].lower()],
                        target_ids = lirisk['Target ID'],
                        probability = lirisk['Probability'], impact = lirisk['Impact']))
                        

g = Graph(edges=edges, nodes=nodes, risks=risks)

g.cut_edges(start_id = 39, end_id = 26)
g.cut_edges(start_id = 42, end_id = 40)

g.export_json("IN_Supply_Chain_Model.json")