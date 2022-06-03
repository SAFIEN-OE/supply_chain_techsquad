import graph_pd
import pandas as pd
import geopandas as gpd
import simulator

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

model_in = "IN_Supply_Chain_Model.xlsx"

g = graph_pd.Graph()
g.populate_from_xlsx(model_in)

# risks = pd.read_excel(model_in, sheet_name = "Risks - Type", engine = 'openpyxl')
# g.compute_risk(risks)

# risks = pd.read_excel(model_in, sheet_name = "Risks - List", engine = 'openpyxl')
# g.compute_risk(risks, node_id_risk_col = 'ID', edge_id_risk_col = 'ID')

# lrisks = pd.read_excel('IN_Supply_Chain_Model.xlsx', sheet_name = "Risks - Location", engine = 'openpyxl')
# lrisks = gpd.GeoDataFrame(lrisks, crs = 'EPSG:4326')
# box_risks = graph_pd.Graph.geometry_from_points(lrisks, shape = 'Box', shape_column = 'Box or Circle', point_columns = ['Box 1 Lat', 'Box 1 Lon', 'Box 2 Lat', 'Box 2 Lon'])
# circle_risks = graph_pd.Graph.geometry_from_points(lrisks, shape = 'Circle', shape_column = 'Box or Circle', point_columns = ['Circle Lat', 'Circle Lon'])
# lrisks = box_risks.append(circle_risks)
# g.compute_location_risk(lrisks)

# g.compute_min_cost_flow(use_expected_capacity = True)

g.cut_edges(source=39, sink=26)
g.cut_edges(source=40, sink=39)
g.cut_edges(source=40, sink=41)

flow = [] 
flow.append({'source' : 10, 'sink' : 8, 'amount' : 3000})
flow.append({'source' : 2, 'sink' : 10, 'amount' : 1000})
print(g.get_nodes())
g = simulator.simulate(g, flow)
print(g.get_nodes())