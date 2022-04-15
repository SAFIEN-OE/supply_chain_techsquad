import graph_pd
import pandas as pd

model_in = "IN_Supply_Chain_Model copy.xlsx"

# Wrapper. Under the hood is just pandas dataframes
# TODO: Create more friendly interface for the graph (basically want to decouple pandas from class)
g = graph_pd.Graph()
g.populate_from_xlsx(model_in)

# Reads in risks and performs in-place updating of graph to account for risks
# TODO: Find better scheme for handling column naming in Risk files -- currently assuming matching names
#          in risks and edges/nodes refer to attributes that should be aligned for calculating risks
#           e.g., Type in risks is matched to Type in nodes to find which nodes to apply risk to
risks = pd.read_excel(model_in, sheet_name = "Risks - Type", engine='openpyxl')
g.compute_risk(risks)

risks = pd.read_excel(model_in, sheet_name = "Risks - List", engine='openpyxl')
g.compute_risk(risks)

g.compute_min_cost_flow()