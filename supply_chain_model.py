import pandas as pd
import numpy as np
import os
import random
from ortools.graph import pywrapgraph

df_nodes = pd.read_excel('IN_Supply_Chain_Model.xlsx', sheet_name = 'Nodes', engine='openpyxl')
df_edges = pd.read_excel('IN_Supply_Chain_Model.xlsx', sheet_name = 'Edges', engine='openpyxl')
df_risks_location = pd.read_excel('IN_Supply_Chain_Model.xlsx', sheet_name = 'Risks - Location', engine='openpyxl')
df_risks_type = pd.read_excel('IN_Supply_Chain_Model.xlsx', sheet_name = 'Risks - Type', engine='openpyxl')
df_risks_list = pd.read_excel('IN_Supply_Chain_Model.xlsx', sheet_name = 'Risks - List', engine='openpyxl')

df_nodes_orig = df_nodes.copy()
df_risks_location_orig = df_risks_location.copy()
df_risks_type_orig = df_risks_type.copy()
df_risks_list_orig = df_risks_list.copy()

df_nodes_gephi = df_nodes.copy()
df_nodes_gephi = df_nodes_gephi[['Node ID', 'Node Name', 'Latitude', 'Longitude']]
df_nodes_gephi = df_nodes_gephi.rename(columns = {'Node ID': 'Id', 'Node Name': 'Label'})

df_edges_gephi = df_edges.copy()
df_edges_gephi = df_edges_gephi[['Start Node', 'End Node', 'Capacity']]
df_edges_gephi = df_edges_gephi.rename(columns = {'Start Node': 'Source', 'End Node': 'Target',
                                                 'Capacity': 'Weight'})
df_edges_gephi['Type'] = 'Directed'

nodes_len = len(df_nodes)

# Assign the 'risks - type' value to each node
df_comb1 = pd.merge(df_nodes, df_risks_type, how = 'left', on = 'Type')
df_comb1['Expected_Impact'] = df_comb1['Probability'] * df_comb1['Impact']
df_comb1 = df_comb1[['Node ID', 'Expected_Impact']]
df_comb1 = df_comb1.groupby(['Node ID']).sum()

# Add the 'risks - type' values to df_nodes
df_nodes = pd.merge(df_nodes, df_comb1, how = 'left', on = 'Node ID')
df_nodes = df_nodes.rename(columns = {'Expected_Impact': 'Risks - Type'})

# Add column 'Node & ID' that combines the word 'Node' with the node ID in order to merge with df_risks_list
df_nodes['Node & ID'] = 'Node ' + df_nodes['Node ID'].astype(str)

# Add column 'Node/Edge & ID' that combines values from columns 'Edge or Node' and 'Node ID' to merge with df_nodes
df_risks_list['Node/Edge & ID'] = df_risks_list['Edge or Node'] + ' ' + df_risks_list['ID'].astype(str)

# Assign the 'risks - list' value to each node
df_comb1 = pd.merge(df_nodes, df_risks_list, how = 'left', left_on = 'Node & ID', right_on = 'Node/Edge & ID')
df_comb1 = df_comb1.fillna(0)
df_comb1['Expected_Impact'] = df_comb1['Probability'] * df_comb1['Impact']
df_comb1 = df_comb1[['Node ID', 'Expected_Impact']]
df_comb1 = df_comb1.groupby(['Node ID']).sum()

# Add the 'risks - list' values to df_nodes
df_nodes = pd.merge(df_nodes, df_comb1, how = 'left', on = 'Node ID')
df_nodes = df_nodes.rename(columns = {'Expected_Impact': 'Risks - List'})
df_nodes.pop('Node & ID')

# Add column 'Risk ID' to df_risks_location that gives a unique name to each risk
df_risks_location['Risk ID'] = df_risks_location['Box or Circle'] + '_' + df_risks_location.index.astype(str)

# Create lists for each risk in df_risks_location
gbl = globals()
threatlist = df_risks_location['Risk ID'].tolist()
for x in threatlist:
    gbl[x] = []

# Calculating the risk from each risk in df_risks_location and attaching to each node in df_nodes
for x in range(len(df_nodes)):
    for y, z in zip(range(len(df_risks_location)), threatlist):
        if  df_risks_location.loc[y, 'Box or Circle'] == 'Box':
            if ((df_nodes.loc[x, 'Latitude'] <=
                max(df_risks_location.loc[y, 'Box 1 Lat'], df_risks_location.loc[y, 'Box 2 Lat'])) &
                   (df_nodes.loc[x, 'Latitude'] >=
                min(df_risks_location.loc[y, 'Box 1 Lat'], df_risks_location.loc[y, 'Box 2 Lat'])) &
                    (df_nodes.loc[x, 'Longitude'] <=
                max(df_risks_location.loc[y, 'Box 1 Lon'], df_risks_location.loc[y, 'Box 2 Lon'])) &
                    (df_nodes.loc[x, 'Longitude'] >=
                min(df_risks_location.loc[y, 'Box 1 Lon'], df_risks_location.loc[y, 'Box 2 Lon']))):
                    gbl[z].append(df_risks_location.loc[y, 'Probability'] * df_risks_location.loc[y, 'Impact'])
            else:
                gbl[z].append(0)
        else:
            interva = (6371 * np.arccos(np.cos(np.radians(90 - df_nodes.loc[x, 'Latitude'])) *
                np.cos(np.radians(90 - df_risks_location.loc[y, 'Circle Lat'])) +
                np.sin(np.radians(90- df_nodes.loc[x, 'Latitude'])) *
                np.sin(np.radians(90 - df_risks_location.loc[y, 'Circle Lat'])) *
                np.cos(np.radians(df_nodes.loc[x, 'Longitude'] - df_risks_location.loc[y, 'Circle Lon']))))
            if ((interva >= df_risks_location.loc[y, 'Inner Distance']) & 
                (interva <= df_risks_location.loc[y, 'Outer Distance'])): 
                    gbl[z].append(df_risks_location.loc[y, 'Probability'] * df_risks_location.loc[y, 'Impact'])
            else:
                gbl[z].append(0)
                
for i in threatlist:
    df_nodes[i] = gbl[i]

# Add column 'Total Risks' that sums all the risks from df_risks_type, df_risks_list, df_risks_location
df_nodes['Total Risks'] = df_nodes[['Risks - Type', 'Risks - List'] + threatlist].sum(axis=1)
df_nodes.loc[df_nodes['Total Risks'] > 1, 'Total Risks'] = 1
df_nodes['Node ID'] = df_nodes.index + 1 

# df_nodes1 is nodes part 1 (black nodes) with supply and 0 demand
df_nodes1 = df_nodes.copy()
df_nodes1['Demand'] = 0

# df_nodes2 is nodes part 2 (blue nodes) with demand and 0 supply, current storage, storage capacity, total risks
df_nodes2 = df_nodes.copy()
df_nodes2['Node ID'] += nodes_len
df_nodes2['Node Name'] = df_nodes2['Node Name'] + ' Part 2'
df_nodes2['Supply'] = 0
df_nodes2['Storage Capacity'] = 0
df_nodes2['Current Storage'] = 0
df_nodes2['Total Risks'] = 0

# Reset df_nodes to be the concatenation of df_nodes1 and df_nodes2
df_nodes = pd.concat([df_nodes1, df_nodes2])
df_nodes = df_nodes.reset_index(drop = True)

# Create df_nodes_risks to use later in the code
df_nodes_risks = pd.merge(df_nodes_orig, df_nodes, how = 'left', on = 'Node ID')
df_nodes_risks = (df_nodes_risks[['Node ID', 'Node Name_x', 'Description_x', 'Type_x', 'Throughput_x',
                 'Storage Capacity_x', 'Supply_x', 'Demand_x', 'Current Storage_x', 'Resupply_x', 
                 'Latitude_x', 'Longitude_x', 'Total Risks']])
df_nodes_risks.columns = [x.replace('_x', '') for x in df_nodes_risks.columns]

# Create df_nodes_to_edges to use later in the code. This creates edges part 1 (blue edges).
df_nodes_to_edges = df_nodes_risks.copy()
df_nodes_to_edges = (df_nodes_to_edges.rename(columns = ({'Node ID': 'Start Node', 'Node Name': 'Edge Name', 
                                                         'Throughput': 'Capacity'})))
df_nodes_to_edges = df_nodes_to_edges[['Start Node', 'Edge Name', 'Capacity', 'Total Risks']]
df_nodes_to_edges['Edge Name'] = "NEW EDGE " + df_nodes_to_edges['Start Node'].astype(str)
df_nodes_to_edges['End Node'] = df_nodes_to_edges['Start Node'] + nodes_len

# Assign the 'risks - type' value to each edge
df_comb1 = pd.merge(df_edges, df_risks_type, how = 'left', on = 'Type')
df_comb1['Expected_Impact'] = df_comb1['Probability'] * df_comb1['Impact']
df_comb1 = df_comb1[['Edge ID', 'Expected_Impact']]
df_comb1 = df_comb1.groupby(['Edge ID']).sum()

# Add the 'risks - type' values to df_edges
df_edges = pd.merge(df_edges, df_comb1, how = 'left', on = 'Edge ID')
df_edges = df_edges.rename(columns = {'Expected_Impact': 'Risks - Type'})

# Add column 'Edge & ID' that combines the word 'Edge' with the edge ID in order to merge with df_risks_list
df_edges['Edge & ID'] = 'Edge ' + df_edges['Edge ID'].astype(str)

# Assign the 'risks - list' value to each edge
df_comb1 = pd.merge(df_edges, df_risks_list, how = 'left', left_on = 'Edge & ID', right_on = 'Node/Edge & ID')
df_comb1 = df_comb1.fillna(0)
df_comb1['Expected_Impact'] = (df_comb1['Probability'] * df_comb1['Impact'])
df_comb1 = df_comb1[['Edge ID', 'Expected_Impact']]
df_comb1 = df_comb1.groupby(['Edge ID']).sum()

# Add the 'risks - list' values to df_edges
df_edges = pd.merge(df_edges, df_comb1, how = 'left', on = 'Edge ID')
df_edges = df_edges.rename(columns = {'Expected_Impact': 'Risks - List'})
df_edges.pop('Edge & ID')

# Combine df_nodes and df_edges to eventually apply nodes' risks - locations to edges
df_comb1 = pd.merge(df_nodes, df_edges, how = 'right', left_on = 'Node ID', right_on = 'Start Node')

# Combine df_nodes and df_edges to eventually apply nodes' risks - locations to edges
df_comb2 = pd.merge(df_nodes, df_edges, how = 'right', left_on = 'Node ID', right_on = 'End Node')

# Combine comb1 and comb2 then filter to see total risks - location for each edge
df_comb3 = pd.concat([df_comb1, df_comb2])
df_comb3['Edge_Risks_Location'] = (df_comb3['Total Risks'] - df_comb3['Risks - Type_x'] - df_comb3['Risks - List_x'])
df_comb3 = df_comb3[['Edge ID', 'Edge_Risks_Location']]
df_comb3 = df_comb3.groupby(['Edge ID']).sum()

# Combine df_edges and df_comb3 and create 'Total Risks' column
df_edges = pd.merge(df_edges, df_comb3, how = 'inner', on = 'Edge ID')
df_edges = df_edges.rename(columns = {'Edge_Risks_Location': 'Risks - Location'})
df_edges['Total Risks'] = df_edges[['Risks - Type', 'Risks - List', 'Risks - Location']].sum(axis=1)
df_edges.loc[df_edges['Total Risks'] > 1, 'Total Risks'] = 1

# Set 'Start Node' to new nodes (part 2 blue nodes) to create edges part 2 (black edges)
df_edges['Start Node'] += nodes_len

# Combine df_edges and df_nodes_to_edges to get all black (part 1) and blue (part 2) edges together in df_edges
df_edges = pd.concat([df_edges, df_nodes_to_edges])
df_edges = df_edges.reset_index(drop = True)
df_edges['Edge ID'] = df_edges.index + 1

# Create df_edges_risks to use later in the code
df_edges_risks = df_edges.copy()

# Merge df_edges and df_nodes (twice) to become the new df_edges
df_edges = pd.merge(df_edges, df_nodes, how = 'left', left_on = 'Start Node', right_on = 'Node ID')
df_edges = pd.merge(df_edges, df_nodes, how = 'left', left_on = 'End Node', right_on = 'Node ID')
df_edges = (df_edges[['Edge ID', 'Start Node', 'End Node', 'Edge Name', 'Description_x', 'Type_x', 'Capacity', 
                      'Risks - Type_x', 'Risks - List_x', 'Risks - Location', 'Total Risks_x', 'Type_y', 'Type']])
df_edges = df_edges.rename(columns = {'Type_y': 'Start Node Type', 'Type': 'End Node Type'})
df_edges.columns = [x.replace('_x', '') for x in df_edges.columns]

# Create df_edges_copy to use later in the code
df_edges_copy = df_edges.copy()
df_edges_copy = df_edges_copy[['Start Node', 'End Node', 'Capacity', 'Total Risks']]
df_edges_copy = (df_edges_copy.rename(columns = {'Start Node': 'Start Node ID', 'End Node': 'End Node ID', 
                        'Capacity': 'End Node Capacity', 'Total Risks': 'Edge Risk'}))
df_edges_copy['End Node Expected Capacity'] = (df_edges_copy['End Node Capacity'] *
                        (1 - df_edges_copy['Edge Risk']))

# Create df_nodes_copy to use later in the code
df_nodes_copy = df_nodes.copy()
df_nodes_copy['Supply/Demand/Storage'] = (df_nodes_copy['Current Storage'] + df_nodes_copy['Supply'] 
                                        - df_nodes_copy['Demand'])
df_nodes_copy = df_nodes_copy.rename(columns = {'Node ID': 'Supply/Demand/Storage Node ID'})
df_nodes_copy = df_nodes_copy[['Supply/Demand/Storage Node ID', 'Supply/Demand/Storage']]

# Create empty lists to use in min cost flow calculation
start_nodes = []
end_nodes = []
capacity = []
expected_capacity = []
supplies = []
id_supplies = []
risk = []
remaining_capacity = []
remaining_expected_capacity = []

# Append lists in the correct format
for i in range(len(df_edges_copy)):
    start_nodes.append(int(df_edges_copy.loc[i, 'Start Node ID']))
    end_nodes.append(int(df_edges_copy.loc[i, 'End Node ID']))
    capacity.append(int(float(df_edges_copy.loc[i, 'End Node Capacity'])))
    expected_capacity.append(int(float(df_edges_copy.loc[i, 'End Node Expected Capacity'])))
    risk.append(int(1000 * float(df_edges_copy.loc[i, 'Edge Risk'])))

# Append lists in the correct format
for i in range(len(df_nodes_copy)):
    supplies.append(int(df_nodes_copy.loc[i, 'Supply/Demand/Storage']))
    id_supplies.append(int(df_nodes_copy.loc[i, 'Supply/Demand/Storage Node ID']))

# Baseline - part 1: Max flow with min cost with all node/edge risks equal to 0
flow = pywrapgraph.SimpleMinCostFlow()

# Add each arc
for i in range(len(start_nodes)):
    flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacity[i], 0)

# Add node supplies
for i in range(len(supplies)):
    flow.SetNodeSupply(int(id_supplies[i]), supplies[i])

start_node_output = []
end_node_output = []
flow_output = []
capacity_output = []
risk_output = []

start_node_output_v2 = []
end_node_output_v2 = []
flow_output_v2 = []
capacity_output_v2 = []
risk_output_v2 = []

# Find the minimum cost flow
if flow.SolveMaxFlowWithMinCost():
    for i in range(flow.NumArcs()):
        remaining_capacity.append(capacity[i] - flow.Flow(i))
        if start_nodes[i] > nodes_len:
            start_node_output.append(flow.Tail(i))
            end_node_output.append(flow.Head(i))
            flow_output.append(flow.Flow(i))
            capacity_output.append(flow.Capacity(i))
            risk_calc = (flow.UnitCost(i)/1000)
            risk_output.append(risk_calc)
        else:
            start_node_output_v2.append(flow.Tail(i))
            end_node_output_v2.append(flow.Head(i))
            flow_output_v2.append(flow.Flow(i))
            capacity_output_v2.append(flow.Capacity(i))
            risk_calc = (flow.UnitCost(i)/1000)
            risk_output_v2.append(risk_calc)

    total_flow_output_baseline_part1 = sum(flow_output) + sum(flow_output_v2)
    accum_risk_baseline_part1 = float(flow.OptimalCost()) / 1000
                
# Create df_baseline_part1 that includes flow along "true" edges (black edges part 2)
df_baseline_part1 = (pd.DataFrame(zip(start_node_output, end_node_output, flow_output, capacity_output, risk_output), 
                             columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))
df_baseline_part1['Start Node'] = df_baseline_part1['Start Node'] - nodes_len
df_baseline_part1['Edge ID'] = df_baseline_part1.index + 1

# Create df_baseline_part1_v2 that includes flow along "fake" edges (blue edges part 1)
df_baseline_part1_v2 = (pd.DataFrame(zip(start_node_output_v2, end_node_output_v2, flow_output_v2, capacity_output_v2, 
                   risk_output_v2), columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))

# Add baseline flow part 1 to df_nodes_final
df_nodes_final = (pd.merge(df_nodes_risks, df_baseline_part1_v2, how = 'left', left_on = 'Node ID', 
                           right_on = 'Start Node'))
df_nodes_final = df_nodes_final.drop(['Start Node', 'End Node', 'Capacity', 'Risk'], axis = 1)
df_nodes_final = df_nodes_final.rename(columns = {'Flow': 'Flow - Baseline part 1'})

# Create df_baseline_part1_gephi from df_baseline_part1 to format for csv export and gephi import
df_baseline_part1_gephi = df_baseline_part1[df_baseline_part1['Flow'] > 0]
df_baseline_part1_gephi = df_baseline_part1_gephi[['Start Node', 'End Node', 'Flow']]
df_baseline_part1_gephi['Start-End Node'] = (df_baseline_part1_gephi['Start Node'].astype(str) + ' - ' + 
                                             df_baseline_part1_gephi['End Node'].astype(str))

# Find the flow from each node
df_baseline_part1_start_node_flow = df_baseline_part1[['Start Node', 'Flow']]
df_baseline_part1_start_node_flow = df_baseline_part1_start_node_flow.groupby(['Start Node']).sum()
df_baseline_part1_start_node_flow.reset_index(inplace = True)

# Find the flow to each node
df_baseline_part1_end_node_flow = df_baseline_part1[['End Node', 'Flow']]
df_baseline_part1_end_node_flow = df_baseline_part1_end_node_flow.groupby(['End Node']).sum()
df_baseline_part1_end_node_flow.reset_index(inplace = True)

# Combine flow to/from each node
df_baseline_part1_net_flow = (pd.merge(df_baseline_part1_start_node_flow, df_baseline_part1_end_node_flow, 
                                how = 'outer', left_on = 'Start Node', right_on = 'End Node'))
df_baseline_part1_net_flow = df_baseline_part1_net_flow.fillna(0)
df_baseline_part1_net_flow['Node ID'] = df_baseline_part1_net_flow[['Start Node', 'End Node']].max(axis = 1)

# Calculate net flow from each node
df_baseline_part1_net_flow['Net Flow - Baseline part 1'] = (df_baseline_part1_net_flow['Flow_x'] - 
                                                            df_baseline_part1_net_flow['Flow_y'])
df_baseline_part1_net_flow = df_baseline_part1_net_flow[['Node ID', 'Net Flow - Baseline part 1']]

# Add columns in df_nodes_final and update values
df_nodes_final = pd.merge(df_nodes_final, df_baseline_part1_net_flow, how = 'outer', on = 'Node ID')
df_nodes_final['Remaining Storage - Baseline part 1'] = (df_nodes_final['Current Storage'] - 
                                                  df_nodes_final['Net Flow - Baseline part 1'])
df_nodes_final.loc[df_nodes_final['Remaining Storage - Baseline part 1'] < 0, 'Remaining Storage - Baseline part 1'] = 0
df_nodes_final.loc[df_nodes_final['Demand'] > 0, 'Remaining Storage - Baseline part 1'] = 0
df_nodes_final.loc[df_nodes_final['Demand'] > 0, 'Remaining Storage Capacity - Baseline part 1'] = 0
df_nodes_final['Remaining Storage Capacity - Baseline part 1'] = (df_nodes_final['Storage Capacity'] - 
                                                           df_nodes_final['Remaining Storage - Baseline part 1'])
df_nodes_final['New Supplies - Baseline'] = (df_nodes_final['Supply'] - 
                                             df_nodes_final['Net Flow - Baseline part 1'])
df_nodes_final.loc[df_nodes_final['New Supplies - Baseline'] < 0, 'New Supplies - Baseline'] = 0
df_nodes_final.loc[df_nodes_final['Resupply'] == 'No', 'New Supplies - Baseline'] = 0
df_nodes_final['New Demand - Baseline'] = df_nodes_final['Remaining Storage Capacity - Baseline part 1']
df_nodes_final['New Supply/Demand - Baseline'] = (df_nodes_final['New Supplies - Baseline'] - 
                                                  df_nodes_final['New Demand - Baseline'])

# Create df_new_supplies to capture supply from all new nodes (blue nodes part 2)
df_new_supplies = df_nodes_final.copy()
df_new_supplies = df_new_supplies[['Node ID', 'New Supply/Demand - Baseline']]
df_new_supplies['Node ID'] += nodes_len
df_new_supplies['New Supply/Demand - Baseline'] = 0

# Create df_new_supplies_all to find supply from all black (part 1) and blue (part 2) nodes
df_new_supplies_all = pd.concat([df_nodes_final, df_new_supplies])
df_new_supplies_all = df_new_supplies_all.reset_index(drop = True)

# Create empty lists to use in the second min cost flow calculation
new_supplies = []
new_id_supplies = []

# Append lists in the correct format
for i in range(len(df_new_supplies_all)):
    new_supplies.append(int(df_new_supplies_all.loc[i, 'New Supply/Demand - Baseline']))
    new_id_supplies.append(int(df_new_supplies_all.loc[i, 'Node ID']))

# Create df_new_inputs in order to create lists for second min cost flow calculation
list = ({'Start Nodes': start_nodes, 'End Nodes': end_nodes, 'Remaining Capacity': remaining_capacity, 
               'Risk': risk})
df_new_inputs = pd.DataFrame(list)

# Create empty lists to use in the second min cost flow calculation
start_nodes_2 = []
end_nodes_2 = []
remaining_capacity_2 = []
risk_2 = []

# Append lists in the correct format
for i in range(len(df_new_inputs)):
    start_nodes_2.append(int(df_new_inputs.loc[i, 'Start Nodes']))
    end_nodes_2.append(int(df_new_inputs.loc[i, 'End Nodes']))
    remaining_capacity_2.append(int(float(df_new_inputs.loc[i, 'Remaining Capacity'])))
    risk_2.append(int(float(df_new_inputs.loc[i, 'Risk'])))

# Baseline - part 2: Max flow with min cost with all node/edge risks equal to 0
flow = pywrapgraph.SimpleMinCostFlow()

# Add each arc
for i in range(len(start_nodes)):
    flow.AddArcWithCapacityAndUnitCost(start_nodes_2[i], end_nodes_2[i], remaining_capacity_2[i], 0)

# Add node supplies
for i in range(len(new_supplies)):
    flow.SetNodeSupply(int(new_id_supplies[i]), new_supplies[i])

start_node_output = []
end_node_output = []
flow_output = []
capacity_output = []
risk_output = []

start_node_output_v2 = []
end_node_output_v2 = []
flow_output_v2 = []
capacity_output_v2 = []
risk_output_v2 = []

# Find the minimum cost flow
if flow.SolveMaxFlowWithMinCost():
    for i in range(flow.NumArcs()):
        if start_nodes[i] > nodes_len:
            start_node_output.append(flow.Tail(i))
            end_node_output.append(flow.Head(i))
            flow_output.append(flow.Flow(i))
            capacity_output.append(flow.Capacity(i))
            risk_calc = (flow.UnitCost(i)/1000)
            risk_output.append(risk_calc)
        else:
            start_node_output_v2.append(flow.Tail(i))
            end_node_output_v2.append(flow.Head(i))
            flow_output_v2.append(flow.Flow(i))
            capacity_output_v2.append(flow.Capacity(i))
            risk_calc = (flow.UnitCost(i)/1000)
            risk_output_v2.append(risk_calc)

    total_flow_output_baseline_part2 = sum(flow_output) + sum(flow_output_v2)
    total_flow_output_baseline = total_flow_output_baseline_part1 + total_flow_output_baseline_part2
    accum_risk_baseline_part2 = float(flow.OptimalCost()) / 1000
    accum_risk_baseline = accum_risk_baseline_part1 + accum_risk_baseline_part2
    accum_risk_percent_baseline = accum_risk_baseline / total_flow_output_baseline
                
# Create df_baseline_part2 that includes flow along "true" nodes (black edges part 2)
df_baseline_part2 = (pd.DataFrame(zip(start_node_output, end_node_output, flow_output, capacity_output, risk_output), 
                             columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))
df_baseline_part2['Start Node'] = df_baseline_part2['Start Node'] - nodes_len
df_baseline_part2['Edge ID'] = df_baseline_part2.index + 1

# Create df_baseline_part2_v2 that includes flow along "fake" nodes (blue edges part 1)
df_baseline_part2_v2 = (pd.DataFrame(zip(start_node_output_v2, end_node_output_v2, flow_output_v2, capacity_output_v2, 
                   risk_output_v2), columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))

# Add baseline flow part 2 to df_nodes_final
df_nodes_final = (pd.merge(df_nodes_final, df_baseline_part2_v2, how = 'left', left_on = 'Node ID', 
                           right_on = 'Start Node'))
df_nodes_final = df_nodes_final.drop(['Start Node', 'End Node', 'Capacity', 'Risk'], axis = 1)
df_nodes_final = df_nodes_final.rename(columns = {'Flow': 'Flow - Baseline part 2'})

# Create df_baseline_part2_gephi from df_baseline_part2 to format for csv export and gephi import
df_baseline_part2_gephi = df_baseline_part2[df_baseline_part2['Flow'] > 0]
df_baseline_part2_gephi = df_baseline_part2_gephi[['Start Node', 'End Node', 'Flow']]
df_baseline_part2_gephi['Start-End Node'] = (df_baseline_part2_gephi['Start Node'].astype(str) + ' - ' + 
                                             df_baseline_part2_gephi['End Node'].astype(str))

# Create df_baseline_gephi to format for csv export and gephi import
df_baseline_gephi = pd.merge(df_baseline_part1_gephi, df_baseline_part2_gephi, how = 'outer', on = 'Start-End Node')
df_baseline_gephi = df_baseline_gephi.fillna(0)
df_baseline_gephi['Source'] = df_baseline_gephi[['Start Node_x', 'Start Node_y']].max(axis = 1)
df_baseline_gephi['Target'] = df_baseline_gephi[['End Node_x', 'End Node_y']].max(axis = 1)
df_baseline_gephi['Weight'] = df_baseline_gephi['Flow_x'] + df_baseline_gephi['Flow_y']
df_baseline_gephi = df_baseline_gephi[['Source', 'Target', 'Weight']]
df_baseline_gephi['Type'] = 'Directed'
convert_dict = {'Source': int, 'Target': int, 'Weight': int}
df_baseline_gephi = df_baseline_gephi.astype(convert_dict)

# Find the flow from each node
df_baseline_part2_start_node_flow = df_baseline_part2[['Start Node', 'Flow']]
df_baseline_part2_start_node_flow = df_baseline_part2_start_node_flow.groupby(['Start Node']).sum()
df_baseline_part2_start_node_flow.reset_index(inplace = True)

# Find the flow to each node
df_baseline_part2_end_node_flow = df_baseline_part2[['End Node', 'Flow']]
df_baseline_part2_end_node_flow = df_baseline_part2_end_node_flow.groupby(['End Node']).sum()
df_baseline_part2_end_node_flow.reset_index(inplace = True)

# Combine flow to/from each node
df_baseline_part2_net_flow = (pd.merge(df_baseline_part2_start_node_flow, df_baseline_part2_end_node_flow, 
                                how = 'outer', left_on = 'Start Node', right_on = 'End Node'))
df_baseline_part2_net_flow = df_baseline_part2_net_flow.fillna(0)
df_baseline_part2_net_flow['Node ID'] = df_baseline_part2_net_flow[['Start Node', 'End Node']].max(axis = 1)

# Calculate net flow from each node
df_baseline_part2_net_flow['Net Flow - Baseline part 2'] = (df_baseline_part2_net_flow['Flow_x'] - 
                                                            df_baseline_part2_net_flow['Flow_y'])
df_baseline_part2_net_flow = df_baseline_part2_net_flow[['Node ID', 'Net Flow - Baseline part 2']]

# Add columns in df_nodes_final and update values
df_nodes_final = pd.merge(df_nodes_final, df_baseline_part2_net_flow, how = 'outer', on = 'Node ID')
df_nodes_final['Total Flow - Baseline'] = (df_nodes_final['Flow - Baseline part 1'] + 
                                           df_nodes_final['Flow - Baseline part 2'])
df_nodes_final['Remaining Storage - Baseline'] = (df_nodes_final['Remaining Storage - Baseline part 1'] - 
                                                  df_nodes_final['Net Flow - Baseline part 2'])
df_nodes_final.loc[df_nodes_final['Storage Capacity'] == 0, 'Remaining Storage - Baseline'] = 0

# Scenario 1 - part 1: Max flow with min cost using actual capacity
flow = pywrapgraph.SimpleMinCostFlow()

# Add each arc
for i in range(len(start_nodes)):
    flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacity[i], risk[i])

# Add node supplies
for i in range(len(supplies)):
    flow.SetNodeSupply(int(id_supplies[i]), supplies[i])

remaining_capacity = []
    
start_node_output = []
end_node_output = []
flow_output = []
capacity_output = []
risk_output = []

start_node_output_v2 = []
end_node_output_v2 = []
flow_output_v2 = []
capacity_output_v2 = []
risk_output_v2 = []

# Find the minimum cost flow
if flow.SolveMaxFlowWithMinCost():
    for i in range(flow.NumArcs()):
        remaining_capacity.append(capacity[i] - flow.Flow(i))
        if start_nodes[i] > nodes_len:
            start_node_output.append(flow.Tail(i))
            end_node_output.append(flow.Head(i))
            flow_output.append(flow.Flow(i))
            capacity_output.append(flow.Capacity(i))
            risk_calc = (flow.UnitCost(i)/1000)
            risk_output.append(risk_calc)
        else:
            start_node_output_v2.append(flow.Tail(i))
            end_node_output_v2.append(flow.Head(i))
            flow_output_v2.append(flow.Flow(i))
            capacity_output_v2.append(flow.Capacity(i))
            risk_calc = (flow.UnitCost(i)/1000)
            risk_output_v2.append(risk_calc)
    
    total_flow_output_scenario1_part1 = sum(flow_output) + sum(flow_output_v2)
    accum_risk_scenario1_part1 = float(flow.OptimalCost()) / 1000

# Create df_scenario1_part1 that includes flow along "true" edges (black edges part 2)
df_scenario1_part1 = (pd.DataFrame(zip(start_node_output, end_node_output, flow_output, capacity_output, risk_output), 
                             columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))
df_scenario1_part1['Start Node'] = df_scenario1_part1['Start Node'] - nodes_len
df_scenario1_part1['Edge ID'] = df_scenario1_part1.index + 1
df_scenario1_part1.tail(50)

# Create df_scenario1_v2 that includes flow along "fake" nodes (blue edges part 1)
df_scenario1_part1_v2 = (pd.DataFrame(zip(start_node_output_v2, end_node_output_v2, flow_output_v2, capacity_output_v2, 
                   risk_output_v2), columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))

# Add scenario 1 flow to df_nodes_final
df_nodes_final = (pd.merge(df_nodes_final, df_scenario1_part1_v2, how = 'left', left_on = 'Node ID', 
                           right_on = 'Start Node'))
df_nodes_final = df_nodes_final.drop(['Start Node', 'End Node', 'Capacity', 'Risk'], axis = 1)
df_nodes_final = df_nodes_final.rename(columns = {'Flow': 'Flow - Scenario 1 part 1'})

# Create df_scenario1_part1_gephi from df_scenario1_part1 to format for csv export and gephi import
df_scenario1_part1_gephi = df_scenario1_part1[df_scenario1_part1['Flow'] > 0]
df_scenario1_part1_gephi = df_scenario1_part1_gephi[['Start Node', 'End Node', 'Flow']]
df_scenario1_part1_gephi['Start-End Node'] = (df_scenario1_part1_gephi['Start Node'].astype(str) + ' - ' + 
                                              df_scenario1_part1_gephi['End Node'].astype(str))

# Find the flow from each node
df_scenario1_part1_start_node_flow = df_scenario1_part1[['Start Node', 'Flow']]
df_scenario1_part1_start_node_flow = df_scenario1_part1_start_node_flow.groupby(['Start Node']).sum()
df_scenario1_part1_start_node_flow.reset_index(inplace = True)

# Find the flow to each node
df_scenario1_part1_end_node_flow = df_scenario1_part1[['End Node', 'Flow']]
df_scenario1_part1_end_node_flow = df_scenario1_part1_end_node_flow.groupby(['End Node']).sum()
df_scenario1_part1_end_node_flow.reset_index(inplace = True)

# Combine flow to/from each node
df_scenario1_part1_net_flow = (pd.merge(df_scenario1_part1_start_node_flow, df_scenario1_part1_end_node_flow, 
                               how = 'outer', left_on = 'Start Node', right_on = 'End Node'))
df_scenario1_part1_net_flow = df_scenario1_part1_net_flow.fillna(0)
df_scenario1_part1_net_flow['Node ID'] = df_scenario1_part1_net_flow[['Start Node', 'End Node']].max(axis = 1)

# Calculate net flow from each node
df_scenario1_part1_net_flow['Net Flow - Scenario 1 part 1'] = (df_scenario1_part1_net_flow['Flow_x'] - 
                                                               df_scenario1_part1_net_flow['Flow_y'])
df_scenario1_part1_net_flow = df_scenario1_part1_net_flow[['Node ID', 'Net Flow - Scenario 1 part 1']]

# Add columns in df_nodes_final and update values
df_nodes_final = pd.merge(df_nodes_final, df_scenario1_part1_net_flow, how = 'outer', on = 'Node ID')
df_nodes_final['Remaining Storage - Scenario 1 part 1'] = (df_nodes_final['Current Storage'] - 
                                                    df_nodes_final['Net Flow - Scenario 1 part 1'])
df_nodes_final.loc[df_nodes_final['Remaining Storage - Scenario 1 part 1'] < 0, 'Remaining Storage - Scenario 1 part 1'] = 0
df_nodes_final.loc[df_nodes_final['Demand'] > 0, 'Remaining Storage - Scenario 1 part 1'] = 0
df_nodes_final.loc[df_nodes_final['Demand'] > 0, 'Remaining Storage Capacity - Scenario 1 part 1'] = 0
df_nodes_final['Remaining Storage Capacity - Scenario 1 part 1'] = (df_nodes_final['Storage Capacity'] - 
                                                             df_nodes_final['Remaining Storage - Scenario 1 part 1'])
df_nodes_final['New Supplies - Scenario 1'] = (df_nodes_final['Supply'] - 
                                               df_nodes_final['Net Flow - Scenario 1 part 1'])
df_nodes_final.loc[df_nodes_final['New Supplies - Scenario 1'] < 0, 'New Supplies - Scenario 1'] = 0
df_nodes_final.loc[df_nodes_final['Resupply'] == 'No', 'New Supplies - Scenario 1'] = 0
df_nodes_final['New Demand - Scenario 1'] = df_nodes_final['Remaining Storage Capacity - Scenario 1 part 1']
df_nodes_final['New Supply/Demand - Scenario 1'] = (df_nodes_final['New Supplies - Scenario 1'] - 
                                                    df_nodes_final['New Demand - Scenario 1'])

# Create df_new_supplies to capture supply from all new nodes (blue nodes part 2)
df_new_supplies = df_nodes_final.copy()
df_new_supplies = df_new_supplies[['Node ID', 'New Supply/Demand - Scenario 1']]
df_new_supplies['Node ID'] += nodes_len
df_new_supplies['New Supply/Demand - Scenario 1'] = 0

# Create df_new_supplies_all to find supply from all black (part 1) and blue (part 2) nodes
df_new_supplies_all = pd.concat([df_nodes_final, df_new_supplies])
df_new_supplies_all = df_new_supplies_all.reset_index(drop = True)

# Create empty lists to use in the second min cost flow calculation
new_supplies = []
new_id_supplies = []

# Append lists in the correct format
for i in range(len(df_new_supplies_all)):
    new_supplies.append(int(df_new_supplies_all.loc[i, 'New Supply/Demand - Scenario 1']))
    new_id_supplies.append(int(df_new_supplies_all.loc[i, 'Node ID']))

# Create df_new_inputs in order to create lists for second min cost flow calculation
list = ({'Start Nodes': start_nodes, 'End Nodes': end_nodes, 'Remaining Capacity': remaining_capacity, 
               'Risk': risk})
df_new_inputs = pd.DataFrame(list)

# Create empty lists to use in the second min cost flow calculation
start_nodes_2 = []
end_nodes_2 = []
remaining_capacity_2 = []
risk_2 = []

# Append lists in the correct format
for i in range(len(df_new_inputs)):
    start_nodes_2.append(int(df_new_inputs.loc[i, 'Start Nodes']))
    end_nodes_2.append(int(df_new_inputs.loc[i, 'End Nodes']))
    remaining_capacity_2.append(int(float(df_new_inputs.loc[i, 'Remaining Capacity'])))
    risk_2.append(int(float(df_new_inputs.loc[i, 'Risk'])))

# Scenario 1 - part 2: Max flow with min cost using actual capacity
flow = pywrapgraph.SimpleMinCostFlow()

# Add each arc
for i in range(len(start_nodes)):
    flow.AddArcWithCapacityAndUnitCost(start_nodes_2[i], end_nodes_2[i], remaining_capacity_2[i], risk_2[i])

# Add node supplies
for i in range(len(new_supplies)):
    flow.SetNodeSupply(int(new_id_supplies[i]), new_supplies[i])
    
start_node_output = []
end_node_output = []
flow_output = []
capacity_output = []
risk_output = []

start_node_output_v2 = []
end_node_output_v2 = []
flow_output_v2 = []
capacity_output_v2 = []
risk_output_v2 = []

# Find the minimum cost flow
if flow.SolveMaxFlowWithMinCost():
    for i in range(flow.NumArcs()):
        if start_nodes[i] > nodes_len:
            start_node_output.append(flow.Tail(i))
            end_node_output.append(flow.Head(i))
            flow_output.append(flow.Flow(i))
            capacity_output.append(flow.Capacity(i))
            risk_calc = (flow.UnitCost(i)/1000)
            risk_output.append(risk_calc)
        else:
            start_node_output_v2.append(flow.Tail(i))
            end_node_output_v2.append(flow.Head(i))
            flow_output_v2.append(flow.Flow(i))
            capacity_output_v2.append(flow.Capacity(i))
            risk_calc = (flow.UnitCost(i)/1000)
            risk_output_v2.append(risk_calc)
    
    total_flow_output_scenario1_part2 = sum(flow_output) + sum(flow_output_v2)
    total_flow_output_scenario1 = total_flow_output_scenario1_part1 + total_flow_output_scenario1_part2
    accum_risk_scenario1_part2 = float(flow.OptimalCost()) / 1000
    accum_risk_scenario1 = accum_risk_scenario1_part1 + accum_risk_scenario1_part2
    accum_risk_percent_scenario1 = accum_risk_scenario1 / total_flow_output_scenario1

# Create df_scenario1_part2 that includes flow along "true" edges (black edges part 2)
df_scenario1_part2 = (pd.DataFrame(zip(start_node_output, end_node_output, flow_output, capacity_output, risk_output), 
                             columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))
df_scenario1_part2['Start Node'] = df_scenario1_part2['Start Node'] - nodes_len
df_scenario1_part2['Edge ID'] = df_scenario1_part2.index + 1
df_scenario1_part2.tail(50)

# Create df_scenario1_part2_v2 that includes flow along "fake" nodes (blue edges part 1)
df_scenario1_part2_v2 = (pd.DataFrame(zip(start_node_output_v2, end_node_output_v2, flow_output_v2, capacity_output_v2, 
                   risk_output_v2), columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))

# Add scenario 1 flow part 2 to df_nodes_final
df_nodes_final = (pd.merge(df_nodes_final, df_scenario1_part2_v2, how = 'left', left_on = 'Node ID', 
                           right_on = 'Start Node'))
df_nodes_final = df_nodes_final.drop(['Start Node', 'End Node', 'Capacity', 'Risk'], axis = 1)
df_nodes_final = df_nodes_final.rename(columns = {'Flow': 'Flow - Scenario 1 part 2'})

# Create df_scenario1_part2_gephi from df_scenario1_part2 to format for csv export and gephi import
df_scenario1_part2_gephi = df_scenario1_part2[df_scenario1_part2['Flow'] > 0]
df_scenario1_part2_gephi = df_scenario1_part2_gephi[['Start Node', 'End Node', 'Flow']]
df_scenario1_part2_gephi['Start-End Node'] = (df_scenario1_part2_gephi['Start Node'].astype(str) + ' - ' + 
                                              df_scenario1_part2_gephi['End Node'].astype(str))

# Create df_scenario1_gephi to format for csv export and gephi import
df_scenario1_gephi = pd.merge(df_scenario1_part1_gephi, df_scenario1_part2_gephi, how = 'outer', on = 'Start-End Node')
df_scenario1_gephi = df_scenario1_gephi.fillna(0)
df_scenario1_gephi['Source'] = df_scenario1_gephi[['Start Node_x', 'Start Node_y']].max(axis = 1)
df_scenario1_gephi['Target'] = df_scenario1_gephi[['End Node_x', 'End Node_y']].max(axis = 1)
df_scenario1_gephi['Weight'] = df_scenario1_gephi['Flow_x'] + df_scenario1_gephi['Flow_y']
df_scenario1_gephi = df_scenario1_gephi[['Source', 'Target', 'Weight']]
df_scenario1_gephi['Type'] = 'Directed'
convert_dict = {'Source': int, 'Target': int, 'Weight': int}
df_scenario1_gephi = df_scenario1_gephi.astype(convert_dict)

# Find the flow from each node

df_scenario1_part2_start_node_flow = df_scenario1_part2[['Start Node', 'Flow']]
df_scenario1_part2_start_node_flow = df_scenario1_part2_start_node_flow.groupby(['Start Node']).sum()
df_scenario1_part2_start_node_flow.reset_index(inplace = True)

# Find the flow to each node
df_scenario1_part2_end_node_flow = df_scenario1_part2[['End Node', 'Flow']]
df_scenario1_part2_end_node_flow = df_scenario1_part2_end_node_flow.groupby(['End Node']).sum()
df_scenario1_part2_end_node_flow.reset_index(inplace = True)

# Combine flow to/from each node
df_scenario1_part2_net_flow = (pd.merge(df_scenario1_part2_start_node_flow, df_scenario1_part2_end_node_flow, 
                                how = 'outer', left_on = 'Start Node', right_on = 'End Node'))
df_scenario1_part2_net_flow = df_scenario1_part2_net_flow.fillna(0)
df_scenario1_part2_net_flow['Node ID'] = df_scenario1_part2_net_flow[['Start Node', 'End Node']].max(axis = 1)

# Calculate net flow from each node
df_scenario1_part2_net_flow['Net Flow - Scenario 1 part 2'] = (df_scenario1_part2_net_flow['Flow_x'] - 
                                                               df_scenario1_part2_net_flow['Flow_y'])
df_scenario1_part2_net_flow = df_scenario1_part2_net_flow[['Node ID', 'Net Flow - Scenario 1 part 2']]

# Add columns in df_nodes_final and update values
df_nodes_final = pd.merge(df_nodes_final, df_scenario1_part2_net_flow, how = 'outer', on = 'Node ID')
df_nodes_final['Total Flow - Scenario 1'] = (df_nodes_final['Flow - Scenario 1 part 1'] + 
                                             df_nodes_final['Flow - Scenario 1 part 2'])
df_nodes_final['Remaining Storage - Scenario 1'] = (df_nodes_final['Remaining Storage - Scenario 1 part 1'] - 
                                                    df_nodes_final['Net Flow - Scenario 1 part 2'])
df_nodes_final.loc[df_nodes_final['Storage Capacity'] == 0, 'Remaining Storage - Scenario 1'] = 0

# Scenario 2 - part 1: Max flow with min cost using expected capacity
flow = pywrapgraph.SimpleMinCostFlow()

# Add each arc
for i in range(len(start_nodes)):
    flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], expected_capacity[i], risk[i])
    
# Add node supplies
for i in range(len(supplies)):
    flow.SetNodeSupply(int(id_supplies[i]), supplies[i])
    
start_node_output = []
end_node_output = []
flow_output = []
capacity_output = []
risk_output = []

start_node_output_v2 = []
end_node_output_v2 = []
flow_output_v2 = []
capacity_output_v2 = []
risk_output_v2 = []

# Find the minimum cost flow
if flow.SolveMaxFlowWithMinCost(): 
    for i in range(flow.NumArcs()):
        remaining_expected_capacity.append(capacity[i] - flow.Flow(i))
        if start_nodes[i] > nodes_len:
            start_node_output.append(flow.Tail(i))
            end_node_output.append(flow.Head(i))
            flow_output.append(flow.Flow(i))
            capacity_output.append(flow.Capacity(i))
            risk_calc = (flow.UnitCost(i)/1000)
            risk_output.append(risk_calc)
        else:
            start_node_output_v2.append(flow.Tail(i))
            end_node_output_v2.append(flow.Head(i))
            flow_output_v2.append(flow.Flow(i))
            capacity_output_v2.append(flow.Capacity(i))
            risk_calc = (flow.UnitCost(i)/1000)
            risk_output_v2.append(risk_calc)
    
    total_flow_output_scenario2_part1 = sum(flow_output) + sum(flow_output_v2)
    accum_risk_scenario2_part1 = float(flow.OptimalCost()) / 1000

# Create df_scenario2_part1 that includes flow along "true" nodes (black edges part 2)
df_scenario2_part1 = (pd.DataFrame(zip(start_node_output, end_node_output, flow_output, capacity_output, risk_output), 
                             columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))
df_scenario2_part1['Start Node'] = df_scenario2_part1['Start Node'] - nodes_len
df_scenario2_part1['Edge ID'] = df_scenario2_part1.index + 1

# Create df_scenario2_part1_v2 that includes flow along "fake" nodes (blue edges part 1)
df_scenario2_part1_v2 = (pd.DataFrame(zip(start_node_output_v2, end_node_output_v2, flow_output_v2, capacity_output_v2, 
                   risk_output_v2), columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))

# Add scenario 2 flow to df_nodes_final

df_nodes_final = (pd.merge(df_nodes_final, df_scenario2_part1_v2, how = 'left', left_on = 'Node ID', 
                           right_on = 'Start Node'))
df_nodes_final = df_nodes_final.drop(['Start Node', 'End Node', 'Capacity', 'Risk'], axis = 1)
df_nodes_final = df_nodes_final.rename(columns = {'Flow': 'Flow - Scenario 2 part 1'})

# Create df_scenario2_part1_gephi from df_scenario2_part1 to format for csv export and gephi import
df_scenario2_part1_gephi = df_scenario2_part1[df_scenario2_part1['Flow'] > 0]
df_scenario2_part1_gephi = df_scenario2_part1_gephi[['Start Node', 'End Node', 'Flow']]
df_scenario2_part1_gephi['Start-End Node'] = (df_scenario2_part1_gephi['Start Node'].astype(str) + ' - ' + 
                                              df_scenario2_part1_gephi['End Node'].astype(str))

# Find the flow from each node
df_scenario2_part1_start_node_flow = df_scenario2_part1[['Start Node', 'Flow']]
df_scenario2_part1_start_node_flow = df_scenario2_part1_start_node_flow.groupby(['Start Node']).sum()
df_scenario2_part1_start_node_flow.reset_index(inplace = True)

# Find the flow to each node
df_scenario2_part1_end_node_flow = df_scenario2_part1[['End Node', 'Flow']]
df_scenario2_part1_end_node_flow = df_scenario2_part1_end_node_flow.groupby(['End Node']).sum()
df_scenario2_part1_end_node_flow.reset_index(inplace = True)

# Combine flow to/from each node
df_scenario2_part1_net_flow = (pd.merge(df_scenario2_part1_start_node_flow, df_scenario2_part1_end_node_flow, 
                               how = 'outer', left_on = 'Start Node', right_on = 'End Node'))
df_scenario2_part1_net_flow = df_scenario2_part1_net_flow.fillna(0)
df_scenario2_part1_net_flow['Node ID'] = df_scenario2_part1_net_flow[['Start Node', 'End Node']].max(axis = 1)

# Calculate net flow from each node
df_scenario2_part1_net_flow['Net Flow - Scenario 2 part 1'] = (df_scenario2_part1_net_flow['Flow_x'] - 
                                                               df_scenario2_part1_net_flow['Flow_y'])
df_scenario2_part1_net_flow = df_scenario2_part1_net_flow[['Node ID', 'Net Flow - Scenario 2 part 1']]

# Add columns in df_nodes_final and update values
df_nodes_final = pd.merge(df_nodes_final, df_scenario2_part1_net_flow, how = 'outer', on = 'Node ID')
df_nodes_final['Remaining Storage - Scenario 2 part 1'] = (df_nodes_final['Current Storage'] - 
                                                    df_nodes_final['Net Flow - Scenario 2 part 1'])
df_nodes_final.loc[df_nodes_final['Remaining Storage - Scenario 2 part 1'] < 0, 'Remaining Storage - Scenario 2 part 1'] = 0
df_nodes_final.loc[df_nodes_final['Demand'] > 0, 'Remaining Storage - Scenario 2 part 1'] = 0
df_nodes_final.loc[df_nodes_final['Demand'] > 0, 'Remaining Storage Capacity - Scenario 2 part 1'] = 0
df_nodes_final['Remaining Storage Capacity - Scenario 2 part 1'] = (df_nodes_final['Storage Capacity'] - 
                                                             df_nodes_final['Remaining Storage - Scenario 2 part 1'])
df_nodes_final['New Supplies - Scenario 2'] = (df_nodes_final['Supply'] - 
                                               df_nodes_final['Net Flow - Scenario 2 part 1'])
df_nodes_final.loc[df_nodes_final['New Supplies - Scenario 2'] < 0, 'New Supplies - Scenario 2'] = 0
df_nodes_final.loc[df_nodes_final['Resupply'] == 'No', 'New Supplies - Scenario 2'] = 0
df_nodes_final['New Demand - Scenario 2'] = df_nodes_final['Remaining Storage Capacity - Scenario 2 part 1']
df_nodes_final['New Supply/Demand - Scenario 2'] = (df_nodes_final['New Supplies - Scenario 2'] - 
                                                    df_nodes_final['New Demand - Scenario 2'])

# Create df_new_supplies to capture supply from all new nodes (blue nodes part 2)
df_new_supplies = df_nodes_final.copy()
df_new_supplies = df_new_supplies[['Node ID', 'New Supply/Demand - Scenario 2']]
df_new_supplies['Node ID'] += nodes_len
df_new_supplies['New Supply/Demand - Scenario 2'] = 0

# Create df_new_supplies_all to find supply from all black (part 1) and blue (part 2) nodes
df_new_supplies_all = pd.concat([df_nodes_final, df_new_supplies])
df_new_supplies_all = df_new_supplies_all.reset_index(drop = True)

# Create empty lists to use in the second min cost flow calculation
new_supplies = []
new_id_supplies = []

# Append lists in the correct format
for i in range(len(df_new_supplies_all)):
    new_supplies.append(int(df_new_supplies_all.loc[i, 'New Supply/Demand - Scenario 2']))
    new_id_supplies.append(int(df_new_supplies_all.loc[i, 'Node ID']))

# Create df_new_inputs in order to create lists for second min cost flow calculation
list = ({'Start Nodes': start_nodes, 'End Nodes': end_nodes, 
         'Remaining Expected Capacity': remaining_expected_capacity, 'Risk': risk})
df_new_inputs = pd.DataFrame(list)

# Create empty lists to use in the second min cost flow calculation
start_nodes_2 = []
end_nodes_2 = []
remaining_expected_capacity_2 = []
risk_2 = []

# Append lists in the correct format
for i in range(len(df_new_inputs)):
    start_nodes_2.append(int(df_new_inputs.loc[i, 'Start Nodes']))
    end_nodes_2.append(int(df_new_inputs.loc[i, 'End Nodes']))
    remaining_expected_capacity_2.append(int(float(df_new_inputs.loc[i, 'Remaining Expected Capacity'])))
    risk_2.append(int(float(df_new_inputs.loc[i, 'Risk'])))

# Scenario 2 - part 2: Max flow with min cost using expected capacity
flow = pywrapgraph.SimpleMinCostFlow()

# Add each arc
for i in range(len(start_nodes)):
    flow.AddArcWithCapacityAndUnitCost(start_nodes_2[i], end_nodes_2[i], remaining_expected_capacity_2[i], risk_2[i])
    
# Add node supplies
for i in range(len(supplies)):
    flow.SetNodeSupply(int(new_id_supplies[i]), new_supplies[i])
    
start_node_output = []
end_node_output = []
flow_output = []
capacity_output = []
risk_output = []

start_node_output_v2 = []
end_node_output_v2 = []
flow_output_v2 = []
capacity_output_v2 = []
risk_output_v2 = []

# Find the minimum cost flow
if flow.SolveMaxFlowWithMinCost(): 
    for i in range(flow.NumArcs()):
        if start_nodes[i] > nodes_len:
            start_node_output.append(flow.Tail(i))
            end_node_output.append(flow.Head(i))
            flow_output.append(flow.Flow(i))
            capacity_output.append(flow.Capacity(i))
            risk_calc = (flow.UnitCost(i)/1000)
            risk_output.append(risk_calc)
        else:
            start_node_output_v2.append(flow.Tail(i))
            end_node_output_v2.append(flow.Head(i))
            flow_output_v2.append(flow.Flow(i))
            capacity_output_v2.append(flow.Capacity(i))
            risk_calc = (flow.UnitCost(i)/1000)
            risk_output_v2.append(risk_calc)
    
    total_flow_output_scenario2_part2 = sum(flow_output) + sum(flow_output_v2)
    total_flow_output_scenario2 = total_flow_output_scenario2_part1 + total_flow_output_scenario2_part2
    accum_risk_scenario2_part2 = float(flow.OptimalCost()) / 1000
    accum_risk_scenario2 = accum_risk_scenario2_part1 + accum_risk_scenario2_part2
    accum_risk_percent_scenario2 = accum_risk_scenario2 / total_flow_output_scenario2

# Create df_scenario2_part2 that includes flow along "true" nodes (black edges part 2)
df_scenario2_part2 = (pd.DataFrame(zip(start_node_output, end_node_output, flow_output, capacity_output, risk_output), 
                             columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))
df_scenario2_part2['Start Node'] = df_scenario2_part2['Start Node'] - nodes_len
df_scenario2_part2['Edge ID'] = df_scenario2_part2.index + 1

# Create df_scenario2_part2_v2 that includes flow along "fake" nodes (blue edges part 1)
df_scenario2_part2_v2 = (pd.DataFrame(zip(start_node_output_v2, end_node_output_v2, flow_output_v2, capacity_output_v2, 
                   risk_output_v2), columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))

# Add scenario 2 flow part 2 to df_nodes_final
df_nodes_final = (pd.merge(df_nodes_final, df_scenario2_part2_v2, how = 'left', left_on = 'Node ID', 
                           right_on = 'Start Node'))
df_nodes_final = df_nodes_final.drop(['Start Node', 'End Node', 'Capacity', 'Risk'], axis = 1)
df_nodes_final = df_nodes_final.rename(columns = {'Flow': 'Flow - Scenario 2 part 2'})

# Create df_scenario2_part2_gephi from df_scenario2_part2 to format for csv export and gephi import
df_scenario2_part2_gephi = df_scenario2_part2[df_scenario2_part2['Flow'] > 0]
df_scenario2_part2_gephi = df_scenario2_part2_gephi[['Start Node', 'End Node', 'Flow']]
df_scenario2_part2_gephi['Start-End Node'] = (df_scenario2_part2_gephi['Start Node'].astype(str) + ' - ' + 
                                              df_scenario2_part2_gephi['End Node'].astype(str))

# Create df_scenario2_gephi to format for csv export and gephi import
df_scenario2_gephi = pd.merge(df_scenario2_part1_gephi, df_scenario2_part2_gephi, how = 'outer', on = 'Start-End Node')
df_scenario2_gephi = df_scenario2_gephi.fillna(0)
df_scenario2_gephi['Source'] = df_scenario2_gephi[['Start Node_x', 'Start Node_y']].max(axis = 1)
df_scenario2_gephi['Target'] = df_scenario2_gephi[['End Node_x', 'End Node_y']].max(axis = 1)
df_scenario2_gephi['Weight'] = df_scenario2_gephi['Flow_x'] + df_scenario2_gephi['Flow_y']
df_scenario2_gephi = df_scenario2_gephi[['Source', 'Target', 'Weight']]
df_scenario2_gephi['Type'] = 'Directed'
convert_dict = {'Source': int, 'Target': int, 'Weight': int}
df_scenario2_gephi = df_scenario2_gephi.astype(convert_dict)

# Find the flow from each node
df_scenario2_part2_start_node_flow = df_scenario2_part2[['Start Node', 'Flow']]
df_scenario2_part2_start_node_flow = df_scenario2_part2_start_node_flow.groupby(['Start Node']).sum()
df_scenario2_part2_start_node_flow.reset_index(inplace = True)

# Find the flow to each node
df_scenario2_part2_end_node_flow = df_scenario2_part2[['End Node', 'Flow']]
df_scenario2_part2_end_node_flow = df_scenario2_part2_end_node_flow.groupby(['End Node']).sum()
df_scenario2_part2_end_node_flow.reset_index(inplace = True)

# Combine flow to/from each node
df_scenario2_part2_net_flow = (pd.merge(df_scenario2_part2_start_node_flow, df_scenario2_part2_end_node_flow, 
                                how = 'outer', left_on = 'Start Node', right_on = 'End Node'))
df_scenario2_part2_net_flow = df_scenario2_part2_net_flow.fillna(0)
df_scenario2_part2_net_flow['Node ID'] = df_scenario2_part2_net_flow[['Start Node', 'End Node']].max(axis = 1)

# Calculate net flow from each node
df_scenario2_part2_net_flow['Net Flow - Scenario 2 part 2'] = (df_scenario2_part2_net_flow['Flow_x'] - 
                                                               df_scenario2_part2_net_flow['Flow_y'])
df_scenario2_part2_net_flow = df_scenario2_part2_net_flow[['Node ID', 'Net Flow - Scenario 2 part 2']]

# Add columns in df_nodes_final and update values
df_nodes_final = pd.merge(df_nodes_final, df_scenario2_part2_net_flow, how = 'outer', on = 'Node ID')
df_nodes_final['Total Flow - Scenario 2'] = (df_nodes_final['Flow - Scenario 2 part 1'] + 
                                             df_nodes_final['Flow - Scenario 2 part 2'])
df_nodes_final['Remaining Storage - Scenario 2'] = (df_nodes_final['Remaining Storage - Scenario 2 part 1'] - 
                                                    df_nodes_final['Net Flow - Scenario 2 part 2'])
df_nodes_final.loc[df_nodes_final['Storage Capacity'] == 0, 'Remaining Storage - Scenario 2'] = 0

# Remove intermediary columns from df_nodes_final
df_nodes_final = df_nodes_final.drop(['Remaining Storage - Baseline part 1', 
                'Remaining Storage Capacity - Baseline part 1', 'New Supplies - Baseline', 'New Demand - Baseline', 
                'New Supply/Demand - Baseline', 'Remaining Storage - Scenario 1 part 1', 
                'Remaining Storage Capacity - Scenario 1 part 1', 'New Supplies - Scenario 1', 
                'New Demand - Scenario 1', 'New Supply/Demand - Scenario 1', 'Remaining Storage - Scenario 2 part 1', 
                'Remaining Storage Capacity - Scenario 2 part 1', 'New Supplies - Scenario 2', 
                'New Demand - Scenario 2', 'New Supply/Demand - Scenario 2'], axis = 1)

# Export node and edge files to csv for gephi import
df_baseline_gephi.to_csv('Supply_Chain_Model_Flow_Edges_Baseline.csv', index = False)
df_scenario1_gephi.to_csv('Supply_Chain_Model_Flow_Edges_Scenario1_Actual_Capacity.csv', index = False)
df_scenario2_gephi.to_csv('Supply_Chain_Model_Flow_Edges_Scenario2_Expected_Capacity.csv', index = False)
df_nodes_gephi.to_csv('Supply_Chain_Model_Nodes.csv', index = False)
df_edges_gephi.to_csv('Supply_Chain_Model_All_Edges.csv', index = False)

# Re-number start nodes in df_edges_risks to merge with 3 scenario dfs
df_edges_risks = df_edges_risks[df_edges_risks['Start Node'] > nodes_len]
df_edges_risks['Start Node'] = df_edges_risks['Start Node'] - nodes_len

# Add baseline flow to df_edges_final
df_edges_final = pd.merge(df_edges_risks, df_baseline_part1, how = 'left', on = 'Edge ID')
df_edges_final = pd.merge(df_edges_final, df_baseline_part2, how = 'left', on = 'Edge ID')
df_edges_final = (df_edges_final[['Edge ID', 'Start Node_x', 'End Node_x', 'Edge Name', 'Description', 'Type', 
                                  'Capacity_x', 'Total Risks', 'Flow_x', 'Flow_y']])
df_edges_final = (df_edges_final.rename(columns = {'Flow_x': 'Flow - Baseline part 1', 
                                                   'Flow_y': 'Flow - Baseline part 2'}))
df_edges_final['Total Flow - Baseline'] = (df_edges_final['Flow - Baseline part 1'] + 
                                           df_edges_final['Flow - Baseline part 2'])
df_edges_final.columns = [x.replace('_x', '') for x in df_edges_final.columns]

# Add scenario 1 flow to df_edges_final
df_edges_final = pd.merge(df_edges_final, df_scenario1_part1, how = 'left', on = 'Edge ID')
df_edges_final = pd.merge(df_edges_final, df_scenario1_part2, how = 'left', on = 'Edge ID')
df_edges_final = (df_edges_final.drop(['Start Node_y', 'End Node_y', 'Capacity_y', 'Risk_x', 'Start Node', 'End Node', 
                                       'Capacity', 'Risk_y'], axis = 1))
df_edges_final = (df_edges_final.rename(columns = {'Flow_x': 'Flow - Scenario 1 part 1',
                                                   'Flow_y': 'Flow - Scenario 1 part 2'}))
df_edges_final['Total Flow - Scenario 1'] = (df_edges_final['Flow - Scenario 1 part 1'] + 
                                             df_edges_final['Flow - Scenario 1 part 2'])
df_edges_final.columns = [x.replace('_x', '') for x in df_edges_final.columns]

# Add scenario 2 flow to df_edges_final
df_edges_final = pd.merge(df_edges_final, df_scenario2_part1, how = 'left', on = 'Edge ID')
df_edges_final = pd.merge(df_edges_final, df_scenario2_part2, how = 'left', on = 'Edge ID')
df_edges_final = (df_edges_final.drop(['Start Node_y', 'End Node_y', 'Capacity_y', 'Risk_x', 'Start Node', 'End Node', 
                                       'Capacity', 'Risk_y'], axis = 1))
df_edges_final = (df_edges_final.rename(columns = {'Flow_x': 'Flow - Scenario 2 part 1',
                                                   'Flow_y': 'Flow - Scenario 2 part 2'}))
df_edges_final['Total Flow - Scenario 2'] = (df_edges_final['Flow - Scenario 2 part 1'] + 
                                             df_edges_final['Flow - Scenario 2 part 2'])
df_edges_final.columns = [x.replace('_x', '') for x in df_edges_final.columns]

# Add column 'Required Flow - Scenario 2' to df_edges_final with temporary values of 0
df_edges_final['Required Flow - Scenario 2'] = 0

# Calculate values in 'Required Flow - Scenario 2'
for i in df_edges_final['Required Flow - Scenario 2']:
    if not df_edges_final.loc[i, 'Total Risks'] == 1:
        df_edges_final['Required Flow - Scenario 2'] = (df_edges_final['Total Flow - Scenario 2'] / 
                                                       (1 - df_edges_final['Total Risks']))
    else:
        df_edges_final['Required Flow - Scenario 2'] == 0

# Create lit 'summary_stats' that includes accumulated risk values from the 2 non-baseline scenarios
summary_stats = ([accum_risk_scenario1, accum_risk_percent_scenario1, accum_risk_scenario2, 
                  accum_risk_percent_scenario2])

# Create list 'summary_stats_names'
summary_stats_names = (['Scenario 1 - Risk-based routing actual capacity', 
                        'Scenario 1 - Risk-based routing actual capacity',
                        'Scenario 2 - Risk-based routing expected capacity', 
                        'Scenario 2 - Risk-based routing expected capacity'])

# Create list 'summary_stats_description'
summary_stats_description = (['Accumulated Risk', 'Accumulated Risk Percent',
                              'Accumulated Risk', 'Accumulated Risk Percent',])

# Create 'df_summary' dataframe combining lists 'summary_stats_names', 'summary_stats_description', 'summary_stats'
df_summary = (pd.DataFrame(zip(summary_stats_names, summary_stats_description, summary_stats), 
                          columns = ['Scenario Description', 'Statistic', 'Value']))

# Export 5 original sheets & summary sheet to Excel with 3 scenario flow columns added to 'Edges' sheet
writer = pd.ExcelWriter('OUT_Supply_Chain_Model.xlsx', engine = 'xlsxwriter')
df_summary.to_excel(writer, sheet_name = 'Summary', index = False)
df_nodes_final.to_excel(writer, sheet_name = 'Nodes', index = False)
df_edges_final.to_excel(writer, sheet_name = 'Edges', index = False)
df_risks_location_orig.to_excel(writer, sheet_name = 'Risks - Location', index = False)
df_risks_type_orig.to_excel(writer, sheet_name = 'Risks - Type', index = False)
df_risks_list_orig.to_excel(writer, sheet_name = 'Risks - List', index = False)
writer.save()