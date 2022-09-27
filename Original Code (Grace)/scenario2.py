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
        if start_nodes[i] > nodes_count:
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
df_scenario2_part1['Start Node'] = df_scenario2_part1['Start Node'] - nodes_count
df_scenario2_part1['Edge ID'] = df_scenario2_part1.index + 1

# Create df_scenario2_part1_v2 that includes flow along "fake" nodes (blue edges part 1)
df_scenario2_part1_v2 = (pd.DataFrame(zip(start_node_output_v2, end_node_output_v2, flow_output_v2, capacity_output_v2, 
                   risk_output_v2), columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))

# Add scenario 2 flow to df_nodes_final

df_nodes_final = (pd.merge(df_nodes_final, df_scenario2_part1_v2, how = 'left', left_on = 'Node ID', 
                           right_on = 'Start Node'))
df_nodes_final = df_nodes_final.drop(['Start Node', 'End Node', 'Capacity', 'Risk'], axis = 1)
df_nodes_final = df_nodes_final.rename(columns = {'Flow': 'Flow - Scenario 2 part 1'})

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
df_new_supplies['Node ID'] += nodes_count
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
        if start_nodes[i] > nodes_count:
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
df_scenario2_part2['Start Node'] = df_scenario2_part2['Start Node'] - nodes_count
df_scenario2_part2['Edge ID'] = df_scenario2_part2.index + 1

# Create df_scenario2_part2_v2 that includes flow along "fake" nodes (blue edges part 1)
df_scenario2_part2_v2 = (pd.DataFrame(zip(start_node_output_v2, end_node_output_v2, flow_output_v2, capacity_output_v2, 
                   risk_output_v2), columns = ['Start Node', 'End Node', 'Flow', 'Capacity', 'Risk']))

# Add scenario 2 flow part 2 to df_nodes_final
df_nodes_final = (pd.merge(df_nodes_final, df_scenario2_part2_v2, how = 'left', left_on = 'Node ID', 
                           right_on = 'Start Node'))
df_nodes_final = df_nodes_final.drop(['Start Node', 'End Node', 'Capacity', 'Risk'], axis = 1)
df_nodes_final = df_nodes_final.rename(columns = {'Flow': 'Flow - Scenario 2 part 2'})

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