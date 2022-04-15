import pandas as pd
import numpy as np
from enum import Enum
from ortools.graph import pywrapgraph

STANDARD_RISK = lambda probability, impact: probability * impact

class Graph:

    class Node:

        id = 0
        props = None
        risks = []
        
        def __init__(self, id, risks = [], **kwargs):

            self.id = id
            self.props = pd.Series(data=kwargs)
            self.risks = risks

        # def __str__(self):
        #     return self.props.__str__() + " RISKS: " + self.risks.__str()

        @classmethod
        def fromPandas(cls, series):
            return cls(series['Node ID'], **series.to_dict())

    class Edge:
        
        id = 0
        props = None

        def __init__(self, id, **kwargs):
            self.id = id
            self.props = pd.Series(data=kwargs)

        def __str__(self):
            return self.props.__str__()

        @classmethod
        def fromPandas(cls, series, nodes = {}):
            s = nodes[series['Start Node']] if series['Start Node'] in nodes.keys() else None
            e = nodes[series['End Node']] if series['End Node'] in nodes.keys() else None
            series.pop('Start Node')
            series.pop('End Node')
            return cls(series['Edge ID'], start_node = s, end_node = e, **series.to_dict())

    nodes = None
    edges = None

    def populate_from_xlsx(self, filename, nodes_sheet_name = 'Nodes', edges_sheet_name = 'Edges'):
        g = pd.read_excel(filename, sheet_name = [nodes_sheet_name, edges_sheet_name], engine='openpyxl')

        # for i, n in g['Nodes'].iterrows():
        #     self.nodes[n['Node ID']] = self.Node.fromPandas(n)

        # for i, e in g['Edges'].iterrows():
        #     self.edges[e['Edge ID']] = self.Edge.fromPandas(e, self.nodes)
        self.nodes = g['Nodes']
        self.nodes['Risk'] = 0

        self.edges = g['Edges']
        self.edges['Risk'] = 0

    # TODO: For now, assumes Types are disjoint for edges and nodes
    # Computing risks:
    # For each row:
    #   if all of shared parameters match, then use the given risk metric and store the computed risk under the given name

    def compute_risk(self, risks, risk_metric = STANDARD_RISK, apply_node_risk_to_edges = True, edge_or_nodes_col_name = 'Edge or Node', name_col = 'Risk Name', node_str = 'Node', edge_str = 'Edge', node_id_risk = "Node ID", edge_id_risk = "Edge ID"):
        '''Take risks as input and compute which nodes/edges are affected based on shared attributes (e.g., type, description, etc.)'''
        
        # Compute risks associated with nodes
        node_risks = risks[risks[edge_or_nodes_col_name] == node_str]
        
        intersection = node_risks.columns.intersection(self.nodes.columns)
        
        comb_risks = pd.merge(node_risks, self.nodes, how = 'inner', on = intersection.tolist())

        for i, row in comb_risks.iterrows():
            self.nodes.loc[self.nodes['Node ID'] == row[node_id_risk], 'Risk'] += risk_metric(row['Probability'], row['Impact'])

        # Compute risks associated with edges
        edge_risks = risks[risks[edge_or_nodes_col_name] == edge_str]

        intersection = edge_risks.columns.intersection(self.edges.columns)

        comb_risks = pd.merge(edge_risks, self.edges, how = 'inner', on = intersection.tolist())

        for i, row in comb_risks.iterrows():
            self.edges.loc[self.edges['Edge ID'] == row[edge_id_risk], 'Risk'] += risk_metric(row['Probability'], row['Impact'])

        # Add risk of start_node and end_node to edge risk if appropriate
        if apply_node_risk_to_edges:
            edges_to_nodes = pd.merge(pd.merge(self.edges, self.nodes[['Node ID', 'Risk']], how = 'left', left_on = 'Start Node', right_on = 'Node ID'), self.nodes[['Node ID', 'Risk']], how = 'left', left_on = 'End Node', right_on = 'Node ID')
            for i, row in edges_to_nodes.iterrows():
                # Corrects for overflowing risks (> 1)
                self.edges.loc[self.edges['Edge ID'] == row['Edge ID'], 'Risk'] += row['Risk_y'] + row['Risk']
                self.edges.loc[self.edges['Risk'] > 1, 'Risk'] = 1

    def compute_min_cost_flow(self, use_expected_capacity = False):

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)

        # Operate on copies
        nodes = self.nodes.copy()
        edges = self.edges.copy()
        nodes_count = len(nodes)
        edges_count = len(edges)

        # Add temporary nodes
        temporary_nodes = nodes.copy()
        temporary_nodes['Node ID'] += nodes_count

        nodes['Demand'] = 0
        temporary_nodes['Supply'] = 0
        temporary_nodes['Storage Capacity'] = 0
        temporary_nodes['Current Storage'] = 0
        temporary_nodes['Risk'] = 0

        # Add temporary edges
            # Before: A------B------C
            # After:  A++++++A'-----B+++++B'------C++++++C'
            # Edges marked with '+' are labeled as 'Internal'
            
        internal_edges = pd.DataFrame(columns = ['Edge ID', 'Start Node', 'End Node', 'Edge Name', 'Description', 'Type', 'Capacity', 'Risk'])
        for i, n in nodes.iterrows():
            internal_edges = internal_edges.append({'Edge ID' : edges_count + i,'Start Node' : n['Node ID'], 'End Node' : n['Node ID'] + nodes_count, 'Capacity' : n['Throughput'], 'Edge Name' : 'Node ' + str(n['Node ID'])  + ' Internal','Risk' : 0, 'Description' : 'yellow', 'Type' : 'Internal'}, ignore_index = True)

        nodes = pd.concat([nodes, temporary_nodes])
        nodes = nodes.reset_index(drop = True)

        edges['Start Node'] += nodes_count
        edges = edges.append(internal_edges)
        edges = edges.reset_index(drop = True)

        graph = pywrapgraph.SimpleMinCostFlow()

        for i, e in edges.iterrows():
            graph.AddArcWithCapacityAndUnitCost(e['Start Node'], e['End Node'], e['Capacity'], 0)
        
        for i, n in nodes.iterrows():
            graph.SetNodeSupply(n['Node ID'], n['Supply'] + n['Current Storage'] - n['Demand'])

        net_flows = dict([(i, 0) for i in nodes['Node ID'].tolist()])

        # First Round, satisfying demand for sinks
        # Computing net flow into/out of each node
        if graph.SolveMaxFlowWithMinCost():
            print('# of Arcs:', graph.NumArcs())
            print('Max flow:', graph.MaximumFlow())
            for i in range(graph.NumArcs()):
                net_flows[graph.Tail(i)] += graph.Flow(i)
                net_flows[graph.Head(i)] -= graph.Flow(i)
                edges.loc[(edges['Start Node'] == graph.Tail(i)) & (edges['End Node'] == graph.Head(i)), ['Capacity']] -= graph.Flow(i)
                print('%1s -> %1s   %3s  / %3s' %
                    (graph.Tail(i), graph.Head(i), graph.Flow(i),
                    graph.Capacity(i)))

        print(nodes)

        # New object because appears arcs can't be removed/updated
        graph_2 = pywrapgraph.SimpleMinCostFlow()

        for i, e in edges.iterrows():
            graph_2.AddArcWithCapacityAndUnitCost(e['Start Node'], e['End Node'], e['Capacity'], 0)

        for i, n in nodes.iterrows():
            print(-n['Storage Capacity'] + n['Current Storage'] - net_flows[n['Node ID']])
            graph_2.SetNodeSupply(n['Node ID'], int(-n['Storage Capacity'] + n['Current Storage'] - net_flows[n['Node ID']]))

        if graph_2.SolveMaxFlowWithMinCost():
            print('# of Arcs:', graph.NumArcs())
            print('Max flow:', graph.MaximumFlow())
            for i in range(graph_2.NumArcs()):
                print('%1s -> %1s   %3s  / %3s' %
                    (graph.Tail(i), graph.Head(i), graph.Flow(i) + graph_2.Flow(i),
                    graph.Capacity(i)))


        # Second Round, attempting to satisfy demand for intermediate storage
            # Compute remaining storage as Total_Space - Net_Flow
            # Let remaining storage be new demand (backfill)
            # Re-run flow