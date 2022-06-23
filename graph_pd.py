from platform import node
import pandas as pd
import geopandas as gpd
import geojson
import shapely as shp
import numpy as np
from enum import Enum
import json
from ortools.graph import pywrapgraph

STANDARD_RISK = lambda probability, impact: probability * impact

# Constants based on existing input file format
DEFAULT_NODE_ID_LBL = 'Node ID'
DEFAULT_NODE_LATITUDE_LBL = 'Latitude'
DEFAULT_NODE_LONGITUDE_LBL = 'Longitude'
DEFAULT_NODE_THROUGHPUT_LBL = 'Throughput'
DEFAULT_NODE_STORAGE_CAPACITY_LBL = 'Storage Capacity'
DEFAULT_NODE_RISK_LBL = 'Risk'
DEFAULT_NODE_SUPPLY_LBL = 'Supply'
DEFAULT_NODE_DEMAND_LBL = 'Demand'
DEFAULT_NODE_CURRENT_STORAGE_LBL = 'Current Storage'

DEFAULT_EDGE_ID_LBL = 'Edge ID'
DEFAULT_EDGE_START_LBL = 'Start Node'
DEFAULT_EDGE_END_LBL = 'End Node'
DEFAULT_EDGE_CAPACITY_LBL = 'Capacity'
DEFAULT_EDGE_RISK_LBL = 'Risk'

DEFAULT_LOCATION_OUTER_DISTANCE_LBL = 'Outer Distance'

DEFAULT_RISK_PROBABILITY_LBL = 'Probability'
DEFAULT_RISK_IMPACT_LBL = 'Impact'

DEFAULT_FLAT_PROJECTION = 'EPSG:3857'
DEFAULT_SPHERICAL_PROJECTION = 'EPSG:4326'

class Graph:

    class Node:

        # TODO: In Python, is there really ever a need for default values
        # id = 0
        # throughput = 0
        # storage_capacity = 0
        # overall_risk = 0
        # risks = {}
        # supply = 0
        # demand = 0
        # current_storage = 0
        # lat_long = (0.0, 0.0)
        # props = {}
        
        # Captures reference to enclosing graph; better way to do mapping to containing object?
        def __init__(self, graph, id, props):
            self.graph = graph
            self.id = id #props[DEFAULT_NODE_ID_LBL]
            self.throughput = props[DEFAULT_NODE_THROUGHPUT_LBL]
            self.storage_capacity = props[DEFAULT_NODE_STORAGE_CAPACITY_LBL]
            self.overall_risk = props[DEFAULT_NODE_RISK_LBL] if DEFAULT_NODE_RISK_LBL in props.keys() else 0
            self.supply = props[DEFAULT_NODE_SUPPLY_LBL]
            self.demand = props[DEFAULT_NODE_DEMAND_LBL]
            self.current_storage = props[DEFAULT_NODE_CURRENT_STORAGE_LBL]
            self.lat_long = (props[DEFAULT_NODE_LATITUDE_LBL], props[DEFAULT_NODE_LONGITUDE_LBL])
            self.props = props if isinstance(props, dict) else props.to_dict()
            
        def to_json(self):
            dict = {}
            dict['type'] = 'node'
            dict['id'] = self.id
            dict['location'] = {'latitude' : self.lat_long[0], 'longitude' : self.lat_long[1]}
            dict['geometry'] = geojson.dumps(self.graph.nodes.loc[self.id, 'geometry']) if 'geometry' in self.graph.nodes.columns else 'null'
            dict['risks'] = self.graph.node_ind_risks[self.id]
            return json.dumps(dict)
            
        def __str__(self):
            return self.props.__str__()

        @classmethod
        def fromPandas(cls, series, node_id_lbl = DEFAULT_NODE_ID_LBL):
            return cls(series[node_id_lbl], **series.to_dict())

    class Edge:
        
        # TODO: In Python, is there really ever a need for default values
        # id = 0
        # start_node_id = 0
        # end_node_id = 0
        # capacity = 0
        # overall_risk = 0
        # risks = {}
        # props = {}

        def __init__(self, id, props):
            self.id = id #props[DEFAULT_EDGE_ID_LBL]
            self.graph = self
            self.start_node_id = props[DEFAULT_EDGE_START_LBL]
            self.end_node_id = props[DEFAULT_EDGE_END_LBL]
            self.capacity = props[DEFAULT_EDGE_CAPACITY_LBL]
            self.overall_risk = props[DEFAULT_EDGE_RISK_LBL] if DEFAULT_EDGE_RISK_LBL in props.keys() else 0
            self.props = props.to_dict()
            
        def to_json(self):
            dict = {}
            dict['type'] = 'edge'
            dict['id'] = int(self.id)
            dict['start'] = int(self.start_node_id)
            dict['end'] = int(self.end_node_id)
            dict['geometry'] = geojson.dumps(self.graph.edges.loc[self.id, 'geometry']) if 'geometry' in self.graph.edges.columns else 'null'
            dict['risks'] = self.graph.edge_ind_risks[self.id]
            return json.dumps(dict)

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
    
    node_ind_risks = None
    edge_ind_risks = None

    def __init__(self, nodes = None, edges = None):
        self.nodes = nodes
        self.edges = edges

    def get_nodes(self):
        return self.nodes

    def get_all_edges(self):
        return self.edges

    def get_node(self, id):
        ret = self.nodes.loc[self.nodes['Node ID'] == id].set_index('Node ID', drop = False).to_dict('index')[id]
        ret.update([('Risks', self.node_ind_risks[id])])
        return self.Node(self, id, ret)
        
    def get_sinks_from_source(self, id):
        '''Return a list of the indices of all nodes that have an incoming edge from id'''
        return self.edges.loc[self.edges[DEFAULT_EDGE_START_LBL] == id, DEFAULT_EDGE_END_LBL].tolist()

    def cut_edges(self, id = None, source = None, sink = None):
        if id == None:
            self.edges.drop(self.edges[(self.edges[DEFAULT_EDGE_START_LBL] == source) & (self.edges[DEFAULT_EDGE_END_LBL] == sink)].index, inplace=True)
        else:
            self.edges.drop(id, inplace=True)

    def get_all_edges(self):
        return [self.Edge(i['Edge ID'], i.squeeze()) for x, i in self.edges.iterrows()]
        
    def get_all_nodes(self):
        return [self.get_node(n['Node ID']) for x, n in self.nodes.iterrows()]

    def get_edges(self, id = None, source = None, sink = None):
        if id != None:
            return self.Edge(self, self.edges.loc[id])
        else:
            try:
                return [self.Edge(self, i.squeeze()) for x, i in self.edges.loc[(self.edges[DEFAULT_EDGE_START_LBL] == source) & (self.edges[DEFAULT_EDGE_END_LBL] == sink)].iterrows()]
                #return self.Edge(self.edges.loc[(self.edges[DEFAULT_EDGE_START_LBL] == source) & (self.edges[DEFAULT_EDGE_END_LBL] == sink)].squeeze())
            except KeyError:
                return None
        
    def copy(self):
        return Graph(self.nodes.copy(), self.edges.copy())

    def to_adj_list(self):
        g = {}
        for i, node in self.nodes.iterrows():
            g[node[DEFAULT_NODE_ID_LBL]] = self.get_sinks_from_source(node[DEFAULT_NODE_ID_LBL])
        return g

    def topological_sort(self, DEBUG = False):
        '''Returns a list of node indices, sorted s.t. E(A,B) implies a < b in ordering'''
        adj_list = self.to_adj_list()
        unmarked_nodes = self.nodes[DEFAULT_NODE_ID_LBL].tolist()
        temporary_mark = dict([(id, False) for id in unmarked_nodes])
        sort = []

        def visit(id, tabs = 0):
            if DEBUG:
                print(''.join(['\t' for i in range(tabs)]), id)
            if id not in unmarked_nodes:
                return
            elif temporary_mark[id]:
                raise Exception("Graph is not a DAG; Topological Sort makes no sense!")

            temporary_mark[id] = True
            tabs += 1
            for sink_id in adj_list[id]:
                visit(sink_id, tabs)
            temporary_mark[id] = False

            unmarked_nodes.remove(id)
            sort.append(id)

        while unmarked_nodes:
            cur = unmarked_nodes[0]
            visit(cur)

        return sort

    def flow(self, src, sink, amt, with_risk = False):
        # Flow up to amt
        # Underlying assumption: flow may begin at supply, but always ends at current storage
        tank_lbl = DEFAULT_NODE_SUPPLY_LBL if self.nodes.loc[src, DEFAULT_NODE_SUPPLY_LBL] > 0 else DEFAULT_NODE_CURRENT_STORAGE_LBL
        actual_flow = min(amt, self.nodes.loc[src, tank_lbl])
        self.nodes.loc[self.nodes[DEFAULT_NODE_ID_LBL] == src, tank_lbl] -= actual_flow
        self.nodes.loc[self.nodes[DEFAULT_NODE_ID_LBL] == sink, DEFAULT_NODE_CURRENT_STORAGE_LBL] += actual_flow
    
    def export_to_xlsx(self, filename, nodes_sheet_name = "Nodes", edges_sheet_name = "Edges"):
        with pd.ExcelWriter(filename, engine = 'openpyxl') as writer:
            self.nodes.to_excel(writer, sheet_name = nodes_sheet_name)
            self.edges.to_excel(writer, sheet_name = edges_sheet_name)


    # TODO: For now, assumes Types are disjoint for edges and nodes
    # Note: it appears some geopandas functions rely on the geometry column being called "geometry";
    #       TODO: investigate if geopandas requires "geometry" column name
    #       TODO: add support for "donut" shaped circular areas (Outer - Inner Distance)
    @staticmethod
    def geometry_from_points(gdf, shape, shape_column = None, point_columns = [], spherical_projection = DEFAULT_SPHERICAL_PROJECTION, flat_projection = DEFAULT_FLAT_PROJECTION):
        '''Create geometry objects in GeoDataFrame gdf representing shape based on the points provided in point_columns'''

        # While not used currently, the shape_column parameter should allow for in-place computing of shapes (e.g., just create geometry objects for rows with "circle" in the shape_column)
        shapes = gdf
        if shape_column:
            shapes = gdf[gdf[shape_column] == shape]

        if shape == 'Box':
            shapes['geometry'] = shapes.apply(lambda r: shp.geometry.box(*r[point_columns]), axis = 1)
            shapes = gpd.GeoDataFrame(shapes, crs = spherical_projection)
        elif shape == 'Circle':
            # Project flat to take calculate buffer size (radius around point), then project back to spherical 
            shapes['geometry'] = shapes.apply(lambda r: shp.geometry.Point(*r[point_columns]), axis = 1)
            shapes = gpd.GeoDataFrame(shapes, crs = spherical_projection)
            shapes = shapes.to_crs(crs=flat_projection)
            shapes['geometry'] = shapes['geometry'].buffer(shapes[DEFAULT_LOCATION_OUTER_DISTANCE_LBL]).to_crs(spherical_projection)
        elif shape == 'Point':
            shapes['geometry'] = shapes.apply(lambda r: shp.geometry.Point(*r[point_columns]), axis = 1)
            shapes = gpd.GeoDataFrame(shapes, crs = spherical_projection)
        elif shape == 'Line':
            shapes['geometry'] = shapes.apply(lambda r: shp.geometry.LineString(r[point_columns]), axis = 1)
            shapes = gpd.GeoDataFrame(shapes, crs = spherical_projection)
        elif shape == 'Polygon':
            shapes['geometry'] = shapes.apply(lambda r: shp.geometry.Polygon(*r[point_columns]), axis = 1)
    
        return gdf.merge(shapes) if shape_column else shapes
        
    def populate_from_xlsx(self, filename, nodes_sheet_name = 'Nodes', edges_sheet_name = 'Edges', node_id_col = DEFAULT_NODE_ID_LBL, edge_id_col = DEFAULT_EDGE_ID_LBL, start_node_col = DEFAULT_EDGE_START_LBL, end_node_col = DEFAULT_EDGE_END_LBL, latitude_column = DEFAULT_NODE_LATITUDE_LBL, longitude_column = DEFAULT_NODE_LONGITUDE_LBL):
        g = pd.read_excel(filename, sheet_name = [nodes_sheet_name, edges_sheet_name], engine='openpyxl')
        # Add geometry column for points
        self.nodes = g[nodes_sheet_name]
        self.node_ind_risks = dict([(id, []) for id in self.nodes[node_id_col]])
        self.nodes = self.geometry_from_points(self.nodes, shape = 'Point', point_columns = [latitude_column, longitude_column])

        # Add geometry column for edges
        self.edges = g[edges_sheet_name]
        node_locations = self.nodes[[node_id_col, 'geometry']]
        self.edge_ind_risks = dict([(id, []) for id in self.edges[edge_id_col]])
        self.edges = self.edges.merge(node_locations, left_on = start_node_col, right_on = node_id_col).rename(columns = {'geometry' : 'Start Point'})
        self.edges = self.edges.merge(node_locations, left_on = end_node_col, right_on = node_id_col).rename(columns = {'geometry' : 'End Point'})
        self.edges = self.geometry_from_points(self.edges, shape = 'Line', point_columns = ['Start Point', 'End Point'])
        self.edges = self.edges.drop(['Node ID_x', 'Node ID_y', 'Start Point', 'End Point'], axis = 1)
    
    def compute_location_risk(self, lrisks, risk_metric = STANDARD_RISK):
        '''For each node/edge, compute which risks in lrisks overlap it and apply risk_metric to calculate and update the risk for that node/edge.
            Modifies risks IN PLACE.'''
            
        if DEFAULT_NODE_RISK_LBL not in self.nodes.columns:
            self.nodes[DEFAULT_NODE_RISK_LBL] = 0
        if DEFAULT_EDGE_RISK_LBL not in self.edges.columns:
            self.edges[DEFAULT_EDGE_RISK_LBL] = 0

        for i, node in self.nodes.iterrows():
            applied_risks = lrisks[lrisks.contains(node['geometry'])] 
            for i, ar in applied_risks.iterrows():
                self.node_ind_risks[node['Node ID']].append({'type' : 'location', 'probability' : ar['Probability'], 'impact' : ar['Impact']})
            self.nodes.at[i, DEFAULT_NODE_RISK_LBL] += risk_metric(applied_risks['Probability'], applied_risks['Impact']).sum()

        for i, edge in self.edges.iterrows():
            applied_risks = lrisks[lrisks.intersects(edge['geometry'])]
            for i, ar in applied_risks.iterrows():
                self.edge_ind_risks[node['Node ID']].append({'type' : 'location', 'probability' : ar['Probability'], 'impact' : ar['Impact']})
            self.edges.at[i, DEFAULT_EDGE_RISK_LBL] += risk_metric(applied_risks['Probability'],applied_risks['Impact']).sum()

        # Assume total risk should be capped at 1
        self.nodes.loc[self.nodes[DEFAULT_NODE_RISK_LBL] > 1, DEFAULT_NODE_RISK_LBL] = 1.0
        self.edges.loc[self.edges[DEFAULT_EDGE_RISK_LBL] > 1, DEFAULT_EDGE_RISK_LBL] = 1.0
        
    def to_json(self):
        return [n.to_json() for n in self.get_all_nodes()].append([e.to_json() for e in self.get_all_edges()])

    def compute_risk(self, risks, risk_metric = STANDARD_RISK, apply_node_risk_to_edges = False, edge_or_nodes_col_name = 'Edge or Node', name_col = 'Risk Name', 
                    node_str = 'Node', edge_str = 'Edge', node_id_risk_col = DEFAULT_NODE_ID_LBL, edge_id_risk_col = DEFAULT_EDGE_ID_LBL, 
                    node_id_col = DEFAULT_NODE_ID_LBL, edge_id_col = DEFAULT_EDGE_ID_LBL, risk_probability_col = DEFAULT_RISK_PROBABILITY_LBL,
                    risk_impact_col = DEFAULT_RISK_IMPACT_LBL, edge_start_node_col = DEFAULT_EDGE_START_LBL, edge_end_node_col = DEFAULT_EDGE_END_LBL):
        '''Take risks as input and compute which nodes/edges are affected based on shared attributes (e.g., type, description, etc.).
            Modifies risks IN PLACE. Currently this assumes that all shared attributes must match'''
            
        if DEFAULT_NODE_RISK_LBL not in self.nodes.columns:
            self.nodes[DEFAULT_NODE_RISK_LBL] = 0
        if DEFAULT_EDGE_RISK_LBL not in self.edges.columns:
            self.edges[DEFAULT_EDGE_RISK_LBL] = 0

        # Compute risks associated with nodes, based on intersecting columns
        node_risks = risks[risks[edge_or_nodes_col_name] == node_str].rename(columns = {node_id_risk_col : node_id_col})
        intersection = node_risks.columns.intersection(self.nodes.columns) 
        comb_risks = pd.merge(node_risks, self.nodes, how = 'inner', on = intersection.tolist())

        for i, row in comb_risks.iterrows():
            self.nodes.loc[self.nodes[node_id_col] == row[node_id_col], DEFAULT_NODE_RISK_LBL] += risk_metric(row[risk_probability_col], row[risk_impact_col])

        # Compute risks associated with edges, based on intersecting columns
        edge_risks = risks[risks[edge_or_nodes_col_name] == edge_str].rename(columns = {edge_id_risk_col : edge_id_col})
        intersection = edge_risks.columns.intersection(self.edges.columns)
        comb_risks = pd.merge(edge_risks, self.edges, how = 'inner', on = intersection.tolist())

        for i, row in comb_risks.iterrows():
            self.edges.loc[self.edges[edge_id_col] == row[edge_id_col], DEFAULT_EDGE_RISK_LBL] += risk_metric(row[risk_probability_col], row[risk_impact_col])

        # Add risk of start_node and end_node to edge risk if appropriate
        if apply_node_risk_to_edges:
            edges_to_nodes = pd.merge(pd.merge(self.edges, self.nodes[[node_id_col, DEFAULT_NODE_RISK_LBL]], how = 'left', left_on = edge_start_node_col, right_on = node_id_col), self.nodes[[node_id_col, DEFAULT_NODE_RISK_LBL]], how = 'left', left_on = edge_end_node_col, right_on = node_id_col)
            for i, row in edges_to_nodes.iterrows():
                self.edges.loc[self.edges[edge_id_col] == row[edge_id_col], DEFAULT_EDGE_RISK_LBL] += row[DEFAULT_EDGE_RISK_LBL + '_y'] + row[DEFAULT_EDGE_RISK_LBL]
        
        # Assume total risk should be capped at 1
        self.nodes.loc[self.nodes[DEFAULT_NODE_RISK_LBL] > 1, DEFAULT_NODE_RISK_LBL] = 1.0
        self.edges.loc[self.edges[DEFAULT_EDGE_RISK_LBL] > 1, DEFAULT_EDGE_RISK_LBL] = 1.0

    def compute_min_cost_flow(self, use_expected_capacity = False):
        '''Returns a dictionary mapping edge_id to amount of flow.'''

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
            node_loc = (n['Latitude'], n['Longitude'])
            edge_geometry = shp.geometry.LineString([node_loc, node_loc])
            internal_edges = internal_edges.append({'Edge ID' : edges_count + i,'Start Node' : n['Node ID'], 'End Node' : n['Node ID'] + nodes_count, 'Capacity' : n['Throughput'], 'Edge Name' : 'Node ' + str(n['Node ID'])  + ' Internal','Risk' : 0, 'Description' : 'yellow', 'Type' : 'Internal', 'geometry' : edge_geometry}, ignore_index = True)

        nodes = pd.concat([nodes, temporary_nodes])
        nodes = pd.DataFrame(nodes.reset_index(drop = True))

        edges['Start Node'] += nodes_count
        edges = edges.append(internal_edges)
        edges = pd.DataFrame(edges.reset_index(drop = True))

        graph = pywrapgraph.SimpleMinCostFlow()

        for i, e in edges.iterrows():
            if use_expected_capacity:
                graph.AddArcWithCapacityAndUnitCost(e['Start Node'], e['End Node'], e['Capacity'], int(e['Risk'] * 1000))
            else:
                graph.AddArcWithCapacityAndUnitCost(e['Start Node'], e['End Node'], e['Capacity'], 0)
        
        for i, n in nodes.iterrows():
            graph.SetNodeSupply(n['Node ID'], n['Supply'] + n['Current Storage'] - n['Demand'])

        net_flows = dict([(i, 0) for i in nodes['Node ID'].tolist()])
        # First Round, satisfying demand for sinks
        # Computing net flow into/out of each node
        if graph.SolveMaxFlowWithMinCost():
            print('---First Pass---')
            print('# of Arcs:', graph.NumArcs())
            print('Max flow:', graph.MaximumFlow())
            print('Accumulated Risk:', graph.OptimalCost() / 1000)
            for i in range(graph.NumArcs()):
                net_flows[graph.Tail(i)] += graph.Flow(i)
                net_flows[graph.Head(i)] -= graph.Flow(i)
                edges.loc[(edges['Start Node'] == graph.Tail(i)) & (edges['End Node'] == graph.Head(i)), ['Capacity']] -= graph.Flow(i)
                # print('%1s -> %1s   %3s  / %3s' %
                #     (graph.Tail(i), graph.Head(i), graph.Flow(i),
                #     graph.Capacity(i)))

        # New object because appears arcs can't be removed/updated
        graph_resupply = pywrapgraph.SimpleMinCostFlow()

        for i, e in edges.iterrows():
            if use_expected_capacity:
                graph_resupply.AddArcWithCapacityAndUnitCost(e['Start Node'], e['End Node'], e['Capacity'], int(e['Risk'] * 1000))
            else:
                graph_resupply.AddArcWithCapacityAndUnitCost(e['Start Node'], e['End Node'], e['Capacity'], 0)

        for i, n in nodes.iterrows():
            remaining_storage = 0 if n['Demand'] > 0 else max(0, n['Current Storage'] - net_flows[n['Node ID']])
            available_capacity = n['Storage Capacity'] - remaining_storage
            new_supply = 0 if n['Resupply'] == 'No' else max(0, n['Supply'] - net_flows[n['Node ID']])
            graph_resupply.SetNodeSupply(n['Node ID'], int(new_supply - available_capacity))

        edges['Flow'] = 0

        # Second Round, resupplying nodes
        if graph_resupply.SolveMaxFlowWithMinCost():
            print('---Second Pass---')
            print('# of Arcs:', graph.NumArcs())
            print('Max flow:', graph.MaximumFlow() + graph_resupply.MaximumFlow())
            print('Accumulated Risk:', graph.OptimalCost() / 1000 + graph_resupply.OptimalCost() / 1000)
            for i in range(graph_resupply.NumArcs()):
                # Assumes no duplicate edges
                edges.loc[(edges['Start Node'] == graph_resupply.Tail(i)) & (edges['End Node'] == graph_resupply.Head(i)), ['Flow']] = graph.Flow(i) + graph_resupply.Flow(i)
                #print('%1s -> %1s   %3s  / %3s' %
                #    (graph_resupply.Tail(i), graph_resupply.Head(i), graph.Flow(i) + graph_resupply.Flow(i),
                #    graph_resupply.Capacity(i)))
        
        # Remove temporary nodes and edges and appropriately combine their flows
        nodes['Net Flow'] = 0
        for i, e in edges.iterrows():
            edges.loc[edges['Edge ID'] == e['Edge ID'], ['Flow']] = e['Flow']
            nodes.loc[nodes['Node ID'] == e['Start Node'], ['Net Flow']] += e['Flow']
            nodes.loc[nodes['Node ID'] == e['End Node'], ['Net Flow']] -= e['Flow']
        
        for i, n in nodes.iterrows():
            if n['Node ID'] < nodes_count:
                n['Net Flow'] = n['Net Flow'] - nodes.loc[nodes['Node ID'] == n['Node ID'] + nodes_count, ['Net Flow']]['Net Flow']

        edges = edges.loc[edges['Edge ID'] < edges_count]
        edges['Start Node'] -= nodes_count
        self.edges = edges

        self.nodes = nodes.loc[nodes['Node ID'] < nodes_count]