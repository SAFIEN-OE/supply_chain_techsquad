from abc import abstractmethod
from platform import node
from typing import Iterable
import pandas as pd
import geopandas as gpd
import shapely as shp
import numpy as np
from enum import Enum
from ortools.graph import pywrapgraph
import constants
import json
import random
import pyproj
import geojson

class GraphComponent():

    @abstractmethod
    def get_type(self):
        return self.type

    @abstractmethod
    def get_id(self):
        return self.id

    @abstractmethod
    def get_name(self):
        return self.name

    @abstractmethod
    def get_geometry(self):
        return self.geometry

    @abstractmethod
    def get_risks(self):
        return self.risks

class Node(GraphComponent):
    '''
        Nodes consist of
            id - a unique identifier
            name - a (non-unique) string identifier
            type - a field identifying the type, found in constants
            throughput - the amount of fuel that can pass through the node in one time step
            storage_capacity - the amount of fuel that can remain in the node at the end of one time step
            supply - the amount of supply provided by the node (distinct from storage)
            demand - the amount of fuel demanded by the node at the end of one time step
            current_storage - the amount of fuel already stored by the node
            resupply - [LEGACY]
            location - where the node is in (Lat, Long) coordinates (in arbitrary coordinate system)
            risks - the list of risks applying to the node
            geometry - an object encapsulating where the node is (likely a point or None)
    '''


    def __init__(self, id = None, name = '', type = constants.NODE_TYPE_DEFAULT, 
                    throughput = 0, storage_capacity = 0, supply = 0, demand = 0,
                    current_storage = 0, resupply = False, location = (None, None), 
                    risks = [], geometry = None):
        self.id = id
        self.name = name
        self.type = type
        self.throughput = throughput
        self.storage_capacity = storage_capacity
        self.supply = supply
        self.demand = demand
        self.current_storage = current_storage
        self.resupply = resupply
        self.location = location
        self.risks = risks
        self.geometry = geometry

    def copy(self):
        return Node(self.id, self.name, self.type, self.throughput, self.storage_capacity,
                    self.supply, self.demand, self.current_storage, self.resupply, self.location,
                    self.risks, self.geometry)

    def get_type(self):
        '''returns the constant type of the node'''
        return super().get_type()

    def get_id(self):
        '''returns the unique id of the node'''
        return super().get_id()

    def get_name(self):
        '''returns the (non-unique) string identifier of the node'''
        return super().get_name()

    def get_throughput(self):
        '''returns the amount of fuel that can pass through the node in one time step'''
        return self.throughput

    def get_storage_capacity(self):
        '''returns the amount of fuel that can remain in the node at the end of one time step'''
        return self.storage_capacity

    def get_supply(self, include_storage = False):
        '''returns the supply of fuel provided by the node, plus the current storage of fuel if include_storage==True'''
        if include_storage:
            return self.supply + self.current_storage
        else:
            return self.supply

    def set_supply(self, new_supply):
        '''sets the amount of supply provided by the node (distinct from storage) and returns the updated supply'''
        self.supply = new_supply
        return self.supply
    
    def get_demand(self):
        '''returns the amount of fuel demanded by the node at the end of one time step'''
        return self.demand

    def set_demand(self, new_demand):
        '''sets the amount of fuel demanded by the node at the end of one time step and returns the updated demand'''
        self.demand = new_demand
        return self.demand

    def get_current_storage(self):
        '''returns the amount of fuel already stored by the node'''
        return self.current_storage

    def set_current_storage(self, new_current_storage):
        '''sets the amount of fuel already stored by the node and returns the updated storage'''
        self.current_storage = new_current_storage
        return self.current_storage

    def get_resupply(self):
        '''[LEGACY]'''
        return self.resupply

    def get_geometry(self):
        '''returns an object encapsulating where the node is (likely a point or None)'''
        return super().get_geometry()

    def get_location(self):
        '''returns where the node is in (Long, Lat) coordinates (in arbitrary coordinate system)'''
        return self.location

    def get_risks(self):
        '''returns the list of risks applying to the node'''
        return super().get_risks()

    def to_dict(self):
        '''returns the node serialized as a dictionary'''
        return {
            "id" : self.id,
            "name" : self.name,
            "type" : self.type,
            "throughput" : self.throughput,
            "storage capacity" : self.storage_capacity,
            "supply" : self.supply,
            "demand" : self.demand,
            "current storage" : self.current_storage,
            "resupply" : self.resupply,
            "location" : self.location,
            "risks" : self.risks,
            "geometry" : self.geometry
        }

class Edge(GraphComponent):
    '''
            Edges consist of
                id - a unique identifier
                name - a (non-unique) string identifier
                flow - [LEGACY]
                type - a field identifying the type, found in constants
                start - the id of the node at the beginning of the edge (tail)
                capacity - the max flow across the edge in one time step
                end - the id of the node at the end of the edge (head)
                risks - the set of risks applying to the edge
                geometry - an object encapsulating where the edge is (likely a line or None)
    '''

    def __init__(self, id = None, start = None, end = None, name = '', flow = 0, type = constants.EDGE_TYPE_DEFAULT, capacity = 0, risks = [], geometry = None):
        self.id = id
        self.name = name
        self.flow = flow
        self.type = type
        self.start = start
        self.capacity = capacity
        self.end = end
        self.risks = risks
        self.geometry = geometry

    def copy(self):
        return Edge(self.id, self.start, self.end, self.name, self.flow, self.type, self.capacity, self.risks, self.geometry)

    def get_type(self):
        '''returns the constant type of the edge'''
        return super().get_type()

    def get_id(self):
        '''returns the unique id of the edge'''
        return super().get_id()

    def get_name(self):
        '''returns the (non-unique) string identifier of the edge'''
        return super().get_name()

    def get_capacity(self):
        '''returns the max flow across the edge in one time step'''
        return self.capacity

    def set_capacity(self, new_capacity):
        '''sets the max flow across the edge in one time step and returns the new capacity'''
        self.capacity = new_capacity
        return self.capacity

    def get_flow(self):
        '''[LEGACY]'''
        return self.flow

    def set_flow(self, new_flow):
        '''[LEGACY]'''
        self.flow = new_flow

    def get_start(self):
        '''returns the id of the node at the beginning of the edge (tail)'''
        return self.start

    def get_end(self):
        '''returns the id of the node at the end of the edge (head)'''
        return self.end

    def get_geometry(self):
        '''returns an object encapsulating where the edge is (likely a line or None)'''
        return super().get_geometry()

    def get_risks(self):
        '''returns the list of risks applying to the edge'''
        return super().get_risks()

    def to_dict(self):
        '''returns the edge serialized as a dictionary'''
        return {
            "id" : self.id,
            "name" : self.name,
            "type" : self.type,
            "start" : self.start,
            "end" : self.end,
            "flow" : self.flow,
            "capacity" : self.capacity,
            "risks" : self.risks,
            "geometry" : self.geometry
        }

class Graph:
    '''
            Graphs consist of
                nodes - a dictionary of (id, Node) pairs that map unique ids to Node objects
                edges - a dictionary of (id, Edge) pairs that map unique ids to Edge objects
                risks - a dictionary of (id, Risk) pairs that map unique ids to Risk objects
    '''

    def __init__(self, nodes = None, edges = None, risks = None, filename = None):
        '''
            Initialize a new Graph object, either from a filename (if provided),
            or by passing in existing nodes, edges, and risks objects
        '''

        if filename:
            self.nodes = {}
            self.edges = {}
            self.risks = {}
            with open(filename) as f:
                g = json.load(f)
                try:
                    for obj in g['graph']:
                        match obj['type'][0]:
                            case 'node':
                                self.nodes[obj['id']] = Node(id = obj['id'], name = obj['name'], type = obj['type'], 
                                throughput=obj['throughput'], storage_capacity=obj['storage capacity'],
                                supply = obj['supply'], demand = obj['demand'], current_storage = obj['current storage'],
                                resupply = obj['resupply'], location = obj['location'], risks = obj['risks'])
                            case 'edge':
                                self.edges[obj['id']] = Edge(id = obj['id'], name = obj['name'], start = obj['start'], end = obj['end'],
                                type = obj['type'], flow = obj['flow'], capacity = obj['capacity'],
                                risks = obj['risks'])
                            case 'risk':
                                match obj['type'][1]:
                                    case 'list':
                                        self.risks[obj['id']] = Risk(id = obj['id'], name = obj['name'], description = obj['description'], 
                                        type = obj['type'], affected_objects = obj['affected_objects'], probability = obj['probability'], 
                                        impact = obj['impact'], target_ids = obj['target_ids'])
                                    case 'type':
                                        self.risks[obj['id']] = Risk(id = obj['id'], name = obj['name'], description = obj['description'],
                                        type = obj['type'], affected_objects = obj['affected_objects'], probability = obj['probability'],
                                        impact = obj['impact'], target_types = obj['target_types'])
                                    case 'location':
                                        self.risks[obj['id']] = Risk(id = obj['id'], name = obj['name'], description = obj['description'],
                                        type = obj['type'], probability = obj['probability'], impact = obj['impact'], shape = obj['shape'],
                                        location = obj['location'], geometry = shp.geometry.shape(obj['geometry']))

                except KeyError:
                    print("Key Error: JSON incorrectly formatted")
        else:
            self.nodes = nodes
            self.edges = edges
            self.risks = risks

        '''a dict mapping a node's id to the list of edge ids for all edges that start at the node id'''
        self.edges_by_source = dict([(k, []) for k in self.nodes])
        for k, edge in self.edges.items():
            self.edges_by_source[edge.start].append(k)

    def get_node(self, id):
        '''returns the Node in nodes by id;
            fails silently and returns None if id not in self.nodes.keys'''
        try:
            return self.nodes[id]
        except KeyError:
            return None

    def get_all_nodes(self):
        '''returns the dictionary of (id, Node) pairs that map unique ids to Node objects'''
        return self.nodes

    def get_edge(self, id):
        '''returns the Edge in edges by id;
            fails silently and returns None if id not in self.edges.keys'''
        try:
            return self.edges[id]
        except KeyError:
            return None

    def get_edges(self, start_id, end_id):
        '''returns all the Edge objects in edges from start_id to end_id;
            fails silently and returns [] if start_id is not in self.edges.keys'''
        try:
            return [edge for edge in self.get_edges_from_start(start_id) if edge.end == end_id]
        except KeyError:
            return []

    def get_risk(self, risk_id):
        return self.risks[risk_id]

    def get_all_edges(self):
        '''returns the dictionary of (id, Edge) pairs that map unique ids to Edge objects'''
        return self.edges

    def flatten(self, plan = None):
        '''returns the graph serialized as a dictionary;
            nodes, edges, and risks are each serialized themselves using their flatten() methods'''
        serial = {'graph' : [n.to_dict() for k, n in self.nodes.items()] + [e.to_dict() for k, e in self.edges.items()]
                        + [r.to_dict() for k, r in self.risks.items()]}

        if plan:
            serial['plan'] = plan

        return serial

    def get_edges_from_start(self, start_id):
        '''returns a list containing all of the Edge ids in edges that start at start_id;
            fails silently and returns [] if start_id not in self.edges.keys'''
        try:
            return self.edges_by_source[start_id]
        except KeyError:
            return None

    def cut_edges(self, ids = None, start_id = None, end_id = None):
        '''removes all edges from the Graph whose ids appear in ids or
            that start at start_id and end at end_id; this is useful for
            removing edges in a graph with cycles, without having to 
            modify any input files directly'''

        if ids:
            for id in ids:
                del self.edges[id]
        else:
            edges = self.edges_by_source[start_id]
            for e in edges:
                if self.edges[e].end == end_id:
                    del self.edges[e]
                    self.edges_by_source[start_id].remove(e)

    def copy(self):
        '''returns a deep copy of the Graph object'''

        def dict_copy(d):
            c = {}
            for k, v in d.items():
                c[k] = v.copy()
            return c

        return Graph(dict_copy(self.nodes), dict_copy(self.edges), dict_copy(self.risks))
        
    @staticmethod
    def geometry_from_points(shape, points, outer_distance = 0, spherical_projection = constants.DEFAULT_SPHERICAL_PROJECTION, flat_projection = constants.DEFAULT_FLAT_PROJECTION):
        '''
            Given a string shape from {'box', 'circle', 'point', 'line', 'polygon'}, return a shapely
            geometry object corresponding to that shpae using the points in points; use the given
            sphecerical and flat projections or the default appearing in constants;

            Note: outer_distance is the radius in METERS of the circle created if shape=='circle'
        '''
        
        flat_crs = pyproj.CRS(flat_projection)
        spherical_crs = pyproj.CRS(spherical_projection)
        spherical_to_flat = pyproj.Transformer.from_crs(spherical_crs, flat_crs, always_xy=True).transform
        flat_to_spherical = pyproj.Transformer.from_crs(flat_crs, spherical_crs, always_xy=True).transform
        swap = lambda x, y : (y, x)
        
        match shape.lower():
            case 'box':
                geometry = shp.geometry.box(*points)
            case 'circle':
                geometry = shp.geometry.Point(*points)
                geometry = shp.ops.transform(swap, geometry)
                geometry = shp.ops.transform(spherical_to_flat, geometry)
                geometry = geometry.buffer(outer_distance)
                geometry = shp.ops.transform(flat_to_spherical, geometry)
                geometry = shp.ops.transform(swap, geometry)
            case 'point':
                geometry = shp.geometry.Point(*points)
            case 'line':
                geometry = shp.geometry.LineString(points)
            case 'polygon':
                geometry = shp.geometry.Polygon(*points)
                
        return geometry
        
    def compute_location_risk(self, risks = None, risk_metric = constants.STANDARD_RISK):
        '''Given list of Risk objects, risks, 
            compute whether each risk intersects each node and each edge in Graph
            and update IN-PLACE that node or edge's list of risks to include the
            id, probability, and impact of that risk'''

        if not risks:
            risks = self.risks

        lrisks = [lrisk for risk_id, lrisk in self.risks.items() if 'location' in lrisk.type]

        for node_id, node in self.nodes.items():
            if not node.geometry:
                node.geometry = self.geometry_from_points('Point', node.location)
            for lrisk in lrisks:
                if lrisk.geometry.intersects(node.geometry):
                    node.risks.append({ 'id' : lrisk.id,
                                        'type' : 'location',
                                        'probability' : lrisk.probability,
                                        'impact' : lrisk.impact})
                                        
        for edge_id, edge in self.edges.items():
            if not edge.geometry:
                p1 = self.nodes[edge.start]
                p2 = self.nodes[edge.end]
                edge.geometry = self.geometry_from_points('Line', [p1.location, p2.location])
            for lrisk in lrisks:
                if lrisk.geometry.intersects(edge.geometry):
                    edge.risks.append({ 'id' : lrisk.id,
                                        'type' : 'location',
                                        'probability' : lrisk.probability,
                                        'impact' : lrisk.impact})

    def populate_geometry(self):
        for node_id, node in self.nodes.items():
            node.geometry = self.geometry_from_points('Point', node.location)
        for edge_id, edge in self.edges.items():
            location = [self.nodes[edge.start].location, self.nodes[edge.end].location]
            edge.geometry = self.geometry_from_points('Line', location)

    def clear_risks(self):
        '''Clear IN-PLACE all risks for all nodes and edges appearing in the Graph object
            
            Note: DOES NOT clear the list of risks contained in the Graph object    
        '''

        for node in self.nodes:
            node.risks = []
        for edge in self.edges:
            edge.risks = []

    def to_adj_list(self):
        '''
            Returns a dictionary corresponding to an adjacency list representation of the Graph
            object, where each Node id in the graph maps to a list of the Node ids of all Node
            objects that are at the end of an Edge starting at the key Node
        '''

        g = {}
        for k, node in self.nodes.items():
            g[k] = [self.get_edge(edge).end for edge in self.get_edges_from_start(node.get_id())]
        return g

    def topological_sort(self):
        '''
            Returns a partially ordered list of Node ids, where id_1 appears before id_2 if and
            only if there is no path from edges starting at id_2 and ending at id_1 (i.e.,
            id_1 is upstream of id_2)

            Note: The graph must be a directed acyclic graph (DAG) or there is no notion of a
            topological ordering of the nodes and an Exception will be raised
        '''

        adj_list = self.to_adj_list()
        unmarked_nodes = [id for id in self.nodes]
        temporary_mark = dict([(id, False) for id in unmarked_nodes])
        sort = []

        def visit(id, tabs = None):
            if tabs:
                print('\t' * tabs + str(id))
            if id not in unmarked_nodes:
                return
            elif temporary_mark[id]:
                raise Exception("Graph is not a DAG; Topological Sort makes no sense!")

            temporary_mark[id] = True
            for end in adj_list[id]:
                visit(end)
            temporary_mark[id] = False

            unmarked_nodes.remove(id)
            sort.insert(0, id)

        while unmarked_nodes:
            cur = unmarked_nodes[0]
            visit(cur)

        return sort

    def flow(self, edge, amount, with_risk = False):
        '''
            Helper method for simulation;

            Simulates flow of amount along edge (either by reference or id)
            If with_risk is True, perform flow in three stages:
                1. Simulate risk along the edge, by iterating through
                    each Risk affecting the edge and reducing the flow by
                    impact based on a biased coin flip
                2. Simulate risk on the end node, by iterating through each
                    Risk affecting the end node and reducing the incoming flow 
                    (*which may have already been reduced from amount in stage 1*)
                    by impact based on a biased coin flip
            Otherwise, the full amount is flowed, without regard to risk;

            Returns the amount that actually flows along the edge

            Fails silently and returns 0 flow if the edge does not exist
        '''
        if not isinstance(edge, Edge):
            try:
                edge = self.edges[edge]
            except KeyError:
                return 0
        try:
            start = self.nodes[edge.start]
            end = self.nodes[edge.end]
        except KeyError:
            return 0

        # Stage 1: Apply risk to start node (How much flows out?)
        #   Note: 'amount' leaves the start node, but may be lost
        #           due to risk (captured in 'flow_out')
        #flow_out = start.get_supply() - start.set_supply(max(0, start.get_supply() - amount))
        #flow_out += start.get_current_storage() - start.set_current_storage(max(0, start.get_current_storage() - amount + flow_out))
        flow_out = amount
        temp = flow_out
        temp -= start.get_supply()
        start.set_supply(max(0, start.get_supply() - flow_out))
        start.set_current_storage(start.get_current_storage() - temp)

        # Stage 2: Apply risk to edge (How much flows along?)
        flow_across = flow_out
        if with_risk:
            for risk in edge.risks:
                if random.random() < risk['probability']:
                    print("Simulation Impact: Flow across reduced from", flow_across)
                    flow_across *= (1.0 - risk['impact'])
                    print("\tto", flow_across)
                else:
                    print("Simulation Impact: Flow across maintained", flow_across)

        # Stage 3: Apply risk to end node (How much flows in?)
        flow_in = flow_across
        if with_risk:
            for risk in end.risks:
                if random.random() < risk['probability']:
                    print("Simulation Impact: Flow in reduced from", flow_in)
                    flow_in *= (1.0 - risk['impact'])
                    print("\tto", flow_in)
                else:
                    print("Simulation Impact: Flow in maintained at", flow_in)

        # The amount of fuel entering a node cannot exceed its throughput;
        #   Currently, extra fuel is simply "lost"
        #   Note that for current networks, throughput is normally substantially 
        #   higher than any intended flow
        if flow_in > end.get_throughput():
            print("WARNING: Fuel entering Node '%s' (%s) of value %3s exceeds throughput of %3s" % (end.get_name(), end.get_id(), flow_in, end.get_throughput()))
        flow_in = min(end.get_throughput(), flow_in)

        # Flow satisfies demand, then goes into current storage
        demand = end.get_demand()
        print("Flowing %3s from %s (%s) to %s (%s) to satisfy demand of %s" % (flow_in, start.get_name(), start.get_id(), end.get_name(), end.get_id(), demand))
        if demand > 0:
            overflow = max(0, flow_in - demand)
            end.set_demand(max(0, demand - flow_in))
            end.set_current_storage(end.get_current_storage() + overflow)
        else:
            end.set_current_storage(end.get_current_storage() + flow_in)
        return flow_in

    def simulate(self, plan, with_risk = False):
        '''Returns a graph which is equivalent to the current graph simulated with the plan,
            which is a dictionary mapping edge ids to the flow along the edges;
            
            this will (and should!) raise an error if an edge appears in the Graph but not
            in the plan    
        '''
        
        ret = self.copy()

        ordering = ret.topological_sort()
        #ordering.reverse()

        print("--------BEGINNING SIMULATION--------")

        for node_id in ordering:
            edges = ret.get_edges_from_start(node_id)
            for edge in edges:
                ret.flow(edge, plan[edge], with_risk)

        for node_id, node in ret.get_all_nodes().items():
            if node.get_demand() > 0:
                print("WARNING: Unmet demand for Node '%s' (%s); Still needs %3s" % (node.get_name(), node.get_id(), node.get_demand()))
            if node.get_current_storage() < 0:
                print("WARNING: Negative storage for '%s' (%s); Current storage of %3s" % (node.get_name(), node.get_id(), node.get_current_storage()))

        print("--------ENDING SIMULATION--------")

        return ret

    def export_json(self, filename, plan = None):
        '''Writes the Graph, serialized in JSON, to filename'''

        with(open(filename, 'w', encoding='utf-8')) as f:
            geojson.dump(self.flatten(plan), f, ensure_ascii=False, indent = 4)
        
    def to_json(self, plan = None):
        '''returns a string consisting of the Graph serialized in JSON'''

        return geojson.dumps(self.flatten(plan), ensure_ascii=False, indent = 4)

    def compute_plan(self, use_expected_risk = False):
        '''Returns a dictionary mapping 'Edge' ids to Integer flows'''

        # Operate on copy
        graph_copy = self.copy()

        # Round 1
        first_round_solver = pywrapgraph.SimpleMinCostFlow()

        # Maybe not the most elegant solution, but fast and fine for this size of network
        edges_to_arcs_first = {}
        arcs_to_edges_first = {}
        expected_edge_capacity = {}
        edge_risks = {}
        for node_id, node in graph_copy.nodes.items():
            first_round_solver.SetNodeSupply(node_id, node.get_supply(include_storage = True) - node.get_demand())
        for edge_id, edge in graph_copy.edges.items():
            # Compute expected risk on edges
            expected_edge_capacity[edge_id] = 0
            edge_risks[edge_id] = 0
            if use_expected_risk:
                for r in edge.get_risks():
                    edge_risks[edge_id] += self.get_risk(r['id']).get_probability() * self.get_risk(r['id']).get_impact()
            expected_edge_capacity[edge_id] = int((1 - edge_risks[edge_id]) * edge.get_capacity())
            edges_to_arcs_first[edge_id] = first_round_solver.AddArcWithCapacityAndUnitCost(edge.get_start(), edge.get_end(), expected_edge_capacity[edge_id], 0)
            arcs_to_edges_first[edges_to_arcs_first[edge_id]] = edge_id

        if first_round_solver.SolveMaxFlowWithMinCost():
            print('Max flow:', first_round_solver.MaximumFlow())
        
        # for arc in range(first_round_solver.NumArcs()):
        #     tail = graph_copy.get_node(first_round_solver.Tail(arc))
        #     head = graph_copy.get_node(first_round_solver.Head(arc))

        #     flow = first_round_solver.Flow(arc)
        #     head.set_current_storage(head.get_current_storage() + flow - head.get_demand())
        #     head.set_demand(max(0, head.get_demand() - flow))

        #     storage = tail.get_current_storage()
        #     storage -= flow
        #     tail.set_current_storage(max(0, storage))
        #     tail.set_supply(tail.get_supply() + storage)

        #     # Need to reduce capacity along edge before round 2, to account for flow that has already gone across
        #     expected_edge_capacity[arcs_to_edges_first[arc]] -= flow

        # Round 2
        # second_round_solver = pywrapgraph.SimpleMinCostFlow()
        # edges_to_arcs_second = {}
        # for node_id, node in graph_copy.nodes.items():
        #     second_round_solver.SetNodeSupply(node_id, int(node.get_supply() + node.get_current_storage() - node.get_storage_capacity()))
        # for edge_id, edge in graph_copy.edges.items():
        #     edges_to_arcs_second[edge_id] = second_round_solver.AddArcWithCapacityAndUnitCost(edge.get_start(), edge.get_end(), expected_edge_capacity[edge_id], 0)

        # if second_round_solver.SolveMaxFlowWithMinCost():
        #     print('Max flow:', second_round_solver.MaximumFlow())

        # Reconcile plans
        #   Round 2 will never flow backwards, so can safely sum flow along edges
        plan = dict([(e, 0) for e in graph_copy.get_all_edges().keys()])

        for edge in plan.keys():
            plan[edge] += first_round_solver.Flow(edges_to_arcs_first[edge])
            #plan[edge] += second_round_solver.Flow(edges_to_arcs_second[edge])
            #plan[edge] /= (1 - edge_risks[edge])
            print('%1s -> %1s   %3s  / %3s -- %3s -- %3s' %
                    (first_round_solver.Tail(edges_to_arcs_first[edge]), first_round_solver.Head(edges_to_arcs_first[edge]), plan[edge],
                    self.get_edge(edge).get_capacity(), self.get_edge(edge).get_capacity() * (1 - edge_risks[edge]), first_round_solver.Supply(first_round_solver.Head(edges_to_arcs_first[edge]))))
            # print('%1s -> %1s   %3s  / %3s -- %3s' %
            #         (second_round_solver.Tail(edges_to_arcs_second[edge]), second_round_solver.Head(edges_to_arcs_second[edge]), plan[edge],
            #         self.get_edge(edge).get_capacity(), self.get_edge(edge).get_capacity() * (1 - edge_risks[edge])))
        
        return plan

class Risk:
    '''
            Risks consist of
                id - a unique identifier
                name - a (non-unique) string identifier
                description - a (non-unique) string description
                type - a (non-standardized) string type
                affected_objects - a list of the Node or Edge objects affected by the Risk
                shape - a string description of the shape of the risk
                location - the coordinates of the risk, consisting of one or more (Lat, Long) tuples
                probability - the likelihood of the risk occuring
                impact - the proportion of fuel lost if the risk occurs
                target_types - the type of objects impacted by the risk (some subset of the Nodes and Edges keyed by their type fields)
                target_ids - the ids of objects affected by the Risk (e.g., Node 12 or Edge 27)
                geometry - a shapely object encapsulating where the Risk is located

    '''


    def __init__(self, id = None, name = '', description = '', type = '', 
                        affected_objects = [], shape = None, location = None, probability = 0.0, impact = 0.0,
                        target_types = [], target_ids = [], geometry = None):
        self.id = id
        self.name = name
        self.description = description
        self.type = type
        self.affected_objects = affected_objects
        self.shape = shape
        self.location = location
        self.probability = probability
        self.impact = impact
        self.target_types = target_types
        self.target_ids = target_ids
        self.geometry = geometry

    def copy(self):
        return Risk(id = self.id, name = self.name, description = self.description, type = self.type,
            affected_objects = self.affected_objects, shape = self.shape, location = self.location,
            probability = self.probability, impact = self.impact, target_types = self.target_types,
            target_ids = self.target_ids, geometry = self.geometry)

    def get_probability(self):
        return self.probability

    def get_impact(self):
        return self.impact

    def to_dict(self):
        '''returns the risk serialized as a dictionary'''
        ret = {
            'id' : self.id,
            'name' : self.name,
            'description' : self.description,
            'type' : self.type,
            'probability' : self.probability,
            'impact' : self.impact,
        }
        
        match self.type[1]:
            case 'location':
                ret['shape'] = self.shape
                ret['location'] = self.location
                ret['geometry'] = self.geometry
            case 'type':
                ret['affected_objects'] = self.affected_objects
                ret['target_types'] = self.target_types
            case 'list':
                ret['affected_objects'] = self.affected_objects
                ret['target_ids'] = self.target_ids
                
        return ret