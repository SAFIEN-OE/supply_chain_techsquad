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

    def get_type(self):
        return super().get_type()

    def get_id(self):
        return super().get_id()

    def get_name(self):
        return super().get_name()

    def get_throughput(self):
        return self.throughput

    def get_storage_capacity(self):
        return self.storage_capacity

    def get_supply(self):
        return self.supply

    def set_supply(self, new_supply):
        self.supply = new_supply
        return self.supply
    
    def get_demand(self):
        return self.demand

    def set_demand(self, new_demand):
        self.demand = new_demand
        return self.demand

    def get_current_storage(self):
        return self.current_storage

    def set_current_storage(self, new_current_storage):
        self.current_storage = new_current_storage
        return self.current_storage

    def get_resupply(self):
        return self.resupply

    def get_geometry(self):
        return super().get_geometry()

    def get_location(self):
        return self.location

    def get_risks(self):
        return super().get_risks()

    def to_dict(self):
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

    def get_type(self):
        return super().get_type()

    def get_id(self):
        return super().get_id()

    def get_name(self):
        return super().get_name()

    def get_capacity(self):
        return self.capacity

    def set_capacity(self, new_capacity):
        self.capacity = new_capacity

    def get_flow(self):
        return self.flow

    def set_flow(self, new_flow):
        self.flow = new_flow

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_geometry(self):
        return super().get_geometry()

    def get_risks(self):
        return super().get_risks()

    def to_dict(self):
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

    # TODO: add support for additional properties packaged with nodes and edges
    def __init__(self, nodes = None, edges = None, risks = None, filename = None):

        if filename:
            self.nodes = []
            self.edges = []
            self.risks = []
            with open(filename) as f:
                g = json.load(f)
                try:
                    for obj in g['graph']:
                        match obj['type'][0]:
                            case 'node':
                                self.nodes.append(Node(id = obj['id'], name = obj['name'], type = obj['type'], 
                                throughput=obj['throughput'], storage_capacity=obj['storage capacity'],
                                supply = obj['supply'], demand = obj['demand'], current_storage = obj['current storage'],
                                resupply = obj['resupply'], location = obj['location'], risks = obj['risks']))
                            case 'edge':
                                self.edges.append(Edge(id = obj['id'], name = obj['name'], start = obj['start'], end = obj['end'],
                                type = obj['type'], flow = obj['flow'], capacity = obj['capacity'],
                                risks = obj['risks']))
                            # case 'risk':
                            #     match obj['type'][1]:
                            #         case 'location':
                            #             location = (obj['Latitude'], obj['Longitude'], obj['Inner Distance'], obj['Outer Distance']) if obj['shape'] == 'Circle'
                            #                     else (obj['Top Left Lat'], obj['Top Left Long'], obj['Bottom Right Lat'], obj['Bottom Right Long'])
                            #         case 'type':
                                    
                            #         case 'list':

                except KeyError:
                    print("Key Error: JSON incorrectly formatted")
        else:
            self.nodes = nodes
            self.edges = edges
            self.risks = risks

        self.edges_by_source = dict([(n.get_id(), []) for n in self.nodes])
        for edge in self.edges:
            self.edges_by_source[edge.start].append(edge)

    def get_node(self, id):
        return self.nodes[id]

    def get_all_nodes(self):
        return self.nodes

    def get_edge(self, id):
        return self.edges[id]

    def get_edges(self, start_id, end_id):
        return [edge for edge in self.get_edges_from_start(start_id) if edge.end == end_id]

    def get_all_edges(self):
        return self.edges

    def flatten(self):
        return {'graph' : [n.to_dict() for n in self.nodes] + [e.to_dict() for e in self.edges]
                        + [r.to_dict() for r in self.risks]}

    # TODO: Change to assume edges hold references directly to nodes
    def get_edges_from_start(self, start_id):
        edges = self.edges_by_source[start_id]
        ret = []

        for e in edges:
            ret.append(e)

        return ret

    def cut_edges(self, ids = None, start_id = None, end_id = None):

        if ids:
            for id in ids:
                del self.edges[id]
        else:
            edges = self.edges_by_source[start_id]
            for e in edges:
                if e.end == end_id:
                    # CHECK: Is e a live reference into edges? Might need to change below
                    self.edges.remove(e)
                    self.edges_by_source[start_id].remove(e)

    def copy(self):
        return Graph(self.nodes.copy(), self.edges.copy(), self.risks.copy())
        
    @staticmethod
    def geometry_from_points(shape, points, outer_distance = 0, spherical_projection = constants.DEFAULT_SPHERICAL_PROJECTION, flat_projection = constants.DEFAULT_FLAT_PROJECTION):
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
        
    def compute_location_risk(self, lrisks, risk_metric = constants.STANDARD_RISK):
        for node in self.nodes:
            if not node.geometry:
                node.geometry = geometry_from_points('Point', node.location)
            for lrisk in lrisks:
                if lrisk['geometry'].intersects(node.geometry):
                    node.risks.append({'type' : 'location',
                                        'probability' : lrisk['probability'],
                                        'impact' : lrisk['impact']})
                                        
        for edge in self.edges:
            if not edge.geometry:
                p1 = next((n for n in self.nodes if n.get_id() == edge.start))
                p2 = next((n for n in self.nodes if n.get_id() == edge.end))
                edge.geometry = geometry_from_points('Line', [p1.location, p2.location])
            for lrisk in lrisks:
                if lrisk['geometry'].intersects(edge.geometry):
                    edge.risks.append({'type' : 'location',
                                        'probability' : lrisk['probability'],
                                        'impact' : lrisk['impact']})
                                        
    def clear_risks(self):
        for node in self.nodes:
            node.risks = []
        for edge in self.edges:
            edge.risks = []

    def to_adj_list(self):
        g = {}
        for node in self.nodes:
            g[node.get_id()] = [edge.end for edge in self.get_edges_from_start(node.get_id())]
        return g

    def topological_sort(self):
        adj_list = self.to_adj_list()
        unmarked_nodes = [n.get_id() for n in self.nodes]
        temporary_mark = dict([(id, False) for id in unmarked_nodes])
        sort = []

        def visit(id):
            #g.cut_edges(start_id = 39, end_id = 26)
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

        if isinstance(edge, int):
            edge = next((e for e in self.edges if e.get_id() == edge))

        start = next((n for n in self.nodes if n.get_id() == edge.start))
        end = next((n for n in self.nodes if n.get_id() == edge.end))

        getter = start.get_current_storage
        setter = start.set_current_storage
        if start.get_supply() > 0:
            getter = start.get_supply
            setter = start.set_supply

        # Stage 1: Apply risk to start node (How much flows out?)
        flow_out = amount
        for risk in start.risks:
            if random.random() < risk['probability']:
                print("Simulation Impact: Flow out reduced from", flow_out)
                flow_out *= (1.0 - risk['impact'])
                print("\tto", flow_out)
            else:
                print("Simulation Impact: Flow out maintained at", flow_out)

        flow_out = min(getter(), flow_out, edge.get_capacity())
        setter(max(0, getter() - amount))

        # Stage 2: Apply risk to edge (How much flows along?)
        flow_across = flow_out
        for risk in edge.risks:
            if random.random() < risk['probability']:
                print("Simulation Impact: Flow across reduced from", flow_across)
                flow_across *= (1.0 - risk['impact'])
                print("\tto", flow_across)
            else:
                print("Simulation Impact: Flow across maintained", flow_across)
        edge.set_flow(edge.get_flow() - flow_out)

        # Stage 3: Apply risk to end node (How much flows in?)
        flow_in = flow_across
        for risk in end.risks:
            if random.random() < risk['probability']:
                print("Simulation Impact: Flow in reduced from", flow_in)
                flow_in *= (1.0 - risk['impact'])
                print("\tto", flow_in)
            else:
                print("Simulation Impact: Flow in maintained at", flow_in)

        flow_in = min(end.get_throughput(), flow_in)
        end.set_current_storage(end.get_current_storage() + flow_in)

        return flow_in

    def simulate(self, with_risk = False):
        
        ordering = self.topological_sort()

        for node_id in ordering:
            edges = self.get_edges_from_start(node_id)

            for edge in edges:
                self.flow(edge, edge.get_flow(), with_risk)

    def export_json(self, filename):
        with(open(filename, 'w', encoding='utf-8')) as f:
            geojson.dump(self.flatten(), f, ensure_ascii=False, indent = 4)
        
    # # Ensures each node/edge has correct linkage to associated risks    
    # def update_risks(self):
    #     for risk in self.risks:
    #         match risk.type[1].lower():
    #             case 'location':
                    
    #             case 'type':
                    
                    
    #             case 'list':
            
class Risk:
    
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

    def to_dict(self):
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