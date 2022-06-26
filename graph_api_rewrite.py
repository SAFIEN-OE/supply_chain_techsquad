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
    
    def get_demand(self):
        return self.demand

    def set_demand(self, new_demand):
        self.demand = new_demand

    def get_current_storage(self):
        return self.current_storage

    def get_resupply(self):
        return self.resupply

    def get_geometry(self):
        return super().get_geometry()

    def get_location(self):
        return self.location

    def get_risks(self):
        return super().get_risks()

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

class Graph:

    def __init__(self, nodes = None, edges = None):
        self.nodes = nodes
        self.edges = edges

        self.edges_by_source = dict([(n.get_id(), []) for n in self.nodes])
        for edge in self.edges:
            self.edges_by_source[edge.start.get_id()].append(edge)

    def get_node(self, id):
        return self.nodes[id]

    def get_all_nodes(self):
        return self.nodes

    def get_edge(self, id):
        return self.edges[id]

    def get_edges(self, start_id, end_id):
        return [edge for edge in self.get_edges_from_start(start_id) if edge.end.id == end_id]

    def get_all_edges(self):
        return self.edges

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
                if e.end.id == end_id:
                    # CHECK: Is e a live reference into edges? Might need to change below
                    del e

    def copy(self):
        return Graph(self.nodes.copy(), self.edges.copy())

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
            if id not in unmarked_nodes:
                return
            elif temporary_mark[id]:
                raise Exception("Graph is not a DAG; Topological Sort makes no sense!")

            temporary_mark[id] = True
            for end in adj_list[id]:
                visit(end.get_id())
            temporary_mark[id] = False

            unmarked_nodes.remove(id)
            sort.insert(0, id)

        while unmarked_nodes:
            cur = unmarked_nodes[0]
            visit(cur)

        return sort
