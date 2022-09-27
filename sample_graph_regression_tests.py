from graph_api_rewrite import Node, Edge, Graph
import json

## BASIC TEST 1: Manually constructing a graph

# n0 = Node(id = 0, location = (0, 0))
# n1 = Node(id = 1, location = (1, 1))
# n2 = Node(id = 2, location = (2, 2))

# e0 = Edge(id = 0, start = 1, end = 0)
# e1 = Edge(id = 1, start = 0, end = 2)

# g = Graph(nodes = [n0, n1, n2], edges = [e0, e1])

# print(g.topological_sort())

## BASIC TEST 2: Importing a JSON Network, Simulating Flow, Exporting

# g = Graph(filename = "sample_network.json")

# g.simulate(with_risk=True)

# g.export_json("sample_out.json")

## BASIC TEST 3: Importing a real JSON Network and simulating without any actual flow

# g = Graph(filename = "IN_Supply_Chain_Model.json")
# g.cut_edges(start_id = 39, end_id = 26)
# g.cut_edges(start_id = 42, end_id = 40)

# g.simulate(with_risk=True)

# g.export_json("Graph_Tests_IN_Supply_Chain_Model.json")

## BASIC TEST 4: Generating basic geometric shapes

p = Graph.geometry_from_points('Circle', [(0, 0)], outer_distance = 1000000)

print(p)