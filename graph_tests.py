from graph_api_rewrite import Node, Edge, Graph
import json

# n0 = Node(id = 0, location = (0, 0))
# n1 = Node(id = 1, location = (1, 1))
# n2 = Node(id = 2, location = (2, 2))

# e0 = Edge(id = 0, start = n1, end = n0, location = (0, 0))
# e1 = Edge(id = 1, start = n0, end = n2, location = (1, 1))

# g = Graph(nodes = [n0, n1, n2], edges = [e0, e1])

# print(g.topological_sort())

f = open("sample_network.json")
test = json.load(f)

print(test)

nodes = []
edges = []

for obj in test['graph']:
    if obj['type'][0] == 'node':
        nodes.append(Node(id = obj['id'], name = obj['name'], type = obj['type'], 
        throughput=obj['throughput'], storage_capacity=obj['storage capacity'],
        supply = obj['supply'], demand = obj['demand'], current_storage = obj['current storage'],
        resupply = obj['resupply'], location = obj['location'], risks = obj['risks']))

print(nodes)