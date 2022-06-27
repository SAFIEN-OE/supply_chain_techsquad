from graph_api_rewrite import Node, Edge, Graph
import json

# n0 = Node(id = 0, location = (0, 0))
# n1 = Node(id = 1, location = (1, 1))
# n2 = Node(id = 2, location = (2, 2))

# e0 = Edge(id = 0, start = 1, end = 0)
# e1 = Edge(id = 1, start = 0, end = 2)

# g = Graph(nodes = [n0, n1, n2], edges = [e0, e1])

# print(g.topological_sort())

f = open("sample_network.json")
test = json.load(f)

nodes = []
edges = []

for obj in test['graph']:
    if obj['type'][0] == 'node':
        nodes.append(Node(id = obj['id'], name = obj['name'], type = obj['type'], 
        throughput=obj['throughput'], storage_capacity=obj['storage capacity'],
        supply = obj['supply'], demand = obj['demand'], current_storage = obj['current storage'],
        resupply = obj['resupply'], location = obj['location'], risks = obj['risks']))
    elif obj['type'][0] == 'edge':
        edges.append(Edge(id = obj['id'], name = obj['name'], start = obj['start'], end = obj['end'],
                            type = obj['type'], flow = obj['flow'], capacity = obj['capacity'],
                            risks = obj['risks']))

g = Graph(nodes, edges)

g.flow(edge = 0, amount = 100)

with(open('sample_out.json', 'w', encoding='utf-8')) as f:
    json.dump(g.flatten(), f, ensure_ascii=False, indent = 4)

f = open("sample_out.json")
test = json.load(f)

nodes = []
edges = []

for obj in test['graph']:
    if obj['type'][0] == 'node':
        nodes.append(Node(id = obj['id'], name = obj['name'], type = obj['type'], 
        throughput=obj['throughput'], storage_capacity=obj['storage capacity'],
        supply = obj['supply'], demand = obj['demand'], current_storage = obj['current storage'],
        resupply = obj['resupply'], location = obj['location'], risks = obj['risks']))
    elif obj['type'][0] == 'edge':
        edges.append(Edge(id = obj['id'], name = obj['name'], start = obj['start'], end = obj['end'],
                            type = obj['type'], flow = obj['flow'], capacity = obj['capacity'],
                            risks = obj['risks']))

g = Graph(nodes, edges)

print(g)