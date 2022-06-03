# 1. Read in state of network as file
#   1a. If flow along edges already included in state, continue
#   1b. If flow not included, read in other file specifying flow
# 2. Validate flow w.r.t. edges
#       A----B----C
#       Eq 2a. Flow(A,B) <= Capacity(A,B)
# 3. Validate flow w.r.t. throughput and storage capacity
#       A----B----C
#       Eq 3a. Flow(A,B) - Flow(B,C) <= Available_Storage(B)
#               Flow through B can't "leave behind" more than the available storage
#       Eq 3b. Flow(A,B) <= Throughput(B)
#               Flow from A to B must be less than throughput of B, to be "processed"
#       TODO: Is there also a Throughput(B) constraint for Flow(B, C)
# 4. Perform flow updates
#       For each edge (a, b):
#           If source(a):
#               Supply(a) <-- Supply(a) - Flow(a, b)
#           Else:
#               Storage_Capacity(a) <-- Storage_Capacity(a) - Flow(a, b)
#           Storage_Capacity(b) <-- Storage_Capacity(b) + Flow(a, b)
#       TODO: Do we allow instantaneous violations of storage capacity constraints,
#             as long as the network is valid at the end?
# 5. Output new network state as file

import graph_pd

def validate_flow(graph, flow):
    # Validate flow relative to network--assumes flow is a list of ((source, sink), amount) dictionaries
    for f in flow:
        src, sink, amt = f['source'], f['sink'], f['amount']
        e = graph.get_edge(source = src, sink = sink)
        src_node = graph.get_node(src)
        sink_node = graph.get_node(sink)
        if not e:
            raise Exception("Flow incorrectly formatted: Refers to edge not contained in graph.")
        if e.capacity < amt:
            raise Exception("Flow value error: Capacity along edge less than demanded flow.")

        if not (src_node or sink_node):
            raise Exception("Flow incorrectly formatted: Refers to nodes not contained in graph.")

        # This is really inefficient; need to decide if its smarter to just iterate over nodes to check this constraint
        incoming_flow = [fl['amount'] for fl in flow if fl['sink'] == sink]
        outgoing_flow = [fl['amount'] for fl in flow if fl['source'] == sink]
        total_incoming_flow = sum(incoming_flow)
        total_outgoing_flow = sum(outgoing_flow)

        if total_incoming_flow - total_outgoing_flow > sink_node.available_storage:
            raise Exception("Flow value error: Not enough available storage for incoming fuel.")

        if total_incoming_flow > sink_node.throughput:
            raise Exception("Flow value error: Flow into node exceeds throughput.")


def simulate(graph, flow, probabilistic = False):
    '''Takes a graph and flow as input and returns a deep-copy of graph with flow performed
        If probabilistic is True, risk is taken into account when simulating flow.'''

    #validate_flow(graph, flow)

    output = graph.copy()

    # Sort intended flow by topological ordering of graph
    top_sort = graph.topological_sort()
    flow.sort(key=lambda f: top_sort.index(f['source']))

    for f in flow:
        output.flow(f['source'], f['sink'], f['amount'], with_risk = probabilistic)

    return output


