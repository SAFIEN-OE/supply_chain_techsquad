import json
from graph_pd import Graph

class JSONEncoder(json.JSONEncoder):
    
    def default(self, obj):
        
        if isinstance(obj, Graph.Node) or isinstance(obj, Graph.Edge):
            return obj.to_json()
        elif isinstance(obj, Graph):
            return [this.default(n) for n in obj.get_all_nodes()].append(this.default(e) for e in obj.get_all_edges())
            
        return json.JSONEncoder.default(self, obj)