#####################################################
#                                                   #
#           Simulation & Planning Constants         #
#                                                   #
#####################################################


# RISK MEASUREMENTS
STANDARD_RISK = lambda probability, impact: probability * impact

# BASE EXCEL FILE COLUMN NAMES
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

# GIS CONSTANTS
DEFAULT_FLAT_PROJECTION = 'EPSG:3857'
DEFAULT_SPHERICAL_PROJECTION = 'EPSG:4326'

# Node Types
NODE_TYPE_USE = 'Use'
NODE_TYPE_BULK_STORAGE = 'Bulk Storage'
NODE_TYPE_PRODUCTION = 'Production'
NODE_TYPE_DEFAULT = ('node', '')

# Edge Types
EDGE_TYPE_RAILROAD = 'Railroad'
EDGE_TYPE_PIPELINE = 'Pipeline'
EDGE_TYPE_TRUCK = 'Truck'
EDGE_TYPE_SHIP = 'Ship'
EDGE_TYPE_DEFAULT = ('edge', '')