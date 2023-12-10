import cdt
import networkx as nx

try:
   from .synthetic_dataset import SyntheticDataset
except ImportError:
   from synthetic_dataset import SyntheticDataset

def getSacksDataset():
    s_data, s_graph = cdt.data.load_dataset("sachs")
    B = nx.to_numpy_array(s_graph)
    return SyntheticDataset(s_data.to_numpy(), B)



def getTuebingen():
    s_data, s_graph = cdt.data.load_dataset("tuebingen")
    B = nx.to_numpy_array(s_graph)
    return SyntheticDataset(s_data.to_numpy(), B)


