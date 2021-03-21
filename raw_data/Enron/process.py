import re
import os
import itertools
from collections import defaultdict
from itertools import islice, chain

import networkx as nx
import numpy as np
import pickle as pkl
from scipy.sparse import csr_matrix

from datetime import datetime
from datetime import timedelta
import dateutil.parser


def lines_per_n(f, n):
    for line in f:
        yield ''.join(chain([line], itertools.islice(f, n - 1)))

def getDateTimeFromISO8601String(s):
    d = dateutil.parser.parse(s)
    return d

if __name__ == "__main__":

    node_data = defaultdict(lambda : ())  
    with open('vis.graph.nodeList.json') as f:
        for chunk in lines_per_n(f, 5):
            chunk = chunk.split("\n")
            id_string = chunk[1].split(":")[1]
            x = [x.start() for x in re.finditer('\"', id_string)]
            id =  id_string[x[0]+1:x[1]]
            
            name_string = chunk[2].split(":")[1]
            x = [x.start() for x in re.finditer('\"', name_string)]
            name =  name_string[x[0]+1:x[1]]
            
            idx_string = chunk[3].split(":")[1]
            x1 = idx_string.find('(')
            x2 = idx_string.find(')')
            idx =  idx_string[x1+1:x2]
            
            print("ID:{}, IDX:{:<4}, NAME:{}".format(id, idx, name))
            node_data[name] = (id,idx)

    links = []
    ts = []
    with open('vis.digraph.allEdges.json') as f:
        for chunk in lines_per_n(f, 5):
            chunk = chunk.split("\n")
            
            name_string = chunk[2].split(":")[1]
            x = [x.start() for x in re.finditer('\"', name_string)]
            from_id, to_id = name_string[x[0]+1:x[1]].split("_")
            
            time_string = chunk[3].split("ISODate")[1]
            x = [x.start() for x in re.finditer('\"', time_string)]
            timestamp = getDateTimeFromISO8601String(time_string[x[0]+1:x[1]])
            ts.append(timestamp)
            links.append((from_id, to_id, timestamp))
    print (min(ts), max(ts))
    print ("# interactions", len(links))
    links.sort(key =lambda x: x[2])

    # split edges 
    SLICE_MONTHS = 2
    START_DATE = min(ts) + timedelta(200)
    END_DATE = max(ts) - timedelta(200)
    print("Spliting Time Interval: \n Start Time : {}, End Time : {}".format(START_DATE, END_DATE))

    slice_links = defaultdict(lambda: nx.MultiGraph())
    for (a, b, time) in links:
        datetime_object = time
        if datetime_object > END_DATE:
            months_diff = (END_DATE - START_DATE).days//30
        else:
            months_diff = (datetime_object - START_DATE).days//30
        slice_id = months_diff // SLICE_MONTHS
        slice_id = max(slice_id, 0)

        if slice_id not in slice_links.keys():
            slice_links[slice_id] = nx.MultiGraph()
            if slice_id > 0:
                slice_links[slice_id].add_nodes_from(slice_links[slice_id-1].nodes(data=True))
                assert (len(slice_links[slice_id].edges()) ==0)
        slice_links[slice_id].add_edge(a,b, date=datetime_object)

    # print statics of each graph
    used_nodes = []
    for id, slice in slice_links.items():
        print("In snapshoot {:<2}, #Nodes={:<5}, #Edges={:<5}".format(id, \
                            slice.number_of_nodes(), slice.number_of_edges()))
        for node in slice.nodes():
            if not node in used_nodes:
                used_nodes.append(node)
    # remap nodes in graphs. Cause start time is not zero, the node index is not consistent
    nodes_consistent_map = {node:idx for idx, node in enumerate(used_nodes)}
    for id, slice in slice_links.items():
        slice_links[id] = nx.relabel_nodes(slice, nodes_consistent_map)

    # One-Hot features
    onehot = np.identity(slice_links[max(slice_links.keys())].number_of_nodes())
    graphs = []
    for id, slice in slice_links.items():
        tmp_feature = []
        for node in slice.nodes():
            tmp_feature.append(onehot[node])
        slice.graph["feature"] = csr_matrix(tmp_feature)
        graphs.append(slice)
    
    # save
    save_path = "../../data/Enron/graph.pkl"
    with open(save_path, "wb") as f:
        pkl.dump(graphs, f)
    print("Processed Data Saved at {}".format(save_path))
