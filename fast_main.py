import clustering_fast as clf
import numpy as np

users = 5
graph_type = 'nlogn'
M = np.ones((users, users))
cg = clf.Component_Graph(users, M, graph_type)

print cg.graphs[0].edges()
print cg.graph_ids
