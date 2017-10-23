import networkx as nx
import numpy as np
import math

class Component_Graph():
    def __init__(self, n, M, graph_type):
        self.graphs = []
        self.graph_ids = np.ones((n), dtype='int32').tolist()
        if graph_type is 'nlogn':
            G = self.initial_nlogn_graph(n, M)
        elif graph_type is 'compelte':
            G = self.inital_complete_graph(n, M)
        else:
            raise ValueError("the graph_type can only be p2p or min_cut, but it is %s", graph_type)
        self.graphs.append(G)
        self.update_graph_ids()

    def inital_complete_graph(self, n, M):
        G=nx.complete_graph(n)
        for i in range(n):
            for j in range(i):
                G[i][j]['weight']=M[i, j]
        return G
    
    def initial_nlogn_graph(self, n, M):
        G=nx.Graph()
        G.add_nodes_from(np.arange(n))
        edges = self.nlogn_edges(n)
        G.add_edges_from(edges)
        for e in edges:
            G[e[0]][e[1]]['weight'] = M[e[0], e[1]]
        return G

    def nlogn_edges(self, n):
        edges_count=int(n*math.log(n))
        total=n*(n-1)/2
        up_diag_ids=np.random.choice(total, size=edges_count, replace=False)
        edges=[(self.cast_diag2xy(i, n)) for i in up_diag_ids]
        return edges

    def cast_diag2xy(self, up_diag_id, n):
        total=n*(n-1)/2
        if up_diag_id>=total:
            return
        row=0
        col=0
        for i in range(1,n):
            if i*(i+1)/2 > up_diag_id :
                col=i
                break
        row=up_diag_id - (col-1)*col/2

        return row, col

    def update_graph_ids(self):
        for i in range(len(self.graphs)):
            nodes = self.graphs[i].nodes()
            for j in nodes:
                self.graph_ids[j] = i

    def remove_p2p_edge(self, s, t):
        if not self.graph_ids[s]==self.graph_ids[t]:
            raise ValueError("Invalide Edge info, it does not belong to the single component!")
        graph_id = self.graph_ids[s]
        if self.graphs[graph_id].has_edge(s, t):
            self.graphs[graph_id].remove_edge(s,t)
        


    def remove_min_cut_edge(self, s, t, new_weight):
        if not self.graph_ids[s]==self.graph_ids[t]:
            raise ValueError("Invalide Edge info, it does not belong to the single component!")
        graph_id = self.graph_ids[s]
        if not self.graphs[graph_id].has_edge(s, t):
            return
        self.graphs[graph_id][s][t]['weight']=new_weight
        cut_weight, partitions = nx.minimum_cut(self.graphs[graph_id], s,  t, capacity='weight')
        edge_cut_list = []
        for p1_node in partitions[0]:
            for p2_node in partitions[1]:
                if self.graphs[graph_id].has_edge(p1_node, p2_node):
                    self.graphs[graph_id].remove_edge(p1_node, p2_node)


    def remove_edges(self, s, t, update_type, new_weight):

        if not self.graph_ids[s] == self.graph_ids[t]:
            return
        graph_id = self.graph_ids[s]
        if not s in self.graphs[graph_id].nodes():
            raise ValueError("start node is not in the graphs")
        if not t in self.graphs[graph_id].nodes():
            raise ValueError("end node is not in the graphs")

        if update_type is 'p2p':
            self.remove_p2p_edge( s, t)
            return
        if update_type is 'min_cut':
            self.remove_min_cut_edge(s, t, new_weight)
            return
        raise ValueError("the update_type can only be p2p or min_cut, but it is %s", update_type)

    def update_graph(self, user_id):
        graph_id = self.graph_ids[user_id]    
        components = list(nx.connected_component_subgraphs(self.graphs[graph_id]))
        if len(components) > 1:
            self.graphs.pop(graph_id)
            for c in components:
                self.graphs.append(c)
                self.update_graph_ids()

    def neighbor_list(self, node_id):
        graph_id = self.graph_ids[node_id]
        return self.graphs[graph_id].nodes()