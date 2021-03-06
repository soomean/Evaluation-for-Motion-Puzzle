import numpy as np


class Graph():
    def __init__(self,
                 layout='styletransfer',   # graph_args
                 max_hop=1,
                 dilation=1,
                 strategy='spatial'):
        self.max_hop = max_hop  # 1
        self.dilation = dilation  # 1

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)  # NOTE: 0(self node)에서 max_hop(=1;direct neighbor)까지 거리 관계를 나타낸 행렬, 그 외 모두 inf

        self.A1 = self.get_adjacency('uniform')
        self.A2 = self.get_adjacency('distance')
        self.A3 = self.get_adjacency('spatial')

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1

        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3),
                              (5, 21), (6, 5), (7, 6), (8, 7),
                              (9, 21), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15),
                              (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1

        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2

        elif layout == 'styletransfer':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 0), (2, 1), (3, 2), (4, 3),
                              (5, 0), (6, 5), (7, 6), (8, 7),
                              (9, 0), (10, 9), (11, 10), (12, 11),
                              (13, 10), (14, 13), (15, 14), (16, 15),
                              (17, 10), (18, 17), (19, 18), (20, 19)]
            self.edge = self_link + neighbor_link
            self.center = 0

        elif layout == 'styletransfer_down1':
            self.num_node = 10
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 0), (0, 4), (3, 2), (2, 4),
                             (5, 4), (6, 5), (7, 6), (8, 5), (9, 8)]
            self.edge = self_link + neighbor_link
            self.center = 4

        elif layout == 'styletransfer_down2':
            self.num_node = 5
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 2), (1, 2), (3, 2), (4, 2)]
            self.edge = self_link + neighbor_link
            self.center = 2

        elif layout == 'styletransfer_down3':
            self.num_node = 2
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1)]
            self.edge = self_link + neighbor_link
            self.center = 1

        else:
            raise ValueError("This layout does not exist.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)  # NOTE: 0~1
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:  # NOTE: 0~1
            adjacency[self.hop_dis == hop] = 1  # NOTE: represents I when hop = 0, A otherwise
        normalize_adjacency = normalize_digraph(adjacency)  # NOTE: normalize each adjacency matrix

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency

        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==hop]

        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] < self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)

                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
                    
            A = np.stack(A)

        else:
            raise ValueError("This strategy does not exist.")

        return A


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))  # Adjacency matrix
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d

    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)

    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    
    return DAD
    