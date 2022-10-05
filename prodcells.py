# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:17:23 2018

@author: fajardogomez
"""
# from sage.all import *
from sympy.combinatorics import Permutation
import networkx as nx
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg.interpolative import estimate_rank

class PCELL():
    LABEL_NAME = 'label'
    G = None
# =============================================================================
# Initializes the prodsimplicial cell
# =============================================================================
    def __init__(self, G=None):
        self.set_graph(G)

# =============================================================================
# Initializes the underlying DiGraph. Note that cell.G can access nx methods.
# =============================================================================
    def set_graph(self, G):
        self.G = nx.DiGraph(G)
        
# =============================================================================
# Plots the graph, optional input with_labels adds labels to edges    
# =============================================================================
    def draw(self, with_labels=False, **kwargs):
        fig, G = self.draw_directed_graph(self.G, with_labels, **kwargs)
        return fig, G

# =============================================================================
# Class method function to plot graphs        
# =============================================================================
    @classmethod
    def draw_directed_graph(cls, G, with_labels=False, **kwargs):
        # kwargs.setdefault('layer_by','layer')
        # kwargs.setdefault('node_color','#0081D7' )
        # kwargs.setdefault('angle', 10)
        # kwargs.setdefault('node_size',60)
    
        nodes = dict(G.nodes(data=True))
        if len(nodes) == 0:
            layer_by = 'length'
        if 'layer_by' in kwargs:
            layer_by = kwargs['layer_by']
        if 'filename' in kwargs:
            filename = kwargs['filename']
        if 'wordlen' in kwargs:
            wordlen = kwargs['wordlen']
        else:
            wordlen = int(max([len(x.split(','))/2 for x in nodes]))
        if 'node_color' in kwargs:
            color = kwargs['node_color']
        if 'angle' in kwargs:
            angle = kwargs['angle']
        if 'node_size' in kwargs:
            nsize = kwargs['node_size']    
        

        if layer_by == 'length':
            root_len = int(max([len(n.split(','))/2 for n in nodes]))
            for n in nodes:
                nodes[n]['length'] = root_len - int(len(n.split(','))/2)
        if layer_by == 'deletions':
            layer_by = 'layer'
        done = False
        try:
            layer_pairs = list()
            layernums = [nodes[x][layer_by] for x in nodes]
            i=min(layernums)
            while not done:
                counter=0
                for n in nodes:
                    if int(nodes[n][layer_by]) == i:
                        counter +=1
                if counter != 0:
                    layer_pairs.append((i,counter))
                i+=1
                if counter == 0:
                    done = True
            num_layers = max([x for (x,y) in layer_pairs])
            max_layer = max([y for (x,y) in layer_pairs])
            nodes = sorted(nodes.items(), key=lambda x:x[1][layer_by])
            # Workaround for compatibility issue in networkx 2.7
            # Add vertices in the order that the layers are in
            H = nx.DiGraph()
            H.add_nodes_from(nodes)
            H.add_edges_from(G.edges())
            # Estimate size of figure with length of word and number of layers
            plt.figure(figsize=(2*max_layer + 0.35*wordlen, 1.5*num_layers))
            pos = nx.multipartite_layout(H, subset_key=layer_by, 
                                         align = 'horizontal', scale=2)
            
        except KeyError:
            pos = nx.spring_layout(G)
        
        # Shift vertex labels up
        pos_higher = {}
        y_off = -0.1  # offset on the y axis
        if num_layers < 5:
            y_off -= 0.05
        for k, v in pos.items():
            pos_higher[k] = (v[0], v[1]+y_off)
        
        nx.draw_networkx_nodes(H, pos, node_color=color, alpha =0.7, node_size=nsize)
        nx.draw_networkx_edges(H, pos_higher, arrowstyle='->', arrowsize=10, 
                               node_size=900, width=2,edge_color="black", alpha = 0.7)
    
        node_labels=dict()
        
        # Labels the vertices
        for node in H.nodes():
            node_labels[node]=str(node)
        text = nx.draw_networkx_labels(H, pos_higher, node_labels, font_size=10)
        
        max_node_len = max([len(str(x)) for x in H.nodes()])
        # Rotate node labels so they don't overlap
        for _,t in text.items():
            t.set_rotation(angle)
        # Labels the edges
        if with_labels:
            edge_labels = nx.get_edge_attributes(H, cls.LABEL_NAME)
            nx.draw_networkx_edge_labels(H, pos, labels = edge_labels)
    
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_axis_off()

        plt.show()
        fig = plt.gcf()
        
        return fig, H

# =============================================================================
# Change the layer a node is in and returns G
# =============================================================================
    def change_layer(self, node, new_layer):
        nodes = dict(self.G.nodes(data=True))
        if node not in nodes:
            print(str(node) + ' is not a node in the graph.')
        else:
            nodes[node]['layer'] = new_layer
        
# =============================================================================
# Creates a simplex of dimension n
# =============================================================================
    @classmethod
    def generate_simplex(cls, n):
        ret = nx.DiGraph()
        
        if n <= 0:
            return ret
        
        for i in range(0, n + 1):
            for j in range(0, i):
                ret.add_edge(str(j), str(i))  # order-orientation
                
        return ret
 
# =============================================================================
# Given a partiton set p, it creates the corresponding prodisimplicial cell W    
# =============================================================================
    @classmethod
    def generate_w(cls, p):
        simplices = list()
        W = None
            
        # Creates simplices of the needed dimensions 
        for i in p:
            simplex = cls.generate_simplex(i)
            simplices.append(simplex)
            
        # Computes the Cartesian product W, a graph to search for in G
        for j in simplices:
            if W is None:
                W = j
                continue
            W = nx.cartesian_product(W, j)
            
        return W

# =============================================================================
# Returns a list of n-dimensional prodsimplicial cells   
# =============================================================================
    def n_cells(self, n):
        ret = list()
        tmp = list()
        used_vertices = dict()
        
        # igraph version of G
        iG = ig.Graph.from_networkx(self.G)
        if n == 0:
            nodes = list(self.G.nodes())
            for node in nodes:
                g = nx.DiGraph()
                g.add_node(node)
                tmp.append({
                    'graph':g, 
                    'isom':{'0':str(node)}, 
                    'part':(0,), 
                    'orientation':1,
                    'vertices': '_'.join(sorted([str(node)]))})
            ret = sorted(tmp, key=lambda d: d['vertices'])
            return ret
        else:
            # Creates a partition to compute all products of dimension n
            for p in partition(n):
                if p not in used_vertices:
                    used_vertices[p] = list()
                W = self.generate_w(p)
                # igraph version of W
                iW = ig.Graph.from_networkx(W)
                isomorphisms = iG.get_subisomorphisms_lad(iW, induced=True)
                
                # Check for duplicate graphs with different isomorphisms
                for ism in isomorphisms:
                    subg_nodes = [iG.vs[x]['_nx_name'] for x in ism]
                    W_nodes = [iW.vs[x]['_nx_name'] for x in list(range(len(ism)))]
                    isom_map = {W_nodes[i]: subg_nodes[i] for i in range(len(subg_nodes))}
                    S = self.G.subgraph(subg_nodes)
                    
                    # Standard source node
                    st_sc_node = min(list(isom_map.keys()))
                    # Neighbors of source in standard cell, in descending order.
                    
                    st_sc_nbrs = sorted(list(W.neighbors(st_sc_node)))
                    st_sc_nbrs.reverse()
                    
                    # Cell source node
                    sc_node = isom_map[st_sc_node]
                    # If the standard cell translates to an even permutation of 
                    # this, the cell is poitively oriented
                    sc_nbrs = sorted(list(S.neighbors(sc_node)))
                    
                    # Translated neighbors of the source in the standard cell W
                    tr_sc_nbrs = [isom_map[nbr] for nbr in st_sc_nbrs]
                    
                    # The permutation that maps sc_nbrs to tr_sc_nbrs
                    tr_perm = Permutation([sc_nbrs.index(x) for x in tr_sc_nbrs])
                    
                    if set(subg_nodes)not in used_vertices[p]:
                        used_vertices[p].append(set(subg_nodes))
                        # if tr_perm.is_even():
                        if tr_perm.parity() == 0:
                            orientation = 1
                        else:
                            orientation = -1
                        tmp.append(dict({"graph": S,
                                          "isom": isom_map,
                                          "part": p,
                                          "orientation": orientation,
                                          "vertices" : '_'.join(sorted(subg_nodes))
                                          }))
                    
            ret = sorted(tmp, key=lambda d: d['vertices'])                  
            return ret
# =============================================================================
# Creates the dictionary of bdry_el:a of a standard cell generated by p
# =============================================================================
    @classmethod
    def boundary_els(cls, p):
        # Initializes the partition as a list
        plist = list(p)
        ret = list()
        
        for m in range(0,len(plist)):
            n = plist[m]
            # Sn is a factor of W
            Sn = cls.generate_simplex(n)
            # Removes one node at a time from Sn and computes the induced
            # subgraph
            for i in range(0,n+1):
                nbunch = list(Sn.nodes())
                if len(nbunch) > 0:
                    nbunch.remove(str(i))
                    bd = Sn.subgraph(nbunch)
                    Wbd = nx.DiGraph()
                    # Sets the "prefix" factor
                    if m==0:
                        Wbd=bd
                    elif m>0:
                        Wpre = nx.DiGraph()
                        part = plist[:m]    
                        Wpre = cls.generate_w(part)
                        Wbd = nx.cartesian_product(Wpre, bd)
                        
                    a = sum(plist[:m]) + i
                    
                    # Adds the "suffix" factors 
                    part2 = plist[m+1:len(p)]
                    
                    for j in part2:
                        Wbd = nx.cartesian_product(Wbd, cls.generate_simplex(j))
                    
                    ret.append(dict({"bdry_el":Wbd, "multi": a}))
        return ret
           
# =============================================================================
# Returns the dictionary of facets:a of a cell defined by the partition p
# =============================================================================
    def facets(self, p, isom):
        bd_els = self.boundary_els(p)
        ret = list()
        
        # for each face in the standard boundary set, map to the graph
        for elem in bd_els:
            new_isom = dict()
            nodes = elem["bdry_el"].nodes()
            # face_nodes = set()
            for node in nodes:
                # face_nodes.add(isom[node])
                new_isom[node] = isom[node]
            a = elem["multi"]
            ret.append(dict({"multi": a, "isom": new_isom, "bdry_el": elem["bdry_el"]})) #"face":rface, 
        return ret

# =============================================================================
# Calculates the boundary of a cell
# =============================================================================
    def boundary_op(self, n):
        # Source cells are the domain - n dimensional cells
        source_cells = list()
        Ncells = list(self.n_cells(n))
        
        # Target cells are the range - (n-1) dimensional cells
        target_cells = list()
        
        # Keep track of used node sets and partitions
        used_nodes = list()
        used_partitions = list()
        source_cell_or = dict()
        target_cell_or = dict()
        
        if n == 0:
            d = len(Ncells)
            S = dok_matrix((1,d), dtype=np.float64)
        else:
            nm1cells = list(self.n_cells(n-1))
            for i in range(len(nm1cells)):
                elem = nm1cells[i]
                p = elem["part"]
                vset = set(elem["isom"].values())
                if p not in used_partitions:
                    used_partitions.append(p)
                if vset not in used_nodes:
                    used_nodes.append(vset)
                target_cell_or[(used_partitions.index(p), used_nodes.index(vset))] = elem["orientation"]
                # Each target cell is uniquely identified by a partition and node set
                target_cells.append((used_partitions.index(p), used_nodes.index(vset)))
            for i in range(len(Ncells)):
                elem = Ncells[i]
                p = elem["part"]
                vset = set(elem["isom"].values())
                if p not in used_partitions:
                    used_partitions.append(p)
                if vset not in used_nodes:
                    used_nodes.append(vset)
                source_cell_or[(used_partitions.index(p), used_nodes.index(vset))] = elem["orientation"]
                # Each source cell is uniquely identified by a partition and node set
                source_cells.append((used_partitions.index(p), used_nodes.index(vset)))
            target_cells.sort(key=lambda x:x[1])
            source_cells.sort(key=lambda x:x[1])
            # if range is empty (domain is vertices), make a row of of ones of length source_cells
            if len(target_cells)==0:
                d = len(source_cells)########
                S = dok_matrix((1, d), dtype=np.float64)
                S[0, 0:d] = 1
            else:
                source_cells_dict = {
                    source_cells[j]: j for j in range(len(source_cells))}
                target_cells_dict = {
                    target_cells[i]: i for i in range(len(target_cells))}
                d = len(source_cells)
                f = len(target_cells)
                S = dok_matrix((f, d), dtype=np.float64)
                # For each cell, compute the standard boundary and translate it
                for k in range(len(source_cells)):
                    s_cell = Ncells[k]
                    vset = set(s_cell["isom"].values())
                    part = s_cell["part"]
                    s_pair = (used_partitions.index(part), used_nodes.index(vset))
                    
                    # Boundary elements of the given source cell
                    t_cell_aux = PCELL(s_cell["graph"]).facets(part, s_cell["isom"])
                    # For each face
                    for elem in t_cell_aux:
                        # Orientation of the face in the standard cell
                        # induced by the boundary operator
                        # a = sum of dimensions up to bdry factor
                        a = elem["multi"] 
                        
                        subg_nodes = list(elem["isom"].values())
                        subG = self.G.subgraph(subg_nodes)
                        
                        # Standard cell
                        W = elem["bdry_el"]
                        isom_map = elem["isom"]
                        
                        st_sc_node = min(list(isom_map.keys()))
                        st_sc_nbrs = sorted(list(W.neighbors(st_sc_node)))
                        # Standard cell vertices are ordered in decreasing
                        # lexicographic order
                        st_sc_nbrs.reverse()
                        
                        sc_node = isom_map[st_sc_node]
                        sc_nbrs = sorted(list(subG.neighbors(sc_node)))
                        # Actual cell vertices are ordered in increasing
                        # lexicographic order
                        tr_sc_nbrs = [isom_map[nbr] for nbr in st_sc_nbrs]
                        
                        # Permutation between st_sc_nbrs and tr_sc_nbrs
                        tr_perm = Permutation([sc_nbrs.index(x) for x in tr_sc_nbrs])
                        
                        bd_vset = set(elem["isom"].values())
                        
                        # Orientation of the target cell as a standard face
                        inherit_or = 1 if tr_perm.parity() == 0 else -1

                        for t_pair in target_cells:
                        # Find the target cell corresponding to the boundary element
                            if bd_vset == used_nodes[t_pair[1]]:
                                entry = -1 if a % 2==1 else 1 
                                entry *= inherit_or
                                entry *= source_cell_or[s_pair]
                                i = target_cells_dict[t_pair]
                                j = source_cells_dict[s_pair]
                                S[i, j] = entry# S[i, j] = (-1)^a
                
        return S
    
# =============================================================================
# Calculates the Betti numbers 
# =============================================================================
    def betti_number(self, i, eps=None):
        boundop_ip1 = self.boundary_op(i+1)
        boundop_i = self.boundary_op(i)
        if i == 0:
            boundop_i_rank = 0
        else:
            if eps is None:
                try:
                    boundop_i_rank = np.linalg.matrix_rank(boundop_i.toarray())
                except (np.linalg.LinAlgError, ValueError):
                    boundop_i_rank = boundop_i.shape[1]
            else:
                boundop_i_rank = estimate_rank(aslinearoperator(boundop_i), eps)

        if eps is None:
            try:
                boundop_ip1_rank = np.linalg.matrix_rank(boundop_ip1.toarray())
            except (np.linalg.LinAlgError, ValueError):
                boundop_ip1_rank = boundop_ip1.shape[1]
        else:
            boundop_ip1_rank = estimate_rank(aslinearoperator(boundop_ip1), eps)
        betti = (boundop_i.shape[1] - boundop_i_rank) - boundop_ip1_rank
        
        return betti

# =============================================================================
# Returns a set of partitions of n
# =============================================================================
def partition(n):
    ret = set()
    ret.add((n, ))
    for x in range(1, n):
        for y in partition(n - x):
            ret.add(tuple(sorted((x, ) + y)))
    return ret
