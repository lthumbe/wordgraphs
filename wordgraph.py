# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:26:56 2019

@author: lina_
"""
import networkx as nx

from dow import *
from prodcells import PCELL

# =============================================================================
# Given a DOW (string separated by commas), it returns the word 
# reduction graph
# =============================================================================
def word_graph(original_word):
    if is_dow(original_word):
        word = ascending_order(original_word)
        if word != original_word:
            print(original_word + ' is equivalent to '+word+', which will be'
                  +' used for computations')
        graph = nx.DiGraph()
        edges = edge_pairs(word)
        if len(edges)>0:
            for edge in edges:
                (e1, e2) = edge
                (n1,l1) = e1
                (n2,l2) = e2
                graph.add_node(n1, layer=l1)
                graph.add_node(n2, layer=l2)
                graph.add_edge(n1, n2)
        else:
            label = word.replace(',','')
            graph.add_node(label)
        
        return graph
    else:
        print('This word is not a DOW.')

# =============================================================================
# Draw the word graph of a DOW
# =============================================================================
def draw_dow(dow, **kwargs):       
    l = int(len(dow.split(','))/2)
    fig, G = PCELL(word_graph(dow)).draw(filename=dow, wordlen = l, **kwargs)
    return fig, G
# =============================================================================
# Returns a set of partitions of n
# =============================================================================
def get_partition(n):
    ret = set()
    ret.add((n, ))
    for x in range(1, n):
        for y in get_partition(n - x):
            ret.add(tuple(sorted((x, ) + y)))
    return ret

# =============================================================================
# Return all DOWs of size n. n must be at least 1.
# =============================================================================
def getdows(n):
    if n < 1:
        print('n must be an integer greater than zero')
    else:    
        words = [[1,1]]
        wordlen = 2*n
        currlen = 2
        while currlen != wordlen:
            currlen += 2
            newwords = []
            for word in words:
                temp_word = [a + 1 for a in word]
                temp_word.insert(0,1)
                for i in range(1,currlen):
                    newword = list(temp_word)
                    newword.insert(i,1)
                    newwords.append(newword)
            words = newwords
        return words

# =============================================================================
# Computes the separation of a DOW w
# =============================================================================
def separation(w):
    sep = 0
    word_list = w.split(',')
    
    l = len(word_list)
    size = int(l/2)
    for i in range(1,size+1):
        locs = list_duplicates_of(word_list,str(i))
        d = max(locs) - min(locs) - 1
        sep += d
    return sep

# =============================================================================
# Given a DOW (string separated by commas), it returns the betti numbers of the
# associated word reduction graph
# =============================================================================
def betti_nums(word):
    G = word_graph(word)
    dmgG = PCELL(G)
    ret = list()
    i = 1
    while i < 3:
        bi = dmgG.betti_number(i)
        ret.append(bi)
        i+=1
    return ret

# =============================================================================
# Return the tangled cord with n symbols
# =============================================================================
def tangled(n):
    if n < 1:
        print('n must be an integer greater than zero')
        return
    else:    
        word_list = ['1']
        for i in range(1,n):
            word_list.append(str(i+1))
            word_list.append(str(i))
        word_list.append(str(n))
    word = ','.join(word_list)
    return word

# =============================================================================
# Lists duplicate indices of instances of item in the list seq
# =============================================================================
def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs
