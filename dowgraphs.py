# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:57:11 2022

@author: lina_
"""

import streamlit as st
from dow import *
from wordgraph import *
from prodcells import *

import matplotlib.pyplot as plt
import networkx as nx

st.title('Word Graphs')
st.header('Plotting and computing some of their properties.')

st.markdown('Let $\Sigma$ be an alphabet. _Double Occurrence Words_ (DOWs) ' +
            'are words where each symbol appears exactly twice. For example, ' +
            '$14323124$ is a DOW on the alphabet of positive integers. ' +
            'We say a DOW with symbols in $\mathbb{N}^{>0}$ is in ' + 
            '_ascending order_ if its first symbol is $1$ and each new ' + 
            'symbol is exactly one more than the maximum among all previously ' +
            'used symbols. $121323$ is in ascending order, while $343545$ ' + 
            'is not. Two DOWs are equivalent if they have the same ascending ' +
            'order representation.')

st.markdown('Let $u^R$ denote the reverse of $u$, where all symbols are ' + 
            'written in reverse order. We say $uu^R$ is a _return_ word in ' + 
            'a DOW $w$ if $w = xuyu^Rz$ for some $x,y,z \in \Sigma^*$. ' + 
            'We say that $uu$ is a _repeat_ word if $w = xuyuz$ for some ' +
            '$x,y,z \in \Sigma^*$. A repeat (resp. return) word is _maximal_ ' + 
            'if there does not exist a longer repeat (resp. return) word ' + 
            '$vv$ (resp. $vv^R$) such that $u \sqsubseteq v$. If either ' +
            '$uu$ is a maximal repeat word in $w$ or $uu^R$ is a maximal ' +
            'return word in $w$ we say that $u$ is a _maximal repeat factor_.')

st.markdown('Let $u$ be a maximal repeat (resp. return) factor in $w = xuyuz$ ' +
            '(resp. $w=xuyu^Rz$). The deletion of $u$ from $w$ yields ' +
            '$d_u(w)= xyz$. We call $v$ an _immediate successor_ of $w$ if' + 
            'there exists a maximal repeat or return factor $u$ such that ' + 
            '$v$ is ascending order equivalent to $d_u(w)$. A _succcessor_ ' + 
            'of $w$ is any word that can be obtained through iterated deletions, '+
            'in ascending order. The successor relationship defines a partial ' +
            'order DOWs.')

st.markdown('The _global word graph_ of size $n$, $G_n$ has all DOWs with up to ' + 
            '$n$ symbols as vertices and edges of the form')
st.latex( r'''w_1 \to w_2''')
st.markdown('where $w_2$ is an immediate successor of $w_1$. The _word_ ' + 
            '_graph rooted at_ $w$, for a DOW $w$, is the subgraph of a global word ' + 
            'graph containing $w$ that has $w$ as its maximum element and the ' +
            'empty word, $\epsilon$, as its minimum.')

st.markdown('An $n$-dimensional _simplicial digraph_ (also directed clique ' +
            'or transitive tournament) is the directed graph with vertices ' + 
            ' $V = \{v_0, v_1, v_2, \ldots, v_n\}$ where')
st.latex(r'''v_i \to v_j''')
st.markdown('whenever $i < j$. Let $G\square H$ denote the Cartesian product of ' +
            'two directed graphs, $G$ and $H$. A _prodsimplicial cell_ is ' + 
            'one whose 1-skeleton is the Cartesian product of finitely many ' +
            'nontrivial ($n>0$) simplicial digraphs. ')

st.markdown('The scripts in this page draw rooted word graphs and compute ' + 
            'several of their properties, including Betti numbers built on '+
            'complexes of prodsimplicial cells.')

dow = st.text_input("Double Occurrence Word", max_chars = 40, help="Type a " + 
                    "DOW with positive integer symbols separated by commas.", 
                    value="1,2,1,3,2,3,4,4")

draw_button = st.button("Draw word graph")

ncolor = st.sidebar.color_picker('Pick a color for vertices', '#1C82BA')
langle = st.sidebar.slider('Angle of rotation for vertex labels', min_value=0, max_value=360, value = 10)
nsize = st.sidebar.number_input('Node size', min_value=0,value=60)

vlayer_md = 'Graphs use a multipartite graph layout with layers indicating ' + 'the deletion steps. The original DOW is placed on layer 1 ' +'by default and each deletion increases the counter by one.'
st.sidebar.text('Change vertex layout.')


vchoice = None

wordgraph = PCELL(word_graph(dow))
betti1 = wordgraph.betti_number(1)
betti2 = wordgraph.betti_number(2)
gvertices = len(wordgraph.G.nodes())
gedges = len(wordgraph.G.edges())

vertices = wordgraph.G.nodes(data=True)
vlabels = ()
layers = [vertices[v[0]]['layer'] for v in vertices]
for v in vertices:
    vlabels+=(v[0],)

def redraw_dow():
    wordgraph.change_layer(vchoice, nlayer)
    with col1:
        fig, G = wordgraph.draw(node_color=ncolor,node_size=nsize, angle=langle,
                                layer_by=layer_choice) 
        st.pyplot(fig, dpi=300)
        filename = dow + ".png" 
        
        fig.savefig(filename, transparent=True, dpi=300, bbox_inches='tight',pad_inches=0)
        with open(filename, "rb") as file:
             btn = st.download_button(
                     label="Download image",
                     data=file,
                     file_name=filename,
                     mime="image/png"
                    )
 
def draw_dow():
    with col1:
        fig, G = wordgraph.draw(node_color=ncolor,node_size=nsize, angle=langle,
                                layer_by=layer_choice) 
        st.pyplot(fig, dpi=300)
        filename = dow + ".png" 
        
        fig.savefig(filename, transparent=True, dpi=300, bbox_inches='tight',pad_inches=0)
        with open(filename, "rb") as file:
             btn = st.download_button(
                     label="Download image",
                     data=file,
                     file_name=filename,
                     mime="image/png"
                    )
    
    # with col2:
    #     st.text('Properties')
    #     if draw_button:
    #         st.latex(r'''\beta_1: ''' +str(betti1))
    #         st.latex(r'''\beta_2: ''' + str(betti2))
    #         st.latex(r'''\text{vertices: }''' + str(gvertices))
    #         st.latex(r'''\text{edges: }''' + str(gedges))
            
    
vchoice = st.sidebar.selectbox('Choose a vertex to move up/down.', 
                               vlabels)
nlayer = st.sidebar.number_input('New layer', min_value = min(layers)-1,
                        max_value = max(layers)+1,value = vertices[vchoice]['layer'])

layer_choice = st.sidebar.selectbox('Layer vertices by...', ('deletions','length'), help=vlayer_md)
update_button = st.sidebar.button('Update')


col1, col2 = st.columns([3,1])

with col1:
    if draw_button:
        wordgraph = PCELL(word_graph(dow))        
        fig, G = wordgraph.draw(node_color=ncolor,node_size=nsize, angle=langle,
                                layer_by=layer_choice)        

        st.pyplot(fig, dpi=300)
        filename = dow + ".png" 
        
        fig.savefig(filename, transparent=True, dpi=300, bbox_inches='tight',pad_inches=0)
        with open(filename, "rb") as file:
             btn = st.download_button(
                     label="Download image",
                     data=file,
                     file_name=filename,
                     mime="image/png"
                    )

with col2:
    st.markdown('Properties')

    st.latex(r'''\beta_1: ''' +str(betti1))
    st.latex(r'''\beta_2: ''' + str(betti2))
    st.latex(r'''\text{vertices: }''' + str(gvertices))
    st.latex(r'''\text{edges: }''' + str(gedges))
    
    st.markdown('Compute another property')
    prop_choice = st.selectbox('Choose a property',('nth Betti number', 'number of n cells', 'separation'))
    nvalue = st.number_input('Value of n', min_value=0, max_value=5)
    res = 'The '+ prop_choice + ' is '
    if prop_choice == 'nth Betti number':
        res += str(wordgraph.betti_number(nvalue))
    elif prop_choice == 'number of n cells':
        res += str(len(wordgraph.n_cells(nvalue)))
    elif prop_choice == 'separation':
        res += str(separation(dow))
    st.markdown(res)
    
    

if update_button:
    redraw_dow()