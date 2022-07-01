# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:58:34 2018

@author: lina_
"""
# class DOW():
#     LABEL_NAME = 'label'
#     W = None
    
# #    # Store words as strings with symbols separated by commas
# #    def __init__(self, W=None):
# #        W= str(W)
        
# =============================================================================
# Identifies whether a word is a double-occurrence word
# =============================================================================
def is_dow(word):
    # Set of seen letters
    seen = set()
    # Separates the word into letters, ignoring commas and spaces
    word_list = [x.strip('"') for x in word.split(',')]
    length = len(word_list)
    not_doubles = list()
    for i in range(length):
        if word_list[i] in seen:
            continue
        else:
            # Checks if each symbol appears twice
            d = word_list[i]
            seen.add(word_list[i])
            d_list = [x for x in word_list if x != d]
            if (len(word_list) - len(d_list) != 2) and word != '':
                not_doubles.append(d)
                
    if len(not_doubles) > 0:
        print('Not a DOW. The following symbols do not appear twice: ')
        print(not_doubles)
        return False
    else:            
        return True
# =============================================================================
# Subword finder (list and pattern are both lists)
# =============================================================================
def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append(pattern)#print('added', pattern, 'pat[0]=', pattern[0], 'mylist[i]=', mylist[i], 'mylist[i:i+len(pattern)]=', mylist[i:i+len(pattern)], 'i=', i )
    return matches
# =============================================================================
# Deletes sublist from list
# =============================================================================
def delete_sublist(sublist, biglist):
  #  if len(DOW.subfinder(biglist, sublist)) < 1:
        #print('Not a sublist')
    for i in range(len(sublist)):
        idx = biglist.index(sublist[i])
        del biglist[idx]
    return biglist
# =============================================================================
# Identifies whether a pattern is a repeat word
# =============================================================================
def is_repeat(pattern, word):
    pat_list = [x.strip() for x in pattern.split(',')]
    word_list = [x.strip() for x in word.split(',')]
    finds = subfinder(word_list,pat_list)
    if len(finds)==2:
        return True
    else:
        return False
    
# =============================================================================
# Identifies whether a pattern is a return word
# =============================================================================
def is_return(pattern, word):
    pat_list = [x.strip() for x in pattern.split(',')]
    word_list = [x.strip() for x in word.split(',')]
    finds = subfinder(word_list,pat_list)
    sub_list = delete_sublist(pat_list, word_list)
    rfinds = subfinder(sub_list,pat_list[::-1])
    # Reverses one word and checks for equality
    if len(finds)*len(rfinds)==1:
        return True
    else:
        return False

# =============================================================================
# Rewrites a word in ascending order
# =============================================================================
def ascending_order(word):
    if len(word) == 0:
        return ''
    
    # Set of characters that have already been seen in the word
    seen = set()
    # Set of ascending order letters (integers) used
    letters = list()
    # Dictionary of pairs to rewrite the word
    pairs = dict()
    # List of letters in the word
    word_list = [x.strip() for x in word.split(',')]
    length = len(word_list)
    
    # For each character in the word, assign the lowest available letter
    for i in range(length):
        # If the character already has an assignment, move to the next one
        if word_list[i] in seen:
            continue
        # Else, assign a letter
        else:
            if len(letters)==0:
                letters.append(1)
                new_letter = 1
            else:
                new_letter = letters[-1] +1
            seen.add(word_list[i])
            pairs[word_list[i]] = new_letter
            letters.append(new_letter)
    
    ret = ''
    # Rewrite the word in ascending order, separating letters with commas
    for i in range(len(word_list)):
        if len(ret) == 0:
            ret += str(pairs[word_list[i]])
        else:
            ret += ','+str(pairs[word_list[i]])
        
    return ret
  
# =============================================================================
# Creates a dictionary of pairs removed_word:(word,reduction)
# word = comma separated string
# to_reduce = set of repeat and return words in word
# =============================================================================
def reduce(word, to_reduce):
    word_list = [x.strip() for x in word.split(',')]
    ret = set()
    for w in to_reduce:
        w_list = w.split(',')
        red = [x for x in word_list if x not in w_list]
        redString = ascending_order(",".join(red))
        ret.add(redString)
    return ret
# =============================================================================
# Adds one layer of edges on the reduction graph
# =============================================================================
def add_layer(wordset, layernum):   
    ret = set()
    for word in wordset:
        if len(word)>0:
            to_red = find_patterns(word)
            red = reduce(word, to_red)
            for x in red:
                word_add =(word, layernum)
                x_add = (x, layernum+1)
                ret.add((word_add,x_add))
    return ret
# =============================================================================
# Computes edge pairs for the reduction graph
# =============================================================================
def edge_pairs(word):
    word = ascending_order(word)
    ret = set()
    layernum = 1
    wordset = {word}
    leaves = set()
    leaveslist = list()
    # Computes the words that the initial word can be reduced to, these are
    # all 1 edge away from the root. The layer is a set of pairs.
    layer = add_layer(wordset, layernum)
    ret.update(layer)
    layernum += 1
    for tup in layer:
        leaves.add(tup[1][0])
    leaveslist.append(leaves)
    # Check for convergence: the last layer contains only the empty word, 
    # if it's not the last layer added compute more layers
    while len(leaveslist[-1]) > 0 and leaveslist[-1] !={''}:
        new_lvs = set()
        new_lay = add_layer(leaves, layernum)
        for tup in new_lay:
            new_lvs.add(tup[1][0])
        leaveslist.append(new_lvs)
        ret.update(new_lay)
        layernum += 1
        leaves = new_lvs
    return ret
# =============================================================================
# Remove all initial loops from a word  
# =============================================================================
def remove_loops(word):
    word = ascending_order(word)
    word_list = [x.strip() for x in word.split(',')]
    length = len(word_list)
    loops = set() #set of loops
    i=0
    while i<length-1:
        if word_list[i]==word_list[i+1]:
            loops.add(word_list[i])
        i+=1
    # remove loops
    word_list = [x for x in word_list if x not in loops]
    word = ",".join(word_list)
    return word
# =============================================================================
# Returns a set of repeat or return words to reduce by
# =============================================================================
def find_patterns(word):
    word = ascending_order(word)
    
    # Turns the word into a list of letters
    word_list = [x.strip() for x in word.split(',')]
    length = len(word_list)
    
    to_reduce = set()
    
    # Sets of repeat and return words
    rep_patterns = set()
    ret_patterns = set()
    
    i=0
    while i < length-1:
        # If a letter is followed by its successor it could be a repeat word
        if int(word_list[i+1]) == int(word_list[i])+1:
            pattern = list()
            pattern.append(word_list[i])
            j = i+1
            pattern.append(word_list[j])
            while j < length-1 and int(word_list[j+1]) == int(word_list[j])+1:
                j += 1
                pattern.append(word_list[j])
            last1 = j
            pString = ",".join(pattern)
            # Check that it is a repeat word and add to list
            if is_repeat(pString,word):
                rep_patterns.add(pString)
                i = last1
        # If a letter is followed by its predecessor it may be a return word
        elif int(word_list[i+1]) == int(word_list[i])-1:
            pattern = list()
            pattern.append(word_list[i])
            j = i+1
            pattern.append(word_list[j])
            while j < length-1 and int(word_list[j+1]) == int(word_list[j])-1:
                j += 1
                pattern.append(word_list[j])
            last2 = j
            p = pattern[::-1] #+ pattern
            pString = ",".join(p)
            # Check that it is a return word and add to list
            if is_return(pString,word):
                ret_patterns.add(pString)
                i = last2 
        to_reduce.update(rep_patterns)
        to_reduce.update(ret_patterns)
        i += 1
    red_chars = list()
    for tr in to_reduce:
        wlist = [x.strip() for x in tr.split(',')]
        red_chars += wlist
    triv_set = set()
    for w in word_list:
        if w not in red_chars:
            nw = ",".join([w])
            triv_set.add(nw)
    # used letters won't be added as trivial repeat/return words
    used = set()
    for pat in to_reduce:
        used.update(set(pat))
    # deletes used letters
    triv_set.difference_update(used)
    to_reduce.update(triv_set)
    return to_reduce