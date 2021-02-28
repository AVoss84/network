# Next: 2 Models of network generation
# help to identify mechanisms that give rise to observed patterns in real data

# Preferential Attachment model
# produces Power-law degree distributions P(k) = C*k^{-alpha} , C and alpha are constants
# linear log-log scale tails
# maynlow valued degree nodes and some high valued

G  = nx.barabasi_albert_graph(100000, 1)
print(nx.average_clustering(G))
print(nx.average_shortest_path_length(G))

# Number of neighbors per node
degrees = G.degree()         # for undirected graphs    
#in_degrees = G.in_degree()   # for directed graphs
degree_values = sorted(set(degrees.values()))    # get degrees of all nodes in the network
# Relative Frequencies per degree value in the graph
histogram = [list(degrees.values()).count(i)/float(nx.number_of_nodes(G)) for i in degree_values]

import matplotlib.pyplot as plt

plt.plot(degree_values, histogram, 'o')
#plt.bar(degree_values, histogram, 'o')
plt.xlabel('Degree')
plt.ylabel('Fraction of nodes')
plt.xscale('log')
plt.yscale('log')
plt.show()

# Small World Network models:
# Start with ring of n nodes (each node connected to its k nearest neighbors)
# Small average shortest path: high degree nodes 
# act as hubs and connect many pairs of nodes
# High (avg.) local clustering coefficient (-> tendency to create triades/triangles, i.e. my friends become friends too)

# Create small world network
# and calculate its degree distribution

G = nx.watts_strogatz_graph(n=1000, k=6, p=0.04)   # p rewiring probab., k (nearest neighbors), can be disconnected which is sometimes undesirable
# Alternative:
G = nx.connected_watts_strogatz_graph(n=1000, k=6, p=0.04, t=100)  # repeats watts_strogatz_graph t times until it returns a connected network

degrees = G.degree()         # for undirected graphs    
degree_values = sorted(set(degrees.values()))    # get degrees of all nodes in the network
# Relative Frequencies per degree value in the graph
histogram = [list(degrees.values()).count(i)/float(nx.number_of_nodes(G)) for i in degree_values]

import matplotlib.pyplot as plt

plt.bar(degree_values, histogram, 'o')
plt.xlabel('Degree')
plt.ylabel('Fraction of nodes')
plt.xscale('log')
plt.yscale('log')
plt.show()

# Also check out for GNNs: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html 
