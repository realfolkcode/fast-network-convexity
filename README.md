## Fast Approximate Convex Hull Construction in Networks via Node Embedding

## Abstract

Geodesic convexity in networks is an intrinsic property of graphs. It aids in distinguishing between real-world networks and random graphs. One possible application is recommending new connections in a collaborative network by searching for them in the so-called convex hull, which is a minimal subgraph
containing all the shortest paths between its nodes. However, the existing algorithms for constructing convex hulls from subsets of nodes involve extensive search over subgraphs and have poor scalability. Thus, they become inapplicable to large graphs such as social networks.

In this paper, we propose a new approach for fast convex hull construction for a subset of nodes on a
network using graph embeddings. We apply the well-known convexity concept in embedding space to a
similar problem for geometric learning on a graph, optimizing the process of finding all the shortest paths in
the induced subgraph. To preserve the metric characteristics of a network, we train a graph neural network
with an L1-distance loss. As a result, the trained model enables us to approximately verify the convexity of
subgraphs in linear time, contrary to the previous approaches, which have cubic complexity.

## Authors

* Dmitrii Gavrilev
* Ilya Makarov

See `fast-network-convexity-dev.ipynb` to run experiments
