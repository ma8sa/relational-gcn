import numpy as np
import pickle as pkl
import scipy.sparse as sp






no_of_graphs = int(input(" no. pf graphs "))
start = int(input(" starting point of graphs "))
#total_type_edges = int(input(" type of edges \n"))
total_type_edges = 4
data_folder = "./data/custom/"


for graph_num in range(start,no_of_graphs):
    print("--------------------------------------")
    num_nodes = int(input(" no. of nodes \n"))
    A = [np.zeros((num_nodes,num_nodes)) for _ in range(total_type_edges)]
    X = np.zeros(num_nodes)
    labels = np.zeros(num_nodes)

    num_of_edges = int(input(" no.  of edges \n"))
    print(" enter the edges : 0 moving across , 1 moving ahead , 2 no movement , 3 diagonall")

    for edges in range(num_of_edges):
        print("Edge # {}of{}".format(edges,num_of_edges))
        x = int(input("edge details : X : "))
        y = int(input("edge details : Y : "))
        e = int(input("edge details : E : "))
        A[e][x,y] = 1
        A[e][y,x] = 1

    #node_type = int(input(" no.  of edges \n"))
    print(" node type : 0 car,1 pole , 2 marker")
    print(" Labels : 0 moving , 1 static " )
    for e in range(num_nodes):
        X[e] = int(input(" node {} type :".format(e)))
        labels[e] = int(input(" labels for node {} : ".format(e)))
    
    ## dump stuff here 
    X = sp.csr_matrix(X)
    A = [ sp.csr_matrix(a) for a in A ]

    pkl.dump(labels,open(data_folder + "labels" + str(graph_num).zfill(4) + ".pkl","wb"))
    pkl.dump(A,open(data_folder + "A" + str(graph_num).zfill(4) + ".pkl","wb"))
    pkl.dump(X,open(data_folder + "X" + str(graph_num).zfill(4) + ".pkl","wb"))
