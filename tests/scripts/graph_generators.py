"""Testing generators."""

import networkx as nx
from cdt.generators import AcyclicGraphGenerator, CyclicGraphGenerator
from cdt.generators.causal_mechanisms import normal_noise, uniform_noise, gmm_cause, gaussian_cause
mechanisms = ['linear', 'polynomial', 'sigmoid_add',
              'sigmoid_mix', 'gp_add', 'gp_mix', 'NN']
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import pandas as pd


if __name__ == "__main__":


    for nb_points in [200,500,1000]:

        for nb_nodes in [20,50,100]:

            for mecanism in mechanisms:
                print(nb_points)
                print(nb_nodes)
                print(mecanism)

                if(mecanism == "linear"):
                    cause = gaussian_cause
                else:
                    cause = gmm_cause

                agg, data = AcyclicGraphGenerator(mecanism, points=nb_points, nodes=nb_nodes, parents_max=5, initial_variable_generator=cause).generate()

                Adj = nx.to_numpy_matrix(agg)



                cpt = 0
                for i in range(Adj.shape[0]):

                    for j in range(Adj.shape[1]):

                        if(Adj[i,j]==1):
                            cpt+=1
                            cause = data["V" + str(i)].values
                            effect = data["V" + str(j)].values

                            # if(20<cpt and cpt < 23):
                            #     plt.scatter(cause, effect)
                            #     plt.show()

                np.savetxt( "train_graph_" + mecanism + "_d_" + str(nb_nodes) + "_N_" + str(nb_points) + "_target.csv",Adj, delimiter=",")
                data.to_csv("train_graph_" + mecanism + "_d_" + str(nb_nodes) + "_N_" + str(nb_points) + "_data.csv", index=None)

