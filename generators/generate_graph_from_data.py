from ..utils.Graph import DirectedGraph
import numpy as np
import pandas as pd
import torch as th
from torch.autograd import Variable
from..utils.loss import MomentMatchingLoss

class PolynomialModel(th.nn.Module):
    def __init__(self, rank, degree=2):
        super(PolynomialModel,self).__init__()
        self.l = th.nn.Linear(degree**(rank+1), 1, bias=False)

    def polynomial_features(self, x):
        return th.cat([(x[i].t() @ x[i]).view(1,-1) for i in range(x.size()[0])],0)

    def forward(self, x=None, n_examples=None):
        if not x:
            noise = th.FloatTensor(n_examples, 1).normal_()
        else:
            noise = th.FloatTensor(x.size()[0], 1).normal_()
            x = th.cat([x,noise],1)
        inputx = Variable(self.polynomial_features(x))
        return self.l(inputx)


def polynomial_regressor(df, causes, target, train_epochs=30):
    input = df[causes].as_matrix()
    model = PolynomialModel(len(causes),degree=2)
    criterion = MomentMatchingLoss(4)
    optimizer = th.optim.SGD(model.parameters())

    for epochs in train_epochs :



    return model(x)

class RandomGraphFromData(object):
    def __init__(self, df_data, datatype='Numerical'):
        super(RandomGraphFromData, self).__init__()
        self.data = df_data
        self.resimulated_data = None
        self.matrix_criterion = None
        self.llinks = None
        try:
            assert datatype == 'Numerical'
            self.criterion = np.corrcoef
            self.matrix_criterion = True
        except AssertionError:
            print('Not Yet Implemented')
            raise NotImplementedError

    def find_dependencies(self,threshold=0.05):
        if self.matrix_criterion:
            corr = np.absolute(self.criterion(self.data.as_matrix()))
            corr = np.fill_diagonal(corr, 0.)
        else:
            corr = np.zeros((len(self.data.columns),len(self.data.columns)))
            for idxi, i in enumerate(self.data.columns[:-1]):
                for idxj, j in enumerate(self.data.columns[idxi+1:]):
                    corr[idxi, idxj] = np.absolute(self.criterion(self.data[i], self.data[j]))
                    corr[idxj, idxi] = corr[idxi, idxj]

        self.llinks = [(self.data.columns[i], self.data.columns[j])
                  for i in range(len(self.data.columns)-1)
                  for j in range(i+1, len(self.data.columns))
                  if corr[i, j] > threshold]

    def generate_graph(self, draw_proba=.8):
        # Find dependencies
        if not self.llinks:
            self.find_dependencies()

        # Draw random number of edges out of the dependent edges and create an acyclic graph
        graph = DirectedGraph()

        for link in self.llinks:
            if np.random.uniform() < draw_proba:
                if np.random.uniform() < 0.5:
                    link = list(reversed(link))
                else:
                    link = list(link)

                graph.add(link[0], link[1], 1)

        graph.remove_cycles()

        # Resimulation of variables
        generated_variables = []
        nodes = graph.get_list_nodes()
        self.resimulated_data = pd.DataFrame()

        while len(generated_variables) < len(nodes):
            for var in nodes:
                par = graph.get_parents(var)
                if (var not in generated_variables and
                    set(par).issubset(generated_variables)):
                    # Variable can be generated
                    self.resimulated_data[var] = polynomial_regressor(self.data, par, var)



        # Regress using a y=P(Xc,E)= Sum_i,j^d(_alphaij*(X_1+..+X_c)^i*E^j) model
        # Resimulate

        return graph, self.resimulated_data