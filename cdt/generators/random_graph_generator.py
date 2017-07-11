from .functions_default import (noise, cause, effect, rand_bin)
from ..utils.Graph import DirectedGraph
from random import shuffle
from sklearn.preprocessing import scale
import numpy.random as rd
import pandas as pd
import numpy as np
import operator as op


def series_to_cepc_kag(A, B, idxpair):
    strA = ''
    strB = ''
    for i in A.values:
        strA += ' '
        strA += str(i)

    for i in B.values:
        strB += ' '
        strB += str(i)

    return pd.DataFrame([['pair' + str(idxpair), strA, strB]], columns=['SampleID', 'A', 'B'])


class RandomGraphGenerator:
    def __init__(self,
                 num_nodes=200,
                 max_joint_causes=4,
                 noise_qty=.7,
                 number_points=500,
                 categorical_rate=.20):

        self.nodes = num_nodes
        self.noise = noise_qty
        self.n_points = number_points
        self.cat_rate = categorical_rate
        self.num_max_parents = max_joint_causes
        self.joint_functions = [op.add, op.mul]
        self.causes = None
        self.graph = None
        self.data = None
        self.result_links = None
        self.cat_data = None
        self.cat_var = None

        print('Init OK')

    def generate(self, gen_cat=True):
        print('--Beginning Fast build--')
        # Drawing causes
        self.causes = [i for i in range(np.random.randint(
            2, self.nodes / np.floor(np.sqrt(self.nodes))))]
        self.causes = list(set(self.causes))
        self.data = pd.DataFrame(None)
        layer = [[]]
        for i in self.causes:
            self.data['V' + str(i)] = cause(self.n_points)
            layer[0].append(i)

        generated_nodes = len(self.causes)

        links = []
        while generated_nodes < self.nodes:
            print(
                '--Generating nodes : {} out of ~{}'.format(generated_nodes, self.nodes))
            layer.append([])  # new layer

            num_nodes_layer = np.random.randint(2, len(layer[-2]) + 2)
            for i in range(num_nodes_layer):
                layer[-1].append(generated_nodes)
                # draw causes
                last_idx = layer[-2][-1]
                parents = list(set([np.random.randint(0, last_idx)
                                    for i in range(
                        self.num_max_parents)]))  # np.random.randint(self.num_max_parents - 1, self.num_max_parents))]))
                child = []
                # Compute each cause's contribution
                for par in parents:
                    links.append(['V' + str(par), 'V' + str(generated_nodes)])
                    child.append(
                        effect(self.data['V' + str(par)], self.n_points, self.noise))
                # Combine contributions
                shuffle(child)
                result = child[0]
                for i in child[1:]:
                    rd_func = self.joint_functions[np.random.randint(
                        0, len(self.joint_functions))]
                    result = op.add(result, i)
                # Add a final noise
                rd_func = self.joint_functions[np.random.randint(
                    0, len(self.joint_functions))]
                if rd_func == op.mul:
                    noise_var = noise(self.n_points, self.noise).flatten()
                    result = rd_func(result + abs(min(result)),
                                     noise_var + abs(min(noise_var)))
                    # +abs(min(result))
                else:
                    result = rd_func(result, noise(
                        self.n_points, self.noise).flatten())
                result = scale(result)

                self.data['V' + str(generated_nodes)] = result

                generated_nodes += 1
        self.result_links = pd.DataFrame(links, columns=["Cause", "Effect"])
        print('--Dataset Generated--')
        if gen_cat:
            print('--Converting variables to categorical--')
            actual_cat_rate = 0.0
            self.cat_var = []
            self.cat_data = self.data.copy()
            while actual_cat_rate < self.cat_rate:
                print(
                    '--Converting, Actual rate: {:3.3f}/{}--'.format(actual_cat_rate, self.cat_rate))
                var = np.random.randint(0, self.nodes)
                while var in self.cat_var:
                    var = np.random.randint(0, self.nodes)
                self.cat_var.append(var)
                self.cat_data['V' + str(var)] = rand_bin(
                    list(self.cat_data['V' + str(var)]))
                actual_cat_rate = float(len(self.cat_var)) / self.nodes

            self.cat_var = pd.DataFrame(self.cat_var)
        print('Build Directed Graph')
        self.graph = DirectedGraph()
        self.graph.add_multiple_edges([list(i)+[1] for i in self.result_links.as_matrix()])

        print('--Done !--')
        return self.get_data()

    def get_data(self):
        # Returns Target, Numerical data, Mixed Data and Index of categorical
        # variables
        try:
            return self.graph, self.data, self.cat_data, self.cat_var
        except NameError:
            print('Please compute graph using .generate(), graph not build yet')
            raise NameError

    def save_data(self, filename):
        try:
            self.result_links
        except NameError:
            print('Please compute graph using .generate(), graph not build yet')
            raise NameError
        self.result_links.to_csv(
            filename + '_target.csv', sep=',', index=False)
        self.data.to_csv(filename + '_numdata.csv', sep=',', index=False)
        try:
            self.cat_data.to_csv(filename + '_catdata.csv',
                                 sep=',', index=False)
            self.cat_var.to_csv(filename + '_catindex.csv',
                                sep=',', index=False)
        except AttributeError:
            pass
        print('Saved files : ' + filename)

    def generate_pairs(self, num_pairs):
        pairs_df = pd.DataFrame()
        target_df = pd.DataFrame()
        while len(pairs_df.index) < num_pairs:
            self.fast_build(gen_cat=False)
            for idxlk, link in self.result_links.iterrows():
                if rd.randint(0, 2):
                    df = series_to_cepc_kag(self.data['V' + str(link.Cause)],
                                            self.data['V' + str(link.Effect)],
                                            len(pairs_df.index))
                    tar = pd.DataFrame([['pair' + str(len(pairs_df.index)), 1.0]],
                                       columns=['SampleID', 'Target'])
                else:
                    df = series_to_cepc_kag(self.data['V' + str(link.Effect)],
                                            self.data['V' + str(link.Cause)],
                                            len(pairs_df.index))
                    tar = pd.DataFrame([['pair' + str(len(pairs_df.index)), -1.0]],
                                       columns=['SampleID', 'Target'])

                pairs_df = pd.concat([pairs_df, df])
                target_df = pd.concat([target_df, tar])
        pairs_df.to_csv('p_graphgen_G' +
                        str(self.num_max_parents) +
                        '_N' + str(self.nodes) +
                        '_pairs.csv', index=False)
        target_df.to_csv('p_graphgen_G' +
                         str(self.num_max_parents) +
                         '_N' + str(self.nodes) +
                         '_targets.csv', index=False)
        print('Done!')

        return pairs_df, target_df
