import cdt
from cdt.generators import RandomGraphGenerator
from cdt.causality.graph import PC
# Might modify params w/ cdt.Settings
graphG = RandomGraphGenerator(num_nodes=10)
graph, data, *c = graphG.generate()

# model = cdt.causality.graphs.CGNN()  # PyTorch
# model.predict(data, graph=graph, nb_jobs=1, train_epochs=5, test_epochs=5, nb_runs=1)

model = PC()  # TensorFlow
out = model.predict(data, graph=None, nb_jobs=1, train_epochs=5, test_epochs=5, nb_runs=1, verbose=False)
print(out)