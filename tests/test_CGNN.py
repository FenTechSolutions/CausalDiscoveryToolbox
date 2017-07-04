from cdt.generators import RandomGraphGenerator
from cdt.causality.graphical_models import CGNN
# Might modify params w/ cdt.Settings
graphG = RandomGraphGenerator(num_nodes=10)
graph, data, *c = graphG.generate()

# model = cdt.causality.graphical_models.CGNN()  # PyTorch
# model.predict(data, graph=graph, nb_jobs=1, train_epochs=5, test_epochs=5, nb_runs=1)

model = CGNN(backend='TensorFlow')  # TensorFlow
model.predict(data, graph=graph, nb_jobs=1, train_epochs=5, test_epochs=5, nb_runs=1)
