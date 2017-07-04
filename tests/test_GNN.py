from cdt.causality.pairwise_models import GNN
from sklearn.preprocessing import scale
import numpy as np

x = np.random.uniform(-1, 1, 500)
y = x + 0.5 * np.random.uniform(0, 1, 500)

x = scale(x)
y = scale(y)

# PyTorch
model = GNN()
print(model.predict_proba(x, y, nb_jobs=1, train_epochs=5, test_epochs=5, nb_runs=1))

# TensorFlow
model = GNN(backend='TensorFlow')
print(model.predict_proba(x, y, nb_jobs=1, train_epochs=5, test_epochs=5, nb_runs=1))
