"""VARLiNGAM algorithm.

Author: Georgios Koutroulis

.. MIT License
..
.. Copyright (c) 2019 Georgios Koutroulis
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""

import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from cdt.causality.graph import LiNGAM
import pandas as pd


class varLINGAM():
	"""
	# Estimate a VAR-LiNGAM, see
	# - A. HyvÃ¤rinen, S. Shimizu, P.O. Hoyer ((ICML-2008). Causal modelling
	#   combining instantaneous and lagged effects: an identifiable model based
	#   on non-Gaussianity; 
	# - A. HyvÃ¤rinen, K. Zhang, S. Shimizu, P.O. Hoyer (JMLR-2010). Estimation of
	#   a Structural Vector Autoregression Model Using Non-Gaussianity;
	Random generate matrix ids to set zeros.
	Parameters
	----------
	xt : time series
	    Matrix with size nxm (lengthXnum_variables)
	lag: order to estimate the vector autoregressive model 
	Returns
	-------
	Bo, Bhat : Tuple
		Instantaneous and lagged causal coefficients
	"""	
	def __init__(self, lag=1):
		self.lag = lag

	def estimateVARLiNGAM(self, xt):
		I = np.identity(xt.shape[1])

		# -------------------- Step 1: VAR estimation ----------------------------- #
		model = VAR(xt)
		results = model.fit(self.lag)
		Mt_ = results.params[1:, :]
	     
		# -------------------- Step 2: LiNGAM on Residuals ------------------------ #
		resid_VAR = results.resid
		model = LiNGAM()
		data = pd.DataFrame(resid_VAR)
		model._run_LiNGAM(data)

		# ---------------------Step 3: Get instantaneous matrix Bo from LiNGAM----- #
		Bo_ = pd.read_csv("results.csv").values

		# ---------------------Step 4: Calculation of lagged Bhat ----------------- #
		Bhat_ = np.dot((I - Bo_), Mt_)
		return (Bo_, Bhat_)