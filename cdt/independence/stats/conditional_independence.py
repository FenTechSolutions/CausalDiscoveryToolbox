from ...utils.Settings import SETTINGS

class CI_Test(object):
    def __init__(self, **kwargs):
        super(CI_Test, self).__init__()

    def predict(X, Y, Z, **kwargs):
        return NotImplementedError

    
# Mutual Information Based
class CI_MI(CI_Test):
    def __init__(self, **kwargs):
        parametric = kwargs.get("parametric", False)
        semi_parametric = kwargs.get("semi_parametric", False)
        super(CI_MI, self).__init__()
        
    def predict(X, Y, Z, **kwargs):
        return 0 

# Pearson Based
class CI_Pearson(CI_Test):
    def __init__(self, **kwargs):
        parametric = kwargs.get("parametric", False)
        semi_parametric = kwargs.get("semi_parametric", False)
        super(CI_Pearson, self).__init__()
        
    def predict(X, Y, Z, **kwargs):
        return 0 


# HSIC
class CI_HSIC(CI_Test):
    def __init__(self, **kwargs):
        parametric = kwargs.get("parametric", False)
        semi_parametric = kwargs.get("semi_parametric", False)
        super(CI_HSIC, self).__init__()
        
    def predict(X, Y, Z, **kwargs):
        return 0 
