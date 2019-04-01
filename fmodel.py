from foolbox.models import PyTorchModel
from fmodels.ensemble import EnsembleNet
import numpy as np


def create_fmodel():

    model = EnsembleNet()
    model.eval()

    def preprocessing(x_):
        import copy
        x = copy.deepcopy(x_)

        
        assert x.ndim in [3, 4]
        if x.ndim == 3:
            x = np.transpose(x, axes=(2, 0, 1))        
        elif x.ndim == 4:
            x = np.transpose(x, axes=(0, 3, 1, 2))
        
        def grad(dmdp):
            assert dmdp.ndim == 3
            dmdx = np.transpose(dmdp, axes=(1, 2, 0))
            return dmdx
        return x, grad

    fmodel = PyTorchModel(model, bounds=(0,255), num_classes=200, channel_axis=3, preprocessing=preprocessing)
    return fmodel

