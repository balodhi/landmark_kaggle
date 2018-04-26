import torch
import random as rand

class EnsembleDropout(torch.nn.Module):

  def __init__(self):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(EnsembleDropout, self).__init__()
    values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.6]
    v = rand.randint(0, 5)
    self.dropout = torch.nn.Dropout(values[v])

  def forward(self,  x):

    """
    In the forward function we accept a Variable of input data and we must return
    a Variable of output data. We can use Modules defined in the constructor as
    well as arbitrary operators on Variables.
    """
    y1 = self.dropout(x)
    inputSize = y1.size()
    a = inputSize[0]
    b = inputSize[1]
    d1 = y1.view(a, b, 1)

    y2 =  self.dropout(x)
    d2 = y2.view(a, b, 1)

    y3 = self.dropout(x)
    d3 = y3.view(a, b, 1)

    y4 = self.dropout(x)
    d4 = y4.view(a, b, 1)

    y5 = self.dropout(x)
    d5 = y5.view(a, b, 1)

    y6 = self.dropout(x)
    d6 = y6.view(a, b, 1)

    combine = torch.cat((d1, d2, d3,d4, d5, d6), 2)
    out = torch.mean(combine, 2)

    return out
