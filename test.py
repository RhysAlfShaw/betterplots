from bettercorners import cornerplot
import betterstyle as bstyle
import numpy as np


# test the cornerplot function
np.random.seed(0)
data = np.random.randn(10000, 5)

cornerplot(data, 5)
