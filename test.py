from bettercorners import cornerplot
import betterstyle as bs
import matplotlib.pyplot as plt
import numpy as np

bs.set_style("betterstyle")

# test the cornerplot function
np.random.seed(0)
data = np.random.randn(10000, 5)
fig, ax = cornerplot(data, 5)
plt.show()
