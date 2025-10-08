from betterplots.bettercorners import cornerplot
import betterplots.betterstyle as bs
from betterplots import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

bs.set_style("betterstyle")

x = np.linspace(0, 2 * np.pi, 100)
offsets = np.linspace(0, 2 * np.pi, 15, endpoint=False)

fig, ax = plt.subplots()
for i in range(15):
    ax.plot(x, np.sin(x - offsets[i]), label=f"Line {i+1}")
ax.set_title("Plot with Grayscale Linestyle Cycling")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.legend()
plt.savefig("test.png")
