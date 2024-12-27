import os
import matplotlib.pyplot as plt


def set_style(style="betterstyle"):
    if style == "betterstyle":
        style_path = os.path.join(os.path.dirname(__file__), "betterstyle.mplstyle")
    else:
        style_path = style
        exit(f"Style file {style} not found")

    plt.style.use(style_path)


def get_style():
    # return the current style currenlty there is only one style
    return "betterstyle"
