import matplotlib.pyplot as plt

from ConversionFunctions import convert_to_adrian_space, convert_to_csf_space

CURVE = 0
POINTS = 1

class Graph:
    def __init__(self, title, x_label, y_label):
        self.curves = []
        self.title = title
        self.x_label = x_label
        self.y_label = y_label

    def add_curve(self, x, y, title=None, curve_type=CURVE, color=None, marker=None):
        self.curves.append((x, y, title, curve_type, marker, color, ))

    def set_log_scale(self, axis="xy"):
        if 'x' in axis:
            plt.xscale("log")
        if 'y' in axis:
            plt.yscale("log")

    def set_ylim(self, y_min, y_max):
        plt.ylim((y_min, y_max))

    def set_x_label(self, x_label):
        self.x_label = x_label

    def set_y_label(self, x_label):
        self.y_label = x_label

    def convert_to_adrian_space(self, luminance_background):
        self.curves = [
            (*convert_to_adrian_space(x, y, luminance_background), *rest)
            for (x, y, *rest) in self.curves
        ]

    def convert_to_csf_space(self, luminance_background):
        self.curves = [
            (*convert_to_csf_space(x, y, luminance_background), *rest)
            for (x, y, *rest) in self.curves
        ]

    def show(self):
        for curve in self.curves:
            x, y, title, curve_type, marker, color = curve
            if curve_type == POINTS:
                plt.scatter(x, y, label=title, color=color, marker=marker)
            else:
                plt.plot(x, y, label=title, color=color)

        plt.xlabel(self.x_label, fontsize=16)
        plt.ylabel(self.y_label, fontsize=16)
        plt.title(self.title, fontsize=18)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)

        plt.show()
