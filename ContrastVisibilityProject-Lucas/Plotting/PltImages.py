import matplotlib.pyplot as plt


class ImageGrid:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.fig, self.axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
        # self.fig.tight_layout()
        self.axes = self.axes.flatten() if rows * columns > 1 else [self.axes]

    def add_image(self, image, title, cmap, row, column, vmin=None, vmax=None, color_bar=False):
        index = row * self.columns + column
        if 0 <= index < len(self.axes):
            ax = self.axes[index]
            im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
            if color_bar:
                self.fig.colorbar(im, ax=ax)
        else:
            raise ValueError("Invalid row or column index")

    def show(self):
        plt.show()