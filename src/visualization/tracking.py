import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, Arc
from data.mae.transforms import normalize_coordinates


SCALING_CONSTANT = 100


class TrackingVisualization:
    def __init__(self, path):
        self.tracking_data = torch.load(path)
        self.tracking_data = normalize_coordinates(x=self.tracking_data)

    def update_radius(self, i, player_circles, ball_circle, annotations):
        for j, circle in enumerate(player_circles):
            circle.center = (
                self.tracking_data[j + 1][i][0] * SCALING_CONSTANT,
                self.tracking_data[j + 1][i][1] * SCALING_CONSTANT,
            )
            annotations[j].set_position(circle.center)
        ball_circle.center = (
            self.tracking_data[0][i][0] * SCALING_CONSTANT,
            self.tracking_data[0][i][1] * SCALING_CONSTANT,
        )
        return player_circles, ball_circle

    def execute(self):
        ax = plt.axes(
            xlim=(-1 * SCALING_CONSTANT, 1 * SCALING_CONSTANT), ylim=(-1 * SCALING_CONSTANT, 1 * SCALING_CONSTANT)
        )
        ax.axis("off")
        fig = plt.gcf()
        ax.grid(False)

        annotations = [
            ax.annotate(
                i, xy=[0, 0], color="w", horizontalalignment="center", verticalalignment="center", fontweight="bold"
            )
            for i in range(10)
        ]

        player_circles = [plt.Circle((0, 0), 5, color="blue" if i < 5 else "red") for i in range(10)]
        ball_circle = plt.Circle((0, 0), 3, color="black")
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
            fig,
            self.update_radius,
            fargs=(player_circles, ball_circle, annotations),
            frames=self.tracking_data.size(dim=1),
            interval=150,
        )
        plt.show()


def do_work(path: str):
    tv = TrackingVisualization(path=path)
    tv.execute()
