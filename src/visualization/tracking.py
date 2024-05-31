import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from data.mae.transforms import normalize_coordinates


SCALING_CONSTANT = 100


class TrackingVisualization:
    def __init__(self, tracking_data):
        self.tracking_data = tracking_data

    @classmethod
    def from_file_path(cls, path: str):
        tracking_data = torch.load(path)
        tracking_data = normalize_coordinates(x=tracking_data)
        return cls(tracking_data=tracking_data)

    @classmethod
    def from_tensor(cls, tracking_tensor: torch.tensor):
        return cls(tracking_data=tracking_tensor)

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
    tv = TrackingVisualization.from_file_path(path=path)
    tv.execute()
