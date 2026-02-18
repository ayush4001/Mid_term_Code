import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


DIRECTIONS = {
    "UP": (0, 1),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0)
}

class GridEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height

  
        self.grid = np.zeros((height, width), dtype=int)

        self.dynamic_uavs = []   
        self.start = None
        self.goal = None


    def set_start(self, x, y):
        self.start = (x, y)

    def set_goal(self, x, y):
        self.goal = (x, y)


    def add_static_rectangle(self, x, y, w, h):
        """
        (x, y): bottom-left corner
        w, h  : width & height (cells)
        """
        self.grid[y:y+h, x:x+w] = 1

    def add_dynamic_uav(self, x, y, direction_sequence):
        """
        direction_sequence: list like
        ["RIGHT", "RIGHT", "UP", "UP", "LEFT", "LEFT", "DOWN", "DOWN"]
        """
        self.dynamic_uavs.append({
            "pos": np.array([x, y], dtype=int),
            "path": direction_sequence,
            "step": 0
        })


    def is_valid(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        if self.grid[y, x] == 1:
            return False
        return True


    def update(self):
        occupied = set(tuple(uav["pos"]) for uav in self.dynamic_uavs)

        for uav in self.dynamic_uavs:
            direction = uav["path"][uav["step"]]
            dx, dy = DIRECTIONS[direction]

            nx = uav["pos"][0] + dx
            ny = uav["pos"][1] + dy

 
            if self.is_valid(nx, ny) and (nx, ny) not in occupied:
                uav["pos"] = np.array([nx, ny])

      
            uav["step"] = (uav["step"] + 1) % len(uav["path"])


    def get_dynamic_positions(self):
        return [tuple(uav["pos"]) for uav in self.dynamic_uavs]


def visualize_environment(env):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")

    
    sy, sx = np.where(env.grid == 1)
    if len(sx) > 0:
        ax.scatter(
            sx + 0.5,
            sy + 0.5,
            c="black",
            s=120,
            marker="s",
            label="Static Obstacle"
        )


    dyn_scatter = ax.scatter(
        [], [],
        c="red",
        s=90,
        marker="o",
        label="Dynamic UAV"
    )

   
    if env.start:
        ax.scatter(
            env.start[0] + 0.5,
            env.start[1] + 0.5,
            c="blue",
            s=130,
            label="Start UAV"
        )

 
    if env.goal:
        ax.scatter(
            env.goal[0] + 0.5,
            env.goal[1] + 0.5,
            c="green",
            s=180,
            marker="*",
            label="Goal"
        )

    ax.set_xticks(np.arange(0, env.width + 1, 1))
    ax.set_yticks(np.arange(0, env.height + 1, 1))
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Grid-based UAV Environment (Deterministic Dynamic Paths)")
    ax.legend(loc="upper right")

 
    def update(frame):
        env.update()
        pos = env.get_dynamic_positions()
        if pos:
            xs = [p[0] + 0.5 for p in pos]
            ys = [p[1] + 0.5 for p in pos]
            dyn_scatter.set_offsets(np.c_[xs, ys])
        return dyn_scatter,

    ani = animation.FuncAnimation(
        fig, update, frames=400, interval=300, blit=True
    )

    plt.show()


if __name__ == "__main__":
    env = GridEnvironment(width=30, height=30)


    env.set_start(2, 2)
    env.set_goal(27, 27)


    env.add_static_rectangle(5, 5, 6, 4)
    env.add_static_rectangle(14, 4, 4, 4)
    env.add_static_rectangle(10, 15, 12, 5)


    horizontal_patrol = ["RIGHT"] * 10 + ["LEFT"] * 10

    env.add_dynamic_uav(6, 20, horizontal_patrol)

    visualize_environment(env)
