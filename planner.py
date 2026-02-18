import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class KalmanFilter2D:
    def __init__(self, init_pos):
        # State: [x, y, vx, vy]
        self.x = np.array([init_pos[0], init_pos[1], 0.0, 0.0])
        self.P = np.eye(4)

        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.Q = np.eye(4) * 0.05
        self.R = np.eye(2) * 0.1

    def step(self, z):

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

   
        z = np.array(z)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def predict_next(self):
        """Predict obstacle position at next timestep"""
        x_next = self.F @ self.x
        return int(round(x_next[0])), int(round(x_next[1]))

import heapq


MOVES = [
    (0,  1, 1.0),    # UP (least cost)
    (1,  1, 1.5),    # UP-RIGHT
    (-1, 1, 1.5),    # UP-LEFT
    (1,  0, 2.0),    # RIGHT
    (-1, 0, 2.0),    # LEFT
    (1, -1, 3.0),    # DOWN-RIGHT
    (-1,-1, 3.0),    # DOWN-LEFT
    (0, -1, 4.0),    # DOWN (most costly)
    (0,  0, 0.5),    # WAIT (hover)
]

def dijkstra(grid, start, goal, forbidden):
    H, W = grid.shape

    pq = []
    heapq.heappush(pq, (0.0, start))

    parent = {start: None}
    cost_so_far = {start: 0.0}

    while pq:
        g, current = heapq.heappop(pq)

        if current == goal:
            break

        for dx, dy, move_cost in MOVES:
            nx, ny = current[0] + dx, current[1] + dy
            nxt = (nx, ny)

            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid[ny, nx] == 1 or nxt in forbidden:
                continue

            new_cost = g + move_cost

            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                parent[nxt] = current
                heapq.heappush(pq, (new_cost, nxt))

    if goal not in parent:
        return None


    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]

    return path[::-1]

def run_planner(env, max_steps=80):
    obs_pos = env.get_dynamic_positions()[0]
    kf = KalmanFilter2D(obs_pos)

    uav_pos = env.start
    uav_path = [uav_pos]
    history = []

    for _ in range(max_steps):
        
        obs_pos = env.get_dynamic_positions()[0]

        
        kf.step(obs_pos)

        
        predicted_obs = kf.predict_next()
        
        
        forbidden = {predicted_obs}

       
        path = dijkstra(env.grid, uav_pos, env.goal, forbidden)

        if path is None or len(path) < 2:
            # UAV waits
            history.append({"uav": uav_pos, "obs": obs_pos})
            env.update()
            continue

        uav_pos = path[1]
        uav_path.append(uav_pos)

        history.append({"uav": uav_pos, "obs": obs_pos})

        env.update()

        if uav_pos == env.goal:
            break

    return uav_path, history



def animate(env, path, history):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")

  
    sy, sx = np.where(env.grid == 1)
    ax.scatter(sx + 0.5, sy + 0.5, c="black", s=120, marker="s")

    uav_dot, = ax.plot([], [], "bo", label="UAV")
    obs_dot, = ax.plot([], [], "ro", label="Dynamic UAV")
    path_line, = ax.plot([], [], "b--", alpha=0.6)

    ax.scatter(env.goal[0] + 0.5, env.goal[1] + 0.5,
               c="green", s=200, marker="*", label="Goal")

    ax.grid(True)
    ax.legend()

    def update(frame):
        if frame >= len(history):
            return uav_dot, obs_dot, path_line

        uav = history[frame]["uav"]
        obs = history[frame]["obs"]

        # FIX: pass sequences, not scalars
        uav_dot.set_data([uav[0] + 0.5], [uav[1] + 0.5])
        obs_dot.set_data([obs[0] + 0.5], [obs[1] + 0.5])

        px = [p[0] + 0.5 for p in path[:frame + 1]]
        py = [p[1] + 0.5 for p in path[:frame + 1]]
        path_line.set_data(px, py)

        return uav_dot, obs_dot, path_line

        return uav_dot, obs_dot, path_line

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(history),
        interval=400
    )

    plt.title("Predict-Next-Step + Dijkstra UAV Planning")
    plt.show()

    return ani




if __name__ == "__main__":
    from environment import GridEnvironment

    env = GridEnvironment(30, 30)

    env.set_start(2, 2)
    env.set_goal(27, 27)

    env.add_static_rectangle(5, 5, 6, 4)
    env.add_static_rectangle(0, 8, 3, 2)
    env.add_static_rectangle(10, 15, 12, 5)

    loop = ["RIGHT","RIGHT","UP","UP","LEFT","LEFT","DOWN","DOWN"]
    env.add_dynamic_uav(0, 9, loop)

    path, history = run_planner(env)

    

    animate(env, path, history)
