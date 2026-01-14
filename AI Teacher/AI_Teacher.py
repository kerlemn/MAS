import numpy as np
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

do = False # Plot one
#do = True # Compare performances

class StudentEnv:
    def __init__(self, goal, only_final_reward=False):
        self.goal = goal
        self.only_final_reward=only_final_reward
        self.disposition = np.random.uniform(1.0, 1.5, size=2)
        self.state = None
        self.actions = [0, 1, 2]  # 0: arithmetic, 1: algebra, 2: mixed
        self.reset()

    def reset(self):
        self.state = np.random.uniform(0.0, 0.2, size=2)
        return self.state.copy()

    def action_vector(self, a):
        if a == 0:
            return np.array([1.0, 0.0])
        elif a == 1:
            return np.array([0.0, 1.0])
        else:
            return np.array([1/2, 1/2])

    def step(self, action):
        S = self.state
        A = self.action_vector(action)

        mean = (1/5) * S * (1 - S) * self.disposition * A
        noise = np.random.normal(0.0, 0.01, size=2) * A
        delta = mean + noise

        S_next = np.clip(S + delta, 0.0, 1.0)

        done = np.all(S_next >= self.goal)

        if self.only_final_reward:
            reward = 100 if done else 0
        else:
            reward = np.sum(S_next - S)  # learning gain
        self.state = S_next
        return S_next.copy(), reward, done

class Discretizer:
    def __init__(self, bins=100):
        self.bins = bins

    def discretize(self, S):
        return tuple((S * self.bins).astype(int).clip(0, self.bins - 1))
    
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}

    def get_Q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.actions))
        return self.Q[state]

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        Q = self.get_Q(state)
        max_Q = np.max(Q)
        best_actions = np.flatnonzero(Q == max_Q)
        return int(np.random.choice(best_actions))

def run_experiment(epsilon, only_final_reward, plot_curve=False):
    env = StudentEnv(goal=0.9, only_final_reward=only_final_reward)
    disc = Discretizer(bins=100)
    agent = QLearningAgent(actions=env.actions, epsilon=epsilon)

    episodes = 1000
    max_steps = 150

    times_to_arrival = []
    traces = []

    for ep in range(episodes):
        S = env.reset()
        s = disc.discretize(S)

        trace = []

        for t in range(max_steps):
            a = agent.select_action(s)
            S_next, r, done = env.step(a)
            trace.append(S_next)
            s_next = disc.discretize(S_next)

            Q_s = agent.get_Q(s)
            Q_s_next = agent.get_Q(s_next)

            agent.Q[s][a] += agent.alpha * (
                r + agent.gamma * np.max(Q_s_next) - Q_s[a]
            )

            s = s_next
            if done:
                times_to_arrival.append(t)
                traces.append(trace)
                break
    if plot_curve:
        for seq in traces[-20:]:
            x = [p[0] for p in seq]
            y = [p[1] for p in seq]
            plt.plot(x, y, marker='o', alpha=0.6)
        plt.xlabel("Algebra")
        plt.ylabel("Aritmetic")
        disp = env.disposition
        plt.title(f"Last 20 episodes for a student with dispositions d=[{'{:.2f}'.format(disp[0])},{'{:.2f}'.format(disp[1])}]")
        plt.show()
    return sum(times_to_arrival) / len(times_to_arrival)

if do:
    data = {}

    n_runs = 200
    n_workers = cpu_count()

    for epsilon in [0.0, 0.1]:
        for only_final_reward in [True, False]:

            with Pool(processes=n_workers) as pool:
                avg = pool.starmap(
                    run_experiment,
                    [(epsilon, only_final_reward)] * n_runs
                )

            s1 = "sparse-reward" if only_final_reward else "dense-reward"
            s2 = "e-greedy" if epsilon > 0 else "greedy"
            data[f"{s1}|{s2}"] = avg

    x = range(len(data))

    runs = np.stack(list(data.values()), axis=1)

    plt.figure()

    parts = plt.violinplot(
        runs,
        positions=x,
        vert=False,
        showmedians=True,
        showextrema=False
    )

    plt.yticks(x, data.keys())
    plt.xlabel("Average assignments to reach [0.9,0.9] mastery")
    plt.title("Distribution over 200 samples of 1000 episodes each per configuration")
    plt.show()

else:
    run_experiment(epsilon=0.1, only_final_reward=True, plot_curve=True)