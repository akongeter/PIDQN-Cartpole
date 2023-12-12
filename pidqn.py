from dqn import *


def pos_table(n, dim):
    """Create a table of positional encodings."""

    def get_angle(x, h):
        return x / np.power(10000, 2 * (h // 2) / dim)

    def get_angle_vec(x):
        return [get_angle(x, j) for j in range(dim)]

    tab = np.array([get_angle_vec(i) for i in range(n)]).astype(float)
    tab[:, 0::2] = np.sin(tab[:, 0::2])
    tab[:, 1::2] = np.cos(tab[:, 1::2])
    return tab


class PIDQN(DQN):
    def __init__(self, env: gym.Env, tau: float = 0.005, lr: float = 0.0003, batch_size: int = 1024, gamma: float = 0.99, eps_start: float = 0.9, eps_end: float = 0.04, eps_decay: int = 1000, n_hid: [int] = None, memory_size: int = 10000, train_freq: int = 1, train_start: int = 1000, persistence=0.2):
        super(PIDQN, self).__init__(env, tau, lr, batch_size, gamma, eps_start, eps_end, eps_decay, n_hid, memory_size, train_freq, train_start)
        self.prev_action = self.env.action_space.sample()
        self.target_net = PIDQNModel(self.num_actions, self.num_observations, self.n_hid, self.prev_action.size, self.device).to(self.device)
        self.policy_net = PIDQNModel(self.num_actions, self.num_observations, self.n_hid, self.prev_action.size, self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.prev_action = torch.tensor([[self.prev_action]], device=self.device)
        self.memory.set_persistence(persistence)

    def save(self, filename: str = "log"):
        super(PIDQN, self).save(filename)

    def curr_algorithm(self):
        return "PIDQN"

    def predict_(self, obs):
        observations = obs
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                self.prev_action = self.call_model("policy", observations).max(1)[1].view(1, 1)
                return self.prev_action, np.zeros(5)
        else:
            self.prev_action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
            return self.prev_action, np.zeros(5)

    @staticmethod
    def load(filename: str = "log", env: gym.Env = None) -> 'PIDQN':
        model_filename = filename + "-model.txt"
        with open(model_filename, "rb") as f:
            pidqn = pickle.load(f)
        if env is not None:
            pidqn.set_environment(env)
        return pidqn


class PIDQNModel(nn.Module):
    def __init__(self, num_actions: int, num_observations: int, n_hid, action_dim: int, device=None):
        super(PIDQNModel, self).__init__()
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.d_q = 32
        self.N_q = 16
        self.d_v = 8
        self.K = nn.Linear(1, self.d_q)
        self.Q = nn.Linear(num_observations, self.d_q)
        self.V = nn.Linear(1, self.d_v)
        self.qinput = torch.from_numpy(pos_table(self.N_q, num_observations)).float().to(device)
        self.dv_pool = nn.Linear(self.d_v, 1)
        self.Nq_to_obs = nn.Linear(self.N_q, num_observations)
        self.DQN = DQNModel(num_actions, num_observations, n_hid)

    def forward(self, x, is_batch=False):
        # x: [#,N]
        x = torch.unsqueeze(x, -1)  # [#,N,1]
        k = self.K(x)  # [#,N,dq]
        q = self.Q(self.qinput)  # [Nq,dq]
        v = self.V(x)  # [#,N,dv]
        p1 = q @ torch.transpose(k, -1, -2)  # [#,Nq,N]
        p2 = torch.div(p1, np.sqrt(self.d_q))
        p3 = torch.tanh(p2)
        res = p3 @ v  # [#,Nq,dv]
        res = torch.tanh(res)
        res = self.dv_pool(res)  # [#,Nq,1]
        res = torch.squeeze(res, -1)  # [#,Nq]
        res = self.Nq_to_obs(res)
        res = self.DQN(res)
        return res


