from copy import copy
import numpy as np
import torch


class Policy():
    '''Policy
    '''

    def __init__(self, o: int, pd: float, fd: float) -> None:
        self.o = o
        self.pd = pd
        self.fd = fd

    def __repr__(self) -> str:
        return "Policy{"+f"o={self.o},pd={self.pd},fd={self.fd}"+"}"

    @staticmethod
    def getPolicy(action: int, N, P: list, F: list):
        p, f = len(P), len(F)
        if action <= 1:
            return Policy(0, 0.0, F[action])
        return Policy((action-2) % N+1, P[(action-2)//N], 0.0)


class Environment():
    '''Environment
    '''

    def __init__(self, N, E, k, W, I, ef, points: tuple, task, map: list[tuple]) -> None:
        self.N = N
        self.k = k
        self.map = map
        self.W = W
        self.I = I
        self.ef = ef
        self.vt = [.0, .0]
        self.t = 0
        self.points = points
        self.state = {'E': E, 'delta': 0, 'points': points,
                      'H': np.zeros(N), 'task': task}
        self.__calculate_channel_gain()

    def action(self, action: Policy):
        state = copy(self.state)
        # calculate energy
        et = self.__calculate_energy(action)
        lt = self.__calculate_latency(action)
        self.t += 1
        self.vt = [(self.vt[0]*self.t+lt)/self.t,
                   (self.vt[1]*self.t+et)/self.t]
        self.state['E'] -= et
        # update user profile
        if action.o > 0:
            self.state["delta"] = action.o
        # update the channel gain && user task
        self.__walk()
        self.__calculate_channel_gain()
        self.state['task'] = np.array(
            [np.random.choice([60, 90]), np.random.choice([3, 4, 5])], dtype=float)
        return state, action, -lt, -et, copy(self.state), self.vt

    def __calculate_latency(self, action: Policy):
        if action.o > 0:
            return (.0 if self.state['delta'] == action.o else 0.7) + \
                self.state['task'][0]/self.__calculate_transmit(
                    action.pd, self.state['H'][action.o-1]) + self.state['task'][1]/self.ef
        return self.state['task'][1]/action.fd

    def __calculate_energy(self, action: Policy):
        if action.o > 0:
            return action.pd*self.state['task'][0] /\
                self.__calculate_transmit(
                    action.pd, self.state['H'][action.o-1])
        return self.k*action.fd*self.state['task'][1]

    def __calculate_channel_gain(self):
        x, y = self.state['points']
        for idx, (x_, y_) in enumerate(self.map):
            dt = np.sqrt(np.power(x-x_, 2)+np.power(y-y_, 2))
            self.state['H'][idx] = 4.11 * \
                np.power(
                    3*10e8/(4*np.pi*915 * dt), 2.8) / 1e6

    def __calculate_transmit(self, pd, ht):
        return self.W*np.log2(1+pd*ht/self.I)

    def __walk(self):
        x, y = self.state['points']
        walk_distance, walk_direction = np.random.random()*10, np.random.random()*np.pi*2
        x_, y_ = x+walk_distance * \
            np.cos(walk_direction), y+walk_distance*np.sin(walk_direction)
        if x_ > 2000:
            x_ = 2000
        if x_ < 0:
            x_ = 0
        if y_ > 2000:
            y_ = 2000
        if y_ < 0:
            y_ = 0
        self.state['points'] = (x_, y_)

    def getState(self):
        # N+4
        result = [self.state['E'], self.state['delta']] + \
            list(self.state['H']) + \
            [self.state['task'][0], self.state['task'][1]]
        return result

    def getMask(self):
        # can connected at the edge of the range 200m
        x, y = self.state['points']
        mask = torch.zeros(self.N)
        for idx, (x_, y_) in enumerate(self.map):
            if np.sqrt(np.power(x-x_, 2)+np.power(y-y_, 2)) <= 1000:
                mask[idx] = 1
        return mask


class DoubleNet():
    def __init__(self, N, P, F) -> None:
        layer_2 = int(np.power(N+4, 2/3)*np.power(N*P+F, 1/3))
        layer_3 = int(np.power(N+4, 1/3)*np.power(N*P+F, 2/3))
        self.online_latency = torch.nn.Sequential(
            torch.nn.Linear(N+4, layer_2),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_2, layer_3),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_3, N*P+F),
            torch.nn.ReLU()
        )
        self.online_energy = torch.nn.Sequential(
            torch.nn.Linear(N+4, layer_2),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_2, layer_3),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_3, N*P+F),
            torch.nn.ReLU()
        )
        self.target_latency = torch.nn.Sequential(
            torch.nn.Linear(N+4, layer_2),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_2, layer_3),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_3, N*P+F),
            torch.nn.ReLU()
        )
        self.target_energy = torch.nn.Sequential(
            torch.nn.Linear(N+4, layer_2),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_2, layer_3),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_3, N*P+F),
            torch.nn.ReLU()
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(2+N*P+F, int(np.power(2+N*P+F, 2/3))),
            torch.nn.ReLU(),
            torch.nn.Linear(int(np.power(2+N*P+F, 2/3)),
                            int(np.power(2+N*P+F, 1/3))),
            torch.nn.ReLU(),
            torch.nn.Linear(int(np.power(2+N*P+F, 1/3)), N*P+F),
            torch.nn.Softmax(dim=0)
        )

        # copy param
        self.target_latency.load_state_dict(self.online_latency.state_dict())
        self.target_energy.load_state_dict(self.online_energy.state_dict())

        self.target_network_update_interval = 1

        self.exprience_pool = []
        self.lossfn = torch.nn.MSELoss()
        self.optiml = torch.optim.SGD(
            self.online_latency.parameters(), lr=0.1)
        self.optime = torch.optim.SGD(self.online_energy.parameters(), lr=0.1)

    def eval(self, input, vt, mask, eps: float):
        with torch.no_grad():
            QL, QE = self.online_latency(input), self.online_energy(input)
        W = self.attention(
            torch.cat((torch.cat((QL, vt), dim=1), torch.cat((QE, vt), dim=1))))
        Q = torch.mul(torch.mul(QL, W[0]) + torch.mul(QE, W[1]), mask)
        action = np.random.choice(
            N*P+F) if np.random.random() < eps else torch.argmax(Q).item()
        return action, QL[0][action], QE[0][action]

    def train(self, expriences):
        if self.target_network_update_interval % 2 == 0:
            online_state_dict = self.online_latency.state_dict(), self.online_energy.state_dict()
        # self.optiml.zero_grad()
        # self.optime.zero_grad()
        s = torch.tensor([[state['E'], state['delta']]+list(state['H'])+[state['task'][0], state['task'][1]]
                          for (state, action, _lt, _et, state_next, vt), mask in expriences], dtype=torch.float32)
        s1 = torch.tensor([[state_next['E'], state_next['delta']]+list(state_next['H'])+[state_next['task'][0], state_next['task'][1]]
                           for (state, action, _lt, _et, state_next, vt), mask in expriences], dtype=torch.float32)
        mask = torch.tensor(np.array([mask_.numpy()
                            for _, mask_ in expriences]))
        action = [action.o for (state, action, _lt, _et,
                                state_next, vt), _ in expriences]
        lt = torch.tensor([_lt for (state, action, _lt, _et,
                                    state_next, vt), _ in expriences])
        et = torch.tensor([_et for (state, action, _lt, _et,
                                    state_next, vt), _ in expriences])
        QL, QE = self.online_latency(
            s)[range(len(action)), action], self.online_energy(s)[range(len(action)), action]
        al, ae = torch.argmax(torch.mul(self.online_latency(s1), mask), dim=1), torch.argmax(
            torch.mul(self.online_energy(s1), mask), dim=1)
        ql, qe = lt + 0.8*self.target_latency(s1)[range(len(al)),
                                                  al], et + 0.8*self.target_energy(s1)[range(len(ae)), ae]
        lossl = torch.pow(ql-QL, 2)
        losse = torch.pow(qe-QE, 2)
        # dtl = 0.01 * \
        #     torch.mul(QL - ql, torch.autograd.grad(lossl,
        #               QL, torch.ones_like(QL))[0])
        # dte = 0.01 * \
        #     torch.mul(QE - qe, torch.autograd.grad(losse,
        #               QE, torch.ones_like(QE))[0])
        # for param in self.online_latency.parameters():
        #     param = param +dtl
        # for param in self.online_energy.parameters():
        #     param.add_(dte)
        # self.optiml.step()
        # self.optime.step()
        if self.target_network_update_interval % 2 == 0:
            self.target_latency.load_state_dict(online_state_dict[0])
            self.target_energy.load_state_dict(online_state_dict[1])
        return 1

    def train_attention(self, input):
        return self.attention(input)


if __name__ == "__main__":
    # define some param
    N = 20
    T = 3000
    cpuf_edge = [5, 10]
    size_task = [60, 90]
    cpuc_task = [3, 4, 5]
    k = 10e-27
    W = 10
    I = 2 * 10e-13
    sigma = 0.7
    cpuf_mobile = [1, 3]
    transp_mobile = [200, 400]
    F = 2
    P = 2
    E0 = 3000
    pool = 1024

    interval = 2000.0/3, 2000.0/4

    map = [(i*interval[0], j*interval[1]) for i in range(4) for j in range(5)]

    env = Environment(N, E0, k, W, I, 5, (1000., 1000.), (60.0, 4.0), map)
    doublenet = DoubleNet(N, P, F)

    exprience_pool = []
    param_pool = [(i/10., 1.-i/10.) for i in range(11)]

    lossfn = torch.nn.MSELoss()
    optim = torch.optim.SGD(doublenet.attention.parameters(), lr=0.01)

    for t in range(T):
        optim.zero_grad()
        mask_ = env.getMask()
        mask = torch.cat((torch.tensor([1., 1.]), mask_, mask_))
        action, QL, QE = doublenet.eval(torch.tensor(
            [env.getState()], dtype=torch.float32), torch.tensor([env.vt], dtype=torch.float32), mask, 0.1)

        policy = Policy.getPolicy(int(action), N, [200, 400], [1, 3])
        exprience = env.action(policy)
        if len(exprience_pool) <= 1024:
            exprience_pool.append((exprience, mask))
        else:
            exprience_pool[t % 1024] = (exprience, mask)
        QL_ = torch.tensor(
            [list(np.repeat([QL.item()], N*P+F))+exprience[-1]], dtype=torch.float32)
        QE_ = torch.tensor(
            [list(np.repeat([QE.item()], N*P+F))+exprience[-1]], dtype=torch.float32)
        w_ = doublenet.train_attention(torch.cat((QL_, QE_)))
        w = param_pool[np.argmax([QL*wl+QE*we for (wl, we) in param_pool])]
        loss = lossfn(w_, np.repeat(torch.tensor(
            [[w[0]], [w[1]]], dtype=torch.float32), 42, 1))
        loss.backward()
        if t > 0 and t % 100 == 0:
            if t <= 64:
                loss = doublenet.train(exprience_pool)
            else:
                loss = doublenet.train([exprience_pool[i]
                                        for i in np.random.choice(len(exprience_pool), 64)])
            print("Train online network loss:", loss)
            break
        optim.step()
