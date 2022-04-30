from copy import copy
import numpy as np
import torch
import warnings
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float32)

# 定义状态空间
class State:
    def __init__(self,point,N:int,energy=3000.0) -> None:
        self.point=point
        self.H = torch.tensor(np.zeros(N),dtype=torch.float32)
        self.energy = energy
        self.delta = 0
        self.task = self.new_task()
    
    def get_state(self):
        return torch.cat(
            (torch.tensor([self.energy,self.delta]),self.H,self.task)
        )
    
    def new_task(self):
        return torch.tensor([np.random.choice([60.0,90.0]),np.random.choice([3.0,4.5])],dtype=torch.float32)
    
    def __repr__(self) -> str:
        return "State{"+f"point={self.point},H={self.H},energy={self.energy},delta={self.delta},task={self.task}"+"}"


class Policy:
    def __init__(self,o:int,pd:float,fd:float,ro) -> None:
        self.o = o
        self.pd = pd
        self.fd = fd
        self.ro = ro
    
    def __repr__(self) -> str:
        return "Policy{"+f"o={self.o},pd={self.pd},fd={self.fd},ro={self.ro}"+"}"
    
    @staticmethod
    def getPolicy(o: int, N, P: list, F: list):
        p, f = len(P), len(F)
        if o <= 1:
            return Policy(0, 0.0, F[o],o)
        return Policy((o-2) % N+1, P[(o-2)//N], 0.0,o)


class Environment:
    def __init__(self,point,N,MAP) -> None:
        self.state = State(point,N)
        self.map = MAP
        self.vt = torch.tensor(np.array([[0.0,0.0]]),dtype=torch.float32)
        self.t = 0
        self.calculate_channel_gain()
    
    def get_state(self):
        return self.state.get_state()

    def action(self,policy:Policy):
        state = self.get_state()
        # calculate energy and latency
        et = self.__calculate_energy__(policy)
        lt = self.__calculate_latency__(policy)
        # calculate averagy energy and latency
        self.vt[0][0]+=et/(self.t+1)
        self.vt[0][1]+=lt/(self.t+1)
        self.t+=1
        if policy.o>0:self.state.delta = policy.o
        self.__walk__()
        self.calculate_channel_gain()
        self.state.task = self.state.new_task()
        self.state.energy -= et
        return state,policy,self.state.get_state(),(-et,-lt),self.vt

    def __calculate_energy__(self,policy:Policy):
        if policy.o > 0:
            return policy.pd*self.state.task[0] /\
                self.__calculate_transmit(policy.pd, self.state.H[policy.o-1])
        return  (10e-27)*policy.fd*self.state.task[1]

    def __calculate_latency__(self,policy:Policy):
        if policy.o > 0:
            return (.0 if self.state.delta == policy.o else 0.7) + \
                self.state.task[0]/self.__calculate_transmit(
                    policy.pd, self.state.H[policy.o-1]) + self.state.task[1]/5.0
        return self.state.task[1]/policy.fd
    
    def __calculate_transmit(self, pd, ht):
        return 10*np.log2(1+pd*ht/(2 * 10e-13))

    def calculate_channel_gain(self):
        x, y = self.state.point
        for idx, (x_, y_) in enumerate(self.map):
            dt = np.sqrt(np.power(x-x_, 2)+np.power(y-y_, 2))
            self.state.H[idx] = 4.11 * \
                np.power(
                    3*10e8/(4*np.pi*915 * dt), 2.8) / 1e6
    
    def __walk__(self):
        x,y=self.state.point
        seek = np.random.choice(5)
        if seek == 1:x=x-10
        if seek == 2:x=x+10
        if seek == 3:y=y-10
        if seek == 4:y=y+10
        if x > 2000:x = 2000
        if x < 0:x = 0
        if y > 2000:y = 2000
        if y < 0:y = 0
        self.state.point = (x,y)


class DoubleQNet:
    def __init__(self,N:int,P:int,F:int) -> None:
        # 网络参数
        layer_2 = int(np.power(N+4, 2/3)*np.power(N*P+F, 1/3))
        layer_3 = int(np.power(N+4, 1/3)*np.power(N*P+F, 2/3))
        self.online_latency = torch.nn.Sequential(
            torch.nn.Linear(N+4, layer_2),torch.nn.ReLU(),
            torch.nn.Linear(layer_2, layer_3),torch.nn.ReLU(),
            torch.nn.Linear(layer_3, N*P+F),torch.nn.ReLU()
        )
        self.online_energy = torch.nn.Sequential(
            torch.nn.Linear(N+4, layer_2),torch.nn.ReLU(),
            torch.nn.Linear(layer_2, layer_3),torch.nn.ReLU(),
            torch.nn.Linear(layer_3, N*P+F),torch.nn.ReLU()
        )
        self.target_latency = torch.nn.Sequential(
            torch.nn.Linear(N+4, layer_2),torch.nn.ReLU(),
            torch.nn.Linear(layer_2, layer_3),torch.nn.ReLU(),
            torch.nn.Linear(layer_3, N*P+F),torch.nn.ReLU()
        )
        self.target_energy = torch.nn.Sequential(
            torch.nn.Linear(N+4, layer_2),torch.nn.ReLU(),
            torch.nn.Linear(layer_2, layer_3),torch.nn.ReLU(),
            torch.nn.Linear(layer_3, N*P+F),torch.nn.ReLU()
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(3, 12),torch.nn.ReLU(),
            torch.nn.Linear(12,6),torch.nn.ReLU(),
            torch.nn.Linear(6,1)
        )

        # 其他参数
        self.clone_interval = 100
        self.update_setp = 1
        self.optim_l = torch.optim.Adam(self.online_latency.parameters(),lr=0.001)
        self.optim_e = torch.optim.Adam(self.online_energy.parameters(),lr=0.001)
        # self.optim = torch.optim.Adam((self.online_latency.parameters(),self.online_energy.parameters()),lr=0.001)
        self.loss_fn = torch.nn.MSELoss()
    
    def eval(self,input,vt:torch.Tensor,mask,eps:float):
        with torch.no_grad():
            QL, QE = self.online_latency(input), self.online_energy(input)
            wl = torch.cat((QL.view((-1,1)),vt.repeat(len(QL[0]),1)),dim=1)
            we = torch.cat((QE.view((-1,1)),vt.repeat(len(QE[0]),1)),dim=1)
            W = self.attention(torch.cat((wl,we)))
            w1 = W[:len(QL[0]),0]
            w2 = W[len(QL[0]):,0]
            w = torch.nn.Softmax(dim=0)(torch.cat((w1.unsqueeze(0),w2.unsqueeze(0))))
            Q = torch.mul(QL, w[0]) + torch.mul(QE, w[1])
            o = np.random.choice(42) if np.random.random() < eps else torch.argmax(Q).item()
            return o, QL[0][o], QE[0][o]
    
    def train(self,expriences):
        # 更新目标网络参数
        if self.update_setp % self.clone_interval == 0:
            self.target_energy.load_state_dict(self.online_energy.state_dict())
            self.target_latency.load_state_dict(self.online_latency.state_dict())
        # 优化器清除梯度
        self.__zero_grad__()
        # self.optim.zero_grad()
        s = pools[:,0:24]
        a = torch.tensor(pools[:,27].clone().detach(),dtype=torch.long)
        r = pools[:,52:54]
        ns = pools[:,28:52]
        ae,al = torch.argmax(self.online_energy(ns),dim=1),torch.argmax(self.online_latency(ns),dim=1)
        QE,QL = self.target_energy(ns),self.target_latency(ns)
        tqe,tql = 0.9*QE[range(len(QE)),ae],0.9*QL[range(len(QL)),al]
        tre,trl = r[:,0],r[:,1]
        qe,ql = tqe+tre,tql+trl
        rqe,rql = self.online_energy(s)[range(len(QE)),a],self.online_latency(s)[range(len(QE)),a]
        loss_e = self.loss_fn(qe,rqe)
        loss_l = self.loss_fn(ql,rql)
        # loss = loss_e + loss_l
        # loss.backward()
        loss_e.backward()
        loss_l.backward()
        self.optim_l.step()
        self.optim_e.step()
        # self.optim.step()
        self.update_setp+=1
        return loss_e.item(),loss_l.item()
    
    def __zero_grad__(self):
        self.optim_e.zero_grad()
        self.optim_l.zero_grad()


if __name__ == "__main__":
    N = 20
    EPOCH = 10000
    T = 300000
    cpuf_mobile = np.array([1.0, 3.0])
    transp_mobile = np.array([200.0, 400.0])
    k = 10e-27
    W = 10
    I = 2 * 10e-13
    sigma = 0.7
    pools = torch.zeros((10000,56),dtype=torch.float32)

    MAP = np.array([[i*(2000.0/3), j*(2000.0/4)] for i in range(4) for j in range(5)])
    ENV = Environment((1000.0,1000.0),20,MAP)
    DoubleQ = DoubleQNet(N,2,2)
    we = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],dtype=torch.float32)
    wl = 1-we

    optim_a = torch.optim.Adam(DoubleQ.attention.parameters(),lr=0.001)
    loss_fn = torch.nn.MSELoss()

    bt = 16
    train_set = torch.zeros((2*bt,3),dtype=torch.float32)
    labels = torch.zeros((bt,2),dtype=torch.float32)
    i = 0

    write = SummaryWriter("./logs")

    for t in range(T):
        if t % 30 == 0:
            ENV.state.energy = 3000
            ENV.state.point = (1000.0,1000.0)
            ENV.state.delta = 0
            ENV.calculate_channel_gain()
        mask = torch.ones(42)
        o,QL,QE = DoubleQ.eval(ENV.get_state().unsqueeze(0),ENV.vt,mask,0.05)
        idx = torch.argmax(QL*wl+QE*we).item()
        label = torch.tensor([we[idx],wl[idx]])
        policy = Policy.getPolicy(o,N,transp_mobile,cpuf_mobile)
        exprience = ENV.action(policy)
        s,p,ns,r,vt = exprience
        x = torch.cat((QL.unsqueeze(0).unsqueeze(0),vt),dim=1)
        y = torch.cat((QE.unsqueeze(0).unsqueeze(0),vt),dim=1)
        train_set[i] = x
        train_set[i+ bt] = y
        labels[i] = label
        i += 1
        i %=bt
        if i == (bt-1):
            optim_a.zero_grad()
            u = DoubleQ.attention(train_set)
            u1 = u[:bt,:]
            u2 = u[bt:,:]
            ux = torch.cat((u1,u2),dim=1)
            y_ = torch.softmax(ux,dim=1)
            loss_a =loss_fn(y_,labels)
            loss_a.backward()
            optim_a.step()
        pools[t%10000]=torch.cat((s,torch.tensor([p.o,p.fd,p.pd,p.ro],dtype=torch.float32),ns,r[0].unsqueeze(0),r[1].unsqueeze(0),vt[0]))
        if t > 0 and t % 1024 == 0:
            if t<10000:
                loss = DoubleQ.train(pools[np.random.choice(t,512)])
                write.add_scalar("E",loss[0],t)
                write.add_scalar("L",loss[1],t)
            else:
                loss = DoubleQ.train(pools[np.random.choice(10000,512)])
                write.add_scalar("E",loss[0],t)
                write.add_scalar("L",loss[1],t)
            write.flush()
