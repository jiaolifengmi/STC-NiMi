import numpy as np
import random
import Synapsis as Syn


class Connection():
    def __init__(self, preSim, postSim, connectionRate):
        self.input_array = [0 for i in range(postSim)]
        self.array = {}
        # 生成所有下标
        conNumber = []
        for i in range(preSim):
            for j in range(postSim):
                conNumber.append([i, j])
        # 随机挑选连接(一对多)
        self.conSim = int(preSim * postSim * connectionRate)
        self.conTemp = random.sample(conNumber, self.conSim)
        self.conTemp.sort()


class input_con(Connection):
    def __init__(self, preSim, postSim, connectionRate):
        Connection.__init__(self, preSim, postSim, connectionRate)


class Exc_con(Connection):
    def __init__(self, preSim, postSim, connectionRate, ts, T):
        Connection.__init__(self, preSim, postSim, connectionRate)
        for k in range(self.conSim):
            if self.array.get(str(self.conTemp[k][1])):
                self.array[str(self.conTemp[k][1])][str(self.conTemp[k][0])] = Syn.Synapsis(0, [], ts, T)
            else:
                self.array[str(self.conTemp[k][1])] = {str(self.conTemp[k][0]): Syn.Synapsis(0, [], ts, T)}


class Inh_con(Connection):
    def __init__(self, preSim, postSim, connectionRate, ts, T):
        Connection.__init__(self, preSim, postSim, connectionRate)
        for k in range(self.conSim):
            if self.array.get(str(self.conTemp[k][1])):
                self.array[str(self.conTemp[k][1])][str(self.conTemp[k][0])] = Syn.Synapsis(-80, [], ts, T)
            else:
                self.array[str(self.conTemp[k][1])] = {str(self.conTemp[k][0]): Syn.Synapsis(-80, [], ts, T)}

if __name__ == '__main__':
    print("test")
    # print(input_con(2, 5, 1, [3, 0], 1, 1000).input_array)
    syn = Exc_con(3, 2, 1, 1, 1000).array
    print(syn)
    print(syn["1"].keys())
    if syn.get("6"):
        print("true")
    else:
        print("false")