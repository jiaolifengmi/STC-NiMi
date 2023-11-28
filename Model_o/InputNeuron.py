import numpy as np
import matplotlib.pyplot as plt
import math
import random



class InputNeurons():
    def __init__(self, fr, tSim, nTrials, inputSim):
        self.dt = 1/1000
        self.nBins = math.floor(tSim/self.dt)
        self.spikeMat = np.random.rand(nTrials, self.nBins) < fr*self.dt
        self.t = np.arange(0, tSim*1000, self.dt*1000)  # 时间序列
        self.spikeT = []

        for i in range(nTrials):
            self.spikeT.append([])
            for j in range(self.nBins):
                if self.spikeMat[i][j] == 1:
                    self.spikeT[i].append(self.t[j])

        input_neu_ary = random.sample(range(0, nTrials), nTrials)

        # 生成pattern
        j = 0
        k = 10

        with open("pattern.txt", "w") as f:
            for i in range(int(nTrials/inputSim)):
                f.write("pattern " + str(i) + ": ")
                f.write(str(input_neu_ary[j:k]))
                f.write("\n")
                j += 10
                k += 10
                if k > nTrials:
                    break

    # 画图函数
    def draw(self):
        plt.figure()
        plt.title('InputNeurons(spike-t)')
        plt.xlabel('t(ms)')
        plt.ylabel('spike(0/1)')
        plt.plot(self.t, self.spikeMat[0])
        plt.show()


if __name__ == '__main__':
    input_neu = InputNeurons(40, 1, 50, 10)
    print(input_neu.spikeT[4])
    print(len(input_neu.spikeT[24]))
    # input_neu.draw()



