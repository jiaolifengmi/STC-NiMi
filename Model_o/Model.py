import InputNeuron
import Layer
import Connection as Connection
import Synapsis as Syn


class Model():
    def __init__(self, h, T):
        self.h = h  # 步长
        self.T = T  # 时间范围
        self.Num = int(self.T / self.h)
        self.kesu = True

    def creat_model(self):
        # 构建输入神经元集群以及pattern
        self.input_neurons = InputNeuron.InputNeurons(40, 1, 50, 10)

        # 构建兴奋性神经元集群
        self.exc_neurons = Layer.Layer(100, self.h, self.T, 1).neurons

        # 构建抑制性神经元集群
        self.inh_neurons = Layer.Layer(25, self.h, self.T, 0).neurons

        # 构建脉冲输入连接
        self.syn_10_100 = Connection.input_con(10, 100, 1).input_array

        # 构建100到100的自连接：100：100 一对多
        self.syn_100_100 = Connection.Exc_con(100, 100, 0.1, self.h, self.T).array

        # 构建100到25的连接：100：25 一对多
        self.syn_100_25 = Connection.Exc_con(100, 25, 0.1, self.h, self.T).array

        # 构建25到100的连接：25：100 一对多
        self.syn_25_100 = Connection.Inh_con(25, 100, 0.1, self.h, self.T).array

        # 构建25到25的自连接：25：25 一对多
        self.syn_25_25 = Connection.Inh_con(25, 25, 0.1, self.h, self.T).array

    def train_model(self, train_i):
        # 循环，依次进行spike_T、电导、电流、电压的计算
        for time_i in range(self.Num - 1):
            # print(time_i)
            # 兴奋性神经元集群
            for exc_k in range(len(self.exc_neurons)):
                i_temp_10_100 = 0
                t_s_temp = []
                if self.syn_10_100[exc_k] != 0:
                    for input_t in range(len(self.input_neurons.spikeT[exc_k])):
                        if self.input_neurons.spikeT[exc_k][input_t] <= time_i:
                            t_s_temp.append(self.input_neurons.spikeT[exc_k][input_t])
                        else:
                            break
                    self.syn_10_100[exc_k].t_s = t_s_temp
                    self.syn_10_100[exc_k].calculate_g_i(time_i)
                    i_temp_10_100 = -self.syn_10_100[exc_k].gl[time_i] * (self.exc_neurons[exc_k].V[time_i] - self.syn_10_100[exc_k].E_syn)
                    # print(i_temp_10_100)
                i_temp_100_100 = 0
                if self.syn_100_100.get(str(exc_k)):
                    i_temp_100_100 = self.creat_i(time_i, self.exc_neurons, self.exc_neurons[exc_k], self.syn_100_100[str(exc_k)], True)
                    # print(i_temp_100_100)
                i_temp_25_100 = 0
                if self.syn_25_100.get(str(exc_k)):
                    i_temp_25_100 = self.creat_i(time_i, self.inh_neurons, self.exc_neurons[exc_k], self.syn_25_100[str(exc_k)], False)

                self.exc_neurons[exc_k].I[time_i] = i_temp_10_100 + i_temp_100_100 + i_temp_25_100
                self.exc_neurons[exc_k].euler_i(time_i)

            # 抑制性神经元集群
            for inh_k in range(len(self.inh_neurons)):
                # print(inh_k)
                i_temp_100_25 = 0
                if self.syn_100_25.get(str(inh_k)):
                    i_temp_100_25 = self.creat_i(time_i, self.exc_neurons, self.inh_neurons[inh_k], self.syn_100_25[str(inh_k)], False)

                i_temp_25_25 = 0
                if self.syn_25_25.get(str(inh_k)):
                    i_temp_25_25 = self.creat_i(time_i, self.inh_neurons, self.inh_neurons[inh_k], self.syn_25_25[str(inh_k)], False)

                self.inh_neurons[inh_k].I[time_i] = i_temp_100_25 + i_temp_25_25
                self.inh_neurons[inh_k].euler_i(time_i)

        # 每隔1s记录神经元活动
        print("第"+str(train_i)+"次记录")
        with open("excNeurons_SpikeFr.txt", "w") as f1:
            f1.write("第" + str(train_i) + "次" + ": ")
            for index in range(len(self.exc_neurons)):
                print(self.exc_neurons[index].spikeT)
                f1.write(str(index) + ":" + str(self.exc_neurons[index].spike_number) + "; ")
            f1.write("\n")
        with open("inhNeurons_SpikeFr.txt", "w") as f2:
            f2.write("第" + str(train_i) + "次" + ": ")
            for index in range(len(self.inh_neurons)):
                f2.write(str(index) + ":" + str(self.inh_neurons[index].spike_number) + "; ")
            f2.write("\n")
        with open("exc<>exc-g_max.txt", "w") as f3:
            f3.write("第" + str(train_i) + "次" + "\n")
            for post_s in list(self.syn_100_100.keys()):
                for pre_s in list(self.syn_100_100[post_s]):
                    f3.write(pre_s + "-" + post_s + ":" + str(self.syn_100_100[post_s][pre_s].g_max) + "\n")
            f3.write("\n")

        # 每一秒，神经元的发放次数归零，便于下一秒统计
        for index in range(len(self.exc_neurons)):
            self.exc_neurons[index].spike_number = 0
        for index in range(len(self.inh_neurons)):
            self.inh_neurons[index].spike_number = 0

        # 突触强度第一位重置，下一秒跑
        for post_s in list(self.syn_100_100.keys()):
            for pre_s in list(self.syn_100_100[post_s]):
                self.syn_100_100[post_s][pre_s].g_max[0] = self.syn_100_100[post_s][pre_s].g_max[self.Num - 1]

    # def draw(self):

    def creat_i(self, t, pre_neurons, post_neuron, syn_dic, change_w):
        i_temp = 0
        if change_w:
            for syn_k in list(syn_dic.keys()):
                syn_dic[syn_k].t_s = pre_neurons[int(syn_k)].spikeT
                if self.kesu:
                    syn_dic[syn_k].heb(t, pre_neurons[int(syn_k)].spikeF[t], post_neuron.spikeF[t])
                else:
                    syn_dic[syn_k].calculate_g_i(t)
                i_temp += -syn_dic[syn_k].gl[t] * (
                        post_neuron.V[t] - syn_dic[syn_k].E_syn)
            # print(i_temp)
        else:
            for syn_k in list(syn_dic.keys()):
                syn_dic[syn_k].t_s = pre_neurons[int(syn_k)].spikeT
                syn_dic[syn_k].calculate_g_i(t)
                i_temp += -syn_dic[syn_k].gl[t] * (
                        post_neuron.V[t] - syn_dic[syn_k].E_syn)
            # print(i_temp)
        return i_temp


if __name__ == '__main__':
    model = Model(1.0, 200)
    model.creat_model()
    train_time = 1
    end_num = 5

    # 读取pattern
    with open("pattern.txt", "r") as f:
        patterns = []
        for line in f.readlines():
            patterns.append([])
            for s in line[12:len(line) - 2].split(", "):
                patterns[len(patterns) - 1].append(int(s))
    # 记录输入pattern——spike
    input_pattern_spike = [[], []]
    for k in range(len(model.input_neurons.spikeMat)):
        if k in patterns[0]:
            input_pattern_spike[0].append(model.input_neurons.spikeMat[k])
        elif k in patterns[1]:
            input_pattern_spike[1].append(model.input_neurons.spikeMat[k])
    with open("pattern_spike.txt", "w") as f0:
        for p in range(len(input_pattern_spike)):
            f0.write("pattern" + str(p) + "\n")
            for k in range(len(input_pattern_spike[p])):
                f0.write(str(input_pattern_spike[p][k]) + "\n")

    print("begin")
    for i in range(train_time):
        print("第"+str(i)+"次训练：")
        if i % 2 == 1:
            pattern = patterns[0]
        else:
            pattern = patterns[1]
        print(pattern)
        for j in range(len(model.syn_10_100)):
            if j not in pattern:
                model.syn_10_100[j] = 0
            else:
                model.syn_10_100[j] = Syn.Synapsis(0, [], model.h, model.T)

        model.train_model(i)

        if i == end_num:
            model.kesu = False
            print("关闭可塑性")

    # 连接要完全随机，100*100里呈现10%的连接率
    # 两个pattern公用同样的连接
    # 把pattern跟连接写成一个文件
    # pattern的权重要复用，每1s的学习都要复用上一次权重
    # 每一秒的spike_num要清零

    #
