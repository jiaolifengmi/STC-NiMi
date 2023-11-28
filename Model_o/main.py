import Model
import Synapsis as Syn


if __name__ == '__main__':
    model = Model.Model(1.0, 200)
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
        print("第" + str(i) + "次训练：")
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
