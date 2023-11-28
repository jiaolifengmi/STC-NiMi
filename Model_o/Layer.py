import LIF_model as LIF


class Layer():
    def __init__(self, neu_sim, h, t_sim, neu_type):
        self.neurons = []
        if neu_type == 0:
            for i in range(neu_sim):
                self.neurons.append(LIF.Exc_LIF(h, t_sim, 0, 20))
        else:
            for i in range(neu_sim):
                self.neurons.append(LIF.Inh_LIF(h, t_sim, 0, 5))


if __name__ == '__main__':
    print("test")