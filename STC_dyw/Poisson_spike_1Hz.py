import numpy as np
import matplotlib.pyplot as plt
import h5py
from brokenaxes import brokenaxes

def poisson_spike_train_L(trail_No,stimu_times,time_spike_train,firing_rate,repeated_times,repeat_interval):
    t_one = np.arange(0, time_spike_train)
    t_totel = np.arange(0, repeated_times*repeat_interval)
    spike_train = np.zeros((stimu_times,len(t_totel)))
    seed = [3,4,5,6,7,8,9,12,13,14,16]
    # seed = [3,5, 7, 8, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 34, 40, 41, 42, 45, 46, 49, 50, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 74, 75, 76, 78, 79, 81, 82, 86, 87, 88, 91, 92, 93, 94, 95, 96, 100]
    seed = [val for val in seed for i in range(int(stimu_times))]
    for m in range(stimu_times):
        count = 0
        np.random.seed(seed[int(m%stimu_times+stimu_times*trail_No)])
        # np.random.seed(seed_n)
        random_p = np.random.rand(len(t_one))
        for i in range(0, len(random_p)):
            if count >= int((time_spike_train / 1000) * firing_rate):
                break
            if random_p[i] < firing_rate / 1000:
                spike_train[m][i] = 1
                count += 1
        # if sum(spike_train[m]) == 4:
        #     seed_use.append(seed_n)
        #     if m == 4:
        #         print(sum(spike_train[m]))
        #         print(spike_train[m])
        # seed_n += 1
    # print(spike_train[2])
    # print(sum(spike_train[2]))
    for i in range(1,repeated_times):
        spike_train[i*time_spike_train:i*time_spike_train+time_spike_train] = spike_train[(i-1)*time_spike_train:(i-1)*time_spike_train+time_spike_train]
    spike_time = preneuron_spike_time(stimu_times,spike_train)
    # print(seed_use)
    # print(len(seed_use))
    return spike_train,spike_time

def preneuron_spike_time(stimu_times,spike_train):
    spike_time = [[] for i in range(stimu_times)]
    # for i in spike_train[0:int(t_now)*time/pre_neuron_firing_rate_h]:
    for m in range(stimu_times):
        for i in range(0,len(spike_train[m])):
            if spike_train[m][i] > 0:
                spike_time[m].append(i)
                # spike_time.append(i)
    # print(np.shape(spike_time))
    return spike_time


if __name__ == '__main__':


    pre_neuron_firing_rate_L = 1  ##Hz
    times_spike_train_one = 1000  ##ms
    repeat_times = 1  ##900
    repeat_interval = 1000  #ms
    t = np.arange(0, repeat_interval) ##1s
    stimu_times = 900
    preneuron_spikes_train,spikes_time = poisson_spike_train_L(0,stimu_times,times_spike_train_one,pre_neuron_firing_rate_L,repeat_times,repeat_interval)
    print(np.shape(spikes_time))

    time_1_hour = np.arange(0, 5 * 60 * 60 * 1000)  ## 1h
    firing_rate_1_hour = np.zeros(5 * 60 * 60 * 1000)  ##1ms/step
    for n in range(0,stimu_times):
        for i in range(0 + repeat_interval*n, 1000 + repeat_interval*n):
            firing_rate_1_hour[i] = pre_neuron_firing_rate_L

    plt.figure(figsize=(7, 4))
    plt.subplot(212)
    plt.plot(t, preneuron_spikes_train[0][0:1000])
    plt.ylabel("spikes")
    plt.xlabel("time (ms)")
    plt.title("preneuron spike train:1Hz")
    plt.xlim(0,1000)
    plt.tight_layout(0.1)
    plt.savefig('1Hz/11.jpg')

    plt.figure(figsize=(7, 4))
    plt.subplot(212)
    plt.plot(t, preneuron_spikes_train[-1][0:1000])
    plt.ylabel("spikes")
    plt.xlabel("time (ms)")
    plt.title("preneuron spike train:1Hz")
    plt.xlim(0,1000)
    plt.tight_layout(0.1)
    plt.savefig('1Hz/12.jpg')

    plt.subplot(211)
    time_scale = 1000*60  ##ms -> min
    plt.plot(time_1_hour/time_scale, firing_rate_1_hour)
    plt.ylabel("firing rate (Hz)")
    plt.xlabel("time (min)")
    plt.title("preneuron firing rate: 1Hz")
    plt.tight_layout(0.1)
    plt.savefig('1Hz/10.jpg')
    # plt.show()

    fig = plt.figure(figsize=(10, 5))
    bax = brokenaxes(xlims=((-2, 30), (270, 302)), hspace=.05, despine=True)
    time_scale = 1000*60  ##ms -> min
    bax.plot(time_1_hour/time_scale, firing_rate_1_hour)
    bax.set_ylabel("firing rate (Hz)")
    bax.set_xlabel("time (min)")
    bax.set_title("preneuron firing rate: 1Hz")
    plt.savefig('1Hz/12.jpg')
    # plt.show()



