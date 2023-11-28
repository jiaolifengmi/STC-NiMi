import numpy as np
import matplotlib.pyplot as plt
import h5py
from brokenaxes import brokenaxes

def poisson_spike_train_H(trail_No,stimu_times,time_spike_train,firing_rate,repeat_times,repeat_interval):   ##不加以处理，直接是随机poisson分布，spikes有密有疏，分布的不太好
    t_one = np.arange(0, time_spike_train)
    t_totel = np.arange(0, repeat_times*repeat_interval)
    spike_train = np.zeros((stimu_times,len(t_totel)))
    seed = [ 7, 4, 5, 8, 10, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 34, 40, 41, 42, 45, 46, 49, 50, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 74, 75, 76, 78, 79, 81, 82, 86, 87, 88, 91, 92, 93, 94, 95, 96, 100]
    for m in range(stimu_times):
        count = 0
        np.random.seed(seed[m+stimu_times*trail_No])
        random_p = np.random.rand(len(t_one))
        for i in range(0, len(random_p)):
            if count > int((time_spike_train/1000)*firing_rate)+1:
                break
            if random_p[i] < firing_rate / 1000:
                spike_train[m][i] = 1
                count += 1
    spike_time = preneuron_spike_time(stimu_times,spike_train)
    return spike_train,spike_time

# def poisson_spike_train_H(time_spike_train,firing_rate,repeat_times,repeat_interval):
#     t_one = np.arange(0, time_spike_train)
#     t_totel = np.arange(0, repeat_times*repeat_interval)
#     spike_train = np.zeros(len(t_totel))
#     np.random.seed(20)
#     random_p = np.random.rand(len(t_one))
#     for i in range(5, len(random_p)):
#         if random_p[i] < firing_rate / 1000:
#             if spike_train[i-1] == 1 or spike_train[i-2] == 1 or spike_train[i-3] == 1 or spike_train[i-4] == 1 or spike_train[i-5] == 1:
#                 spike_train[i] = 0
#             else:
#                 spike_train[i] = 1
#     for i in range(9,len(random_p)):
#         if sum(spike_train[i-9:i]) == 0 and sum(spike_train[i+1:i+6]) == 0:
#             spike_train[i] = 1
#     for i in range(1,repeat_times):
#         spike_train[i*repeat_interval:i*repeat_interval+time_spike_train] = spike_train[(i-1)*repeat_interval:(i-1)*repeat_interval+time_spike_train]
#     spike_time = preneuron_spike_time(spike_train)
#     return spike_train,spike_time

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

    pre_neuron_firing_rate_h = 100  ##Hz
    time_spike_train = 200   ##ms
    repeat_times = 1  ##900
    repeat_interval = 1000  #ms
    t = np.arange(0, repeat_interval)
    stimu_times = 1
    preneuron_spikes_train,spikes_time = poisson_spike_train_H(0,stimu_times,time_spike_train,pre_neuron_firing_rate_h,repeat_times,repeat_interval)
    # print(len(spikes_time[0]))
    # print(len(spikes_time[1]))
    # print(len(spikes_time[2]))
    # print(spikes_time[0])
    # print(spikes_time[1])
    # print(spikes_time[2])
    # print(np.shape(preneuron_spikes_train))

    time_1_hour = np.arange(0, 5 * 60 * 60 * 1000)
    firing_rate_1_hour = np.zeros(5 * 60 * 60 * 1000)  ##1ms/step
    for n in range(0,stimu_times):
        for i in range(0 + 10*60*1000*n, 1000 + 10*60*1000*n):
            firing_rate_1_hour[i] = pre_neuron_firing_rate_h


    fig = plt.figure(figsize=(10, 5))
    bax = brokenaxes(xlims=((-2, 220), (280, 302)), hspace=.05, despine=True)
    time_scale = 1000*60  ##ms -> min
    bax.plot(time_1_hour/time_scale, firing_rate_1_hour)
    bax.set_ylabel("firing rate (Hz)")
    bax.set_xlabel("time (min)")
    bax.set_title("preneuron firing rate: 100Hz")
    plt.savefig('100Hz/12.jpg')

    plt.figure(figsize = (7,4))
    plt.plot(t, preneuron_spikes_train[0])
    plt.ylabel("spikes")
    plt.xlabel("time (ms)")
    plt.title("preneuron spike train: 100Hz")
    plt.savefig('100Hz/11-0.jpg')

    plt.figure(figsize = (7,4))
    plt.plot(t, preneuron_spikes_train[1])
    plt.ylabel("spikes")
    plt.xlabel("time (ms)")
    plt.title("preneuron spike train: 100Hz")
    plt.savefig('100Hz/11-1.jpg')

    plt.figure(figsize = (7,4))
    plt.plot(t, preneuron_spikes_train[2])
    plt.ylabel("spikes")
    plt.xlabel("time (ms)")
    plt.title("preneuron spike train: 100Hz")
    plt.savefig('100Hz/11-2.jpg')


    plt.show()



