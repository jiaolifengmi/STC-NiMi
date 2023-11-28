import numpy as np
import matplotlib.pyplot as plt
import h5py
from brokenaxes import brokenaxes

def poisson_spike_train_L(trail_No,stimu_times,times_spike_train,firing_rate,repeated_times,repeated_interval):
    t_one = np.arange(0, times_spike_train)
    t_totel = np.arange(0, repeated_times*repeated_interval)
    spike_train = np.zeros((stimu_times, len(t_totel)))
    # seed = [16]
    seed = [0,7,73,31,40,43,51,62,63,73]
    seed = [val for val in seed for i in range(int(stimu_times))]  ##((9900,))
    for m in range(stimu_times):
        count = 0
        np.random.seed(seed[int(m%stimu_times+stimu_times*trail_No)])
        random_p = np.random.rand(len(t_one))
        for i in range(0, len(random_p)):
            if count >= int((times_spike_train / 1000) * firing_rate):
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
        spike_train[i*repeated_interval:i*repeated_interval+times_spike_train] = spike_train[(i-1)*repeated_interval:(i-1)*repeated_interval+times_spike_train]
    spike_time = preneuron_spike_time(stimu_times,spike_train)
    # print(seed_use)
    # print(len(seed_use))
    return spike_train,spike_time
# def poisson_spike_train():   ## non-Poisson distribution（spikes Uniform distribution）
#     t = np.arange(0, time)
#     spike_train = np.zeros(len(t))
#     for i in range(0, len(spike_train),int(1000/pre_neuron_firing_rate_h)):
#         spike_train[i] = 1
#     return spike_train

def preneuron_spike_time(stimu_times,spike_train):
    spike_time = [[] for i in range(stimu_times)]
    # for i in spike_train[0:int(t_now)*time/pre_neuron_firing_rate_h]:
    for m in range(stimu_times):
        for i in range(0,len(spike_train[m])):
            if spike_train[m][i] > 0 and spike_train[m][i+1] <= 0 and spike_train[m][i-1] <= 0:
                spike_time[m].append(i)
    # print(np.shape(spike_time))
    return spike_time


if __name__ == '__main__':


    pre_neuron_firing_rate_L = 20  ##Hz
    times_spike_train_one = 150  ##ms
    repeat_times = 1  ##900
    repeat_interval = 1000  #ms
    t = np.arange(0, 1000) ##1s
    t_preneuron_spike_totel = np.arange(0, repeat_times * repeat_interval)
    stimu_times = 900
    trail_No = 3
    preneuron_spikes_train,spikes_time = poisson_spike_train_L(trail_No-1,stimu_times,times_spike_train_one,pre_neuron_firing_rate_L,repeat_times,repeat_interval)


    # weighH5 = h5py.File('preneuron low_Frequency-spikes data.hdf5', 'w')
    # weighH5.create_dataset(name='spikes_train', data=preneuron_spikes_train)
    # weighH5.create_dataset(name='spikes_time', data=spikes_time)
    # print("preneuron low_Frequency-spikes data h5 file are saved, dataset name：spikes_train  spikes_time")
    # weighH5.close()

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(t_preneuron_spike_totel, preneuron_spikes_train,label = '1Hz 1pulse/s')
    # plt.grid()
    # plt.ylabel("spikes")
    # plt.xlabel("time/ms")
    # plt.title("pre_neuron firing rate: 1Hz")
    # plt.tight_layout(0.1)
    # plt.legend()
    # plt.show()

    time_1_hour = np.arange(0, 5 * 60 * 60 * 1000)  ## 1h
    firing_rate_1_hour1 = np.zeros(5 * 60 * 60 * 1000)  ##1ms/step
    firing_rate_1_hour2 = np.zeros(5 * 60 * 60 * 1000)  ##1ms/step
    firing_rate_1_hour1[500:900*1000+500] = pre_neuron_firing_rate_L


    fig = plt.figure(figsize=(10, 5))
    bax = brokenaxes(xlims=((-2, 70), (270, 302)), hspace=.05, despine=True)
    time_scale = 1000*60  ##ms -> min
    bax.plot(time_1_hour/time_scale, firing_rate_1_hour1)
    bax.set_ylabel("firing rate (Hz)")
    bax.set_xlabel("time (min)")
    bax.set_title("preneuron firing rate: 20Hz")
    plt.savefig('20Hz/1.jpg')

    firing_rate_1_hour2[30*60*1000+500:30*60*1000+900*1000+500] = pre_neuron_firing_rate_L

    fig = plt.figure(figsize=(10, 5))
    bax = brokenaxes(xlims=((-2, 70), (270, 302)), hspace=.05, despine=True)
    time_scale = 1000*60  ##ms -> min
    bax.plot(time_1_hour/time_scale, firing_rate_1_hour2)
    bax.set_ylabel("firing rate (Hz)")
    bax.set_xlabel("time (min)")
    bax.set_title("preneuron firing rate: 20Hz")
    plt.savefig('20Hz/2.jpg')

    plt.figure(figsize=(5, 5))
    plt.subplot(212)
    plt.plot(t, preneuron_spikes_train[0])
    plt.ylabel("spikes")
    plt.xlabel("time (ms)")
    plt.title("preneuron spike train: 20Hz")
    plt.xlim(0,200)
    plt.tight_layout(0.1)
    plt.savefig('20Hz/3.jpg')

    # plt.figure(figsize=(5, 5))
    # plt.subplot(212)
    # plt.plot(t, preneuron_spikes_train[stimu_times])
    # plt.ylabel("spikes")
    # plt.xlabel("time (ms)")
    # plt.title("preneuron spike train: 20Hz")
    # plt.xlim(0,200)
    # plt.tight_layout(0.1)
    # plt.savefig('20Hz/4.jpg')
    #
    # plt.figure(figsize=(5, 5))
    # plt.subplot(212)
    # plt.plot(t, preneuron_spikes_train[stimu_times*2])
    # plt.ylabel("spikes")
    # plt.xlabel("time (ms)")
    # plt.title("preneuron spike train: 20Hz")
    # plt.xlim(0,200)
    # plt.tight_layout(0.1)
    # plt.savefig('20Hz/5.jpg')

    # plt.figure(figsize=(5, 5))
    # plt.subplot(212)
    # plt.plot(t, preneuron_spikes_train[stimu_times*3])
    # plt.ylabel("spikes")
    # plt.xlabel("time (ms)")
    # plt.title("preneuron spike train: 20Hz")
    # plt.xlim(0,200)
    # plt.tight_layout(0.1)
    # plt.savefig('20Hz/6.jpg')
    #
    # plt.figure(figsize=(5, 5))
    # plt.subplot(212)
    # plt.plot(t, preneuron_spikes_train[stimu_times*4])
    # plt.ylabel("spikes")
    # plt.xlabel("time (ms)")
    # plt.title("preneuron spike train: 20Hz")
    # plt.xlim(0,200)
    # plt.tight_layout(0.1)
    # plt.savefig('20Hz/7.jpg')
    #
    # plt.figure(figsize=(5, 5))
    # plt.subplot(212)
    # plt.plot(t, preneuron_spikes_train[stimu_times*5])
    # plt.ylabel("spikes")
    # plt.xlabel("time (ms)")
    # plt.title("preneuron spike train: 20Hz")
    # plt.xlim(0,200)
    # plt.tight_layout(0.1)
    # plt.savefig('20Hz/8.jpg')
    #
    # plt.figure(figsize=(5, 5))
    # plt.subplot(212)
    # plt.plot(t, preneuron_spikes_train[stimu_times*6])
    # plt.ylabel("spikes")
    # plt.xlabel("time (ms)")
    # plt.title("preneuron spike train: 20Hz")
    # plt.xlim(0,200)
    # plt.tight_layout(0.1)
    # plt.savefig('20Hz/9.jpg')
    #
    # plt.figure(figsize=(5, 5))
    # plt.subplot(212)
    # plt.plot(t, preneuron_spikes_train[stimu_times*7])
    # plt.ylabel("spikes")
    # plt.xlabel("time (ms)")
    # plt.title("preneuron spike train: 20Hz")
    # plt.xlim(0,200)
    # plt.tight_layout(0.1)
    # plt.savefig('20Hz/10.jpg')
    #
    # plt.figure(figsize=(5, 5))
    # plt.subplot(212)
    # plt.plot(t, preneuron_spikes_train[stimu_times*8])
    # plt.ylabel("spikes")
    # plt.xlabel("time (ms)")
    # plt.title("preneuron spike train: 20Hz")
    # plt.xlim(0,200)
    # plt.tight_layout(0.1)
    # plt.savefig('20Hz/11.jpg')
    #
    # plt.figure(figsize=(5, 5))
    # plt.subplot(212)
    # plt.plot(t, preneuron_spikes_train[stimu_times*9])
    # plt.ylabel("spikes")
    # plt.xlabel("time (ms)")
    # plt.title("preneuron spike train: 20Hz")
    # plt.xlim(0,200)
    # plt.tight_layout(0.1)
    # plt.savefig('20Hz/12.jpg')

    # plt.show()


