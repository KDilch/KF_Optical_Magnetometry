import matplotlib.pyplot as plt
from os import getpid
from time import time


def plot_xs_sim_ekf_cont(time_arr_simulation, xs, time_arr_filter, xs_est, params):
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(time_arr_simulation, xs[:, 0], time_arr_filter, xs_est[:, 0])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('jx')
    axs[0].grid(True)

    axs[1].plot(time_arr_simulation, xs[:, 1], time_arr_filter, xs_est[:, 1])
    axs[1].set_xlabel('time')
    #TODO plot fft
    # axs[1].axhline(y=abs(2*np.pi*frequencies[np.where(x_fft == np.amax(x_fft))][-1]), color='r', linestyle='-')
    axs[1].set_ylabel('freq')
    axs[1].grid(True)

    axs[2].plot(time_arr_simulation, xs[:, 2], time_arr_filter, xs_est[:, 2])
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('jy')
    axs[2].grid(True)
    # plt.show()
    plt.savefig('data/plots/xs_omega%r_spin_corr%r_pid%r.png' % (params.x_0[2], params.spin_corr_const, getpid()))
    plt.close()


def plot_mse_sim_ekf_cont(time_arr_simulation, xs, xs_est, params):
    """
    :param time_arr_simulation:
    :param xs: n-dim vectors
    :param xs_est:
    :param params:
    :return:
    """
    fig1, axs1 = plt.subplots(3, 1)
    axs1[0].loglog(time_arr_simulation, (xs[:, 0]-xs_est[:, 0])**2)
    axs1[0].set_xlabel('time')
    axs1[0].set_ylabel('jx')
    axs1[0].grid(True)

    axs1[1].loglog(time_arr_simulation, (xs[:, 1]-xs_est[:, 1])**2)
    axs1[1].set_xlabel('time')
    axs1[1].set_ylabel('jy')
    axs1[1].grid(True)

    axs1[2].loglog(time_arr_simulation, (xs[:, 2]-xs_est[:, 2])**2)
    axs1[2].set_xlabel('time')
    axs1[2].set_ylabel('freq')
    axs1[2].grid(True)
    plt.savefig('data/plots/mse_omega%r_spin_corr%r_pid%r.png' % (params.x_0[2], params.spin_corr_const, getpid()))
    plt.close()

def plot_est_cov(time_arr_simulation, xs, xs_est, params, P_ekf_est):
    fig2, axs2 = plt.subplots(3, 1)
    # axs2[0].loglog(time_arr_simulation, (xs[:, 0] - xs_est[:, 0]) ** 2)
    # axs2[0].loglog(time_arr_simulation, P_ekf_est[:, 0, 0])
    axs2[0].plot(time_arr_simulation, (xs[:, 0] - xs_est[:, 0]) ** 2)
    axs2[0].plot(time_arr_simulation, P_ekf_est[:, 0, 0])
    axs2[0].set_xlabel('time')
    axs2[0].set_ylabel('jx')
    axs2[0].grid(True)

    # axs2[1].loglog(time_arr_simulation, (xs[:, 1] - xs_est[:, 1]) ** 2)
    # axs2[1].loglog(time_arr_simulation, P_ekf_est[:, 1, 1])
    axs2[1].plot(time_arr_simulation, (xs[:, 1] - xs_est[:, 1]) ** 2)
    axs2[1].plot(time_arr_simulation, P_ekf_est[:, 1, 1])
    axs2[1].set_xlabel('time')
    axs2[1].set_ylabel('jy')
    axs2[1].grid(True)

    # axs2[2].loglog(time_arr_simulation, (xs[:, 2] - xs_est[:, 2]) ** 2, label="MSE")
    # axs2[2].loglog(time_arr_simulation, P_ekf_est[:, 2, 2], label="Kalman Cov")
    axs2[2].plot(time_arr_simulation, (xs[:, 2] - xs_est[:, 2]) ** 2)
    axs2[2].plot(time_arr_simulation, P_ekf_est[:, 2, 2])
    axs2[2].set_xlabel('time')
    axs2[2].set_ylabel('freq')
    axs2[2].legend()
    axs2[2].grid(True)
    plt.savefig('data/plots/mse_omega_est_cov_filer%r_spin_corr%r_pid%r.png' % (params.x_0[2], params.spin_corr_const, getpid()))
    plt.close()



def plot_avg_xs_from_dataframes(time_arr,
                            simulation_x0_data,
                            simulation_x1_data,
                            simulation_x2_data,
                            ekf_x0_data,
                            ekf_x1_data,
                            ekf_x2_data,
                            params
                            ):
    fig, axs = plt.subplots(3, 1)

    mean_x0 = simulation_x0_data.mean(axis=1)
    mean_ekf_x0 = ekf_x0_data.mean(axis=1)
    mean_x1 = simulation_x1_data.mean(axis=1)
    mean_ekf_x1 = ekf_x1_data.mean(axis=1)
    mean_x2 = simulation_x2_data.mean(axis=1)
    mean_ekf_x2 = ekf_x2_data.mean(axis=1)

    axs[0].plot(time_arr, mean_x0, time_arr, mean_ekf_x0)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Jx')
    axs[0].grid(True)

    axs[1].plot(time_arr, mean_x1, time_arr, mean_ekf_x1)
    axs[1].set_xlabel('time')
    #TODO plot fft
    # axs[1].axhline(y=abs(2*np.pi*frequencies[np.where(x_fft == np.amax(x_fft))][-1]), color='r', linestyle='-')
    axs[1].set_ylabel('Jy')
    axs[1].grid(True)

    axs[2].plot(time_arr, mean_x2, time_arr, mean_ekf_x2)
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('frequency')
    axs[2].grid(True)
    plt.savefig('data/plots_avg/xs_avg_omega%r_spin_corr%r_%r_num_iter%r.png' % (params.x_0[2],
                                                                                    params.spin_corr_const,
                                                                                    time(),
                                                                                    simulation_x0_data.ndim))
    # plt.show()
    plt.close()

def plot_fx_diff(fx, time_arr, xs_sim, xs_est, params):
    res_xs = [abs(fx(xs_sim[i])-fx(xs_est[i])) for i in range(len(xs_sim))]
    fig8, axs8 = plt.subplots(3, 1)

    axs8[0].plot(time_arr, res_xs[:, 0])
    axs8[0].grid(True)

    axs8[1].plot(time_arr, res_xs[:, 1])
    axs8[1].grid(True)

    axs8[2].plot(time_arr, res_xs[:, 2])
    axs8[2].grid(True)

    plt.savefig('data/plots/fx_diff_omega%r_spin_corr%r_%r_num_iter%r.png' % (params.x_0[2],
                                                                                 params.spin_corr_const,
                                                                                 time(),
                                                                                 10))
    plt.close()


def plot_avg_mse_from_dataframes(time_arr,
                                 mse_x0_data,
                                 mse_x1_data,
                                 mse_x2_data,
                                 cov_x0_data,
                                 cov_x1_data,
                                 cov_x2_data,
                                 params):
    fig, axs = plt.subplots(3, 1)

    mean_mse_x0 = mse_x0_data.mean(axis=1)
    mean_mse_x1 = mse_x1_data.mean(axis=1)
    mean_mse_x2 = mse_x2_data.mean(axis=1)
    mean_cov_x0 = cov_x0_data.mean(axis=1)
    mean_cov_x1 = cov_x1_data.mean(axis=1)
    mean_cov_x2 = cov_x2_data.mean(axis=1)

    # axs[0].loglog(time_arr, mean_mse_x0)
    # axs[0].loglog(time_arr, mean_cov_x0)
    axs[0].plot(time_arr, mean_mse_x0)
    axs[0].plot(time_arr, mean_cov_x0)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Jx')
    axs[0].grid(True)

    # axs[1].loglog(time_arr, mean_mse_x1)
    # axs[1].loglog(time_arr, mean_cov_x1)
    axs[1].plot(time_arr, mean_mse_x1)
    axs[1].plot(time_arr, mean_cov_x1)
    axs[1].set_xlabel('time')
    # TODO plot fft
    # axs[1].axhline(y=abs(2*np.pi*frequencies[np.where(x_fft == np.amax(x_fft))][-1]), color='r', linestyle='-')
    axs[1].set_ylabel('Jy')
    axs[1].grid(True)

    # axs[2].loglog(time_arr, mean_mse_x2)
    # axs[2].loglog(time_arr, mean_cov_x2)
    axs[2].plot(time_arr, mean_mse_x2)
    axs[2].plot(time_arr, mean_cov_x2)
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('frequency')
    axs[2].grid(True)
    # plt.show()
    plt.savefig(
        'data/plots_avg/mse_avg_omega%r_spin_corr%r_%r_num_iter%r.png' % (params.x_0[2],
                                                                            params.spin_corr_const,
                                                                            time(),
                                                                            mse_x0_data.ndim)
                                                                            )
    plt.close()
    return

def plot_avg_mse_loglog_from_dataframes(time_arr,
                                 mse_x0_data,
                                 mse_x1_data,
                                 mse_x2_data,
                                 cov_x0_data,
                                 cov_x1_data,
                                 cov_x2_data,
                                 params):
    fig, axs = plt.subplots(3, 1)

    mean_mse_x0 = mse_x0_data.mean(axis=1)
    mean_mse_x1 = mse_x1_data.mean(axis=1)
    mean_mse_x2 = mse_x2_data.mean(axis=1)
    mean_cov_x0 = cov_x0_data.mean(axis=1)
    mean_cov_x1 = cov_x1_data.mean(axis=1)
    mean_cov_x2 = cov_x2_data.mean(axis=1)

    axs[0].loglog(time_arr, mean_mse_x0)
    axs[0].loglog(time_arr, mean_cov_x0)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Jx')
    axs[0].grid(True)

    axs[1].loglog(time_arr, mean_mse_x1)
    axs[1].loglog(time_arr, mean_cov_x1)
    axs[1].set_xlabel('time')
    # TODO plot fft
    # axs[1].axhline(y=abs(2*np.pi*frequencies[np.where(x_fft == np.amax(x_fft))][-1]), color='r', linestyle='-')
    axs[1].set_ylabel('Jy')
    axs[1].grid(True)

    axs[2].loglog(time_arr, mean_mse_x2)
    axs[2].loglog(time_arr, mean_cov_x2)
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('frequency')
    axs[2].grid(True)
    # plt.show()
    plt.savefig(
        'data/plots_avg/mse_avg_loglog_omega%r_spin_corr%r_%r_num_iter%r.png' % (params.x_0[2],
                                                                            params.spin_corr_const,
                                                                            time(),
                                                                            mse_x0_data.ndim)
                                                                            )
    plt.close()
    return

def plot_avg_freq_from_dataframes(time_arr,
                                 simulation_x2_data,
                                 ekf_x2_data,
                                 mse_x2_data,
                                 cov_x2_data,
                                 params):
    fig, axs = plt.subplots(2, 1)

    mean_mse_x2 = mse_x2_data.mean(axis=1)
    mean_cov_x2 = cov_x2_data.mean(axis=1)
    mean_x2 = simulation_x2_data.mean(axis=1)
    mean_ekf_x2 = ekf_x2_data.mean(axis=1)

    axs[0].plot(time_arr, mean_x2, time_arr, mean_ekf_x2)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('field')
    axs[0].grid(True)


    axs[1].loglog(time_arr, mean_mse_x2)
    axs[1].loglog(time_arr, mean_cov_x2)
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('field')
    axs[1].grid(True)
    # plt.show()
    plt.savefig(
        'data/plots_avg/mse_avg_field_omega%r_spin_corr%r_%r_num_iter%r.png' % (params.x_0[2],
                                                                            params.spin_corr_const,
                                                                            time(),
                                                                            100)
                                                                            )
    plt.close()
    return

def plot_avg_omega_with_fft_from_dataframes(time_arr,
                                            simulation_x2_data,
                                            ekf_x2_data,
                                            fft_z_data,
                                            fft_ekf_data,
                                            mse_x2_data,
                                            mse_fft_z,
                                            mse_fft_ekf,
                                            params
                                            ):
    fig, axs = plt.subplots(2, 1)

    mean_x2 = simulation_x2_data.mean(axis=1)
    mean_ekf_x2 = ekf_x2_data.mean(axis=1)
    mean_fft_z = fft_z_data.mean(axis=1)
    mean_fft_ekf = fft_ekf_data.mean(axis=1)
    mean_mse_x2 = mse_x2_data.mean(axis=1)
    mean_mse_fft_z = mse_fft_z.mean(axis=1)
    mean_mse_fft_ekf = mse_fft_ekf.mean(axis=1)

    axs[0].plot(time_arr, mean_x2, label='sim')
    axs[0].plot(time_arr, mean_ekf_x2, label='ekf')
    axs[0].plot(time_arr, mean_fft_z, label='fft_z')
    axs[0].plot(time_arr, mean_fft_ekf, label='fft_ekf')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('freq')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time_arr, mean_mse_x2, time_arr, mean_mse_fft_z, time_arr, mean_mse_fft_ekf)
    axs[1].set_xlabel('time')
    #TODO plot fft
    # axs[1].axhline(y=abs(2*np.pi*frequencies[np.where(x_fft == np.amax(x_fft))][-1]), color='r', linestyle='-')
    axs[1].set_ylabel('Jy')
    axs[1].grid(True)

    plt.savefig('data/plots_avg/fft_avg_omega%r_spin_corr%r_%r_num_iter%r.png' % (params.x_0[2],
                                                                                    params.spin_corr_const,
                                                                                    time(),
                                                                                    mse_x2_data.ndim))
    # plt.show()
    plt.close()



