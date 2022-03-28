import matplotlib.pyplot as plt
import os
from datetime import datetime
from time import time


def plot_simulation(df, params, show=False, save=True,  dir_name='./'):
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(df['time'], df['x0s'])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Jx')
    axs[0].grid(True)

    axs[1].plot(df['time'], df['x1s'])
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('Jy')
    axs[1].grid(True)

    axs[2].plot(df['time'], df['x2s'])
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('frequency')
    axs[2].grid(True)
    if show:
        plt.show()
    if save:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        date = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')
        path = os.path.join(dir_name,
                            'simulation_%s_pid_%r_omega_%r_decoherence_x_y_%r_%r_dt_%r.png' % (date,
                                                                                               os.getpid(),
                                                                                               params.x_0[2],
                                                                                               params.decoherence_x,
                                                                                               params.decoherence_y,
                                                                                               params.dt
                                                                                               ))
        plt.savefig(path)
    plt.close()


def plot_simulation_and_ekf(df, params, show=False, save=True,  dir_name='./'):
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(df['time'], df['x0s'])
    axs[0].plot(df['time'], df['x0s_est'])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Jx')
    axs[0].grid(True)

    axs[1].plot(df['time'], df['x1s'])
    axs[1].plot(df['time'], df['x1s_est'])
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('Jy')
    axs[1].grid(True)

    axs[2].plot(df['time'], df['x2s'])
    axs[1].plot(df['time'], df['x2s_est'])
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('frequency')
    axs[2].grid(True)
    if show:
        plt.show()
    if save:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        date = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')
        path = os.path.join(dir_name,
                            'simulation_ekf_%s_pid_%r_omega_%r_decoherence_x_y_%r_%r_dt_%r.png' % (date,
                                                                                                  os.getpid(),
                                                                                                  params.x_0[2],
                                                                                                  params.decoherence_x,
                                                                                                  params.decoherence_y,
                                                                                                  params.dt
                                                                                                  ))
        plt.savefig(path)
    plt.close()


def plot_err_ekf(df, params, show=False, save=True,  dir_name='./'):
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(df['time'], df['x0_err_cov'])
    axs[0].plot(df['time'], df['mse_x0'])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('mse Jx')
    axs[0].grid(True)

    axs[1].plot(df['time'], df['x1_err_cov'])
    axs[1].plot(df['time'], df['mse_x1'])
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('mse Jy')
    axs[1].grid(True)

    axs[2].plot(df['time'], df['x2_err_cov'])
    axs[2].plot(df['time'], df['mse_x2'])
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('frequency')
    axs[2].grid(True)
    if show:
        plt.show()
    if save:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        date = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')
        path = os.path.join(dir_name,
                            'err_%s_pid_%r_omega_%r_decoherence_x_y_%r_%r_dt_%r.png' % (date,
                                                                                        os.getpid(),
                                                                                        params.x_0[2],
                                                                                        params.decoherence_x,
                                                                                        params.decoherence_y,
                                                                                        params.dt
                                                                                        ))
        plt.savefig(path)
    plt.close()


def plot_err_ekf_loglog(df, params, show=False, save=True,  dir_name='./'):
    fig, axs = plt.subplots(3, 1)
    axs[0].loglog(df['time'], df['x0_err_cov'])
    axs[0].loglog(df['time'], df['mse_x0'])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('mse Jx')
    axs[0].grid(True)

    axs[1].loglog(df['time'], df['x1_err_cov'])
    axs[1].loglog(df['time'], df['mse_x1'])
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('mse Jy')
    axs[1].grid(True)

    axs[2].loglog(df['time'], df['x2_err_cov'])
    axs[2].loglog(df['time'], df['mse_x2'])
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('frequency')
    axs[2].grid(True)
    if show:
        plt.show()
    if save:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        date = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')
        path = os.path.join(dir_name,
                            'err_loglog_%s_pid_%r_omega_%r_decoherence_x_y_%r_%r_dt_%r.png' % (date,
                                                                                               os.getpid(),
                                                                                               params.x_0[2],
                                                                                               params.decoherence_x,
                                                                                               params.decoherence_y,
                                                                                               params.dt
                                                                                               ))
        plt.savefig(path)
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


def plot_simple_model(df, dir_name, params, simulation=True, ekf=True, err=True, err_loglog=True, show=False, save=True):
    if simulation:
        plot_simulation(df, params, dir_name=dir_name, show=show, save=save)
    if ekf:
        plot_simulation_and_ekf(df, params, dir_name=dir_name, show=show, save=save)
    if err:
        plot_err_ekf(df, params, dir_name=dir_name, show=show, save=save)
    if err_loglog:
        plot_err_ekf_loglog(df, params, dir_name=dir_name, show=show, save=save)
    return



