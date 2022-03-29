import matplotlib.pyplot as plt
import os
from datetime import datetime


def plot_simulation(df, show=False, save=True, dir_name='./', filename='4.png'):
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
        plt.savefig(os.path.join(dir_name, filename))
    plt.close()


def plot_simulation_and_ekf(df, show=False, save=True, dir_name='./', filename='3.png'):
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
    axs[2].plot(df['time'], df['x2s_est'])
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('frequency')
    axs[2].grid(True)
    if show:
        plt.show()
    if save:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        plt.savefig(os.path.join(dir_name, filename))
    plt.close()


def plot_err_ekf(df, show=False, save=True, dir_name='./', filename='2.png'):
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
        plt.savefig(os.path.join(dir_name, filename))
    plt.close()


def plot_err_ekf_loglog(df, show=False, save=True, dir_name='./', filename='1.png'):
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
        plt.savefig(os.path.join(dir_name, filename))
    plt.close()


def plot_simple_model(df, dir_name, params, simulation=True, ekf=True, err=True, err_loglog=True, show=False,
                      save=True):
    date = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')
    if simulation:
        filename = 'simulation_%s_pid_%r_omega_%r_decoherence_x_y_%r_%r_dt_%r.png' % (date,
                                                                                      os.getpid(),
                                                                                      params.x_0[2],
                                                                                      params.decoherence_x,
                                                                                      params.decoherence_y,
                                                                                      params.dt
                                                                                      )
        plot_simulation(df, dir_name=dir_name, show=show, save=save, filename=filename)
    if ekf:
        filename = 'simulation_ekf_%s_pid_%r_omega_%r_decoherence_x_y_%r_%r_dt_%r.png' % (date,
                                                                                          os.getpid(),
                                                                                          params.x_0[2],
                                                                                          params.decoherence_x,
                                                                                          params.decoherence_y,
                                                                                          params.dt
                                                                                          )
        plot_simulation_and_ekf(df, dir_name=dir_name, show=show, save=save, filename=filename)
    if err:
        filename = 'err_%s_pid_%r_omega_%r_decoherence_x_y_%r_%r_dt_%r.png' % (date,
                                                                               os.getpid(),
                                                                               params.x_0[2],
                                                                               params.decoherence_x,
                                                                               params.decoherence_y,
                                                                               params.dt
                                                                               )
        plot_err_ekf(df, dir_name=dir_name, show=show, save=save, filename=filename)
    if err_loglog:
        filename = 'err_loglog_%s_pid_%r_omega_%r_decoherence_x_y_%r_%r_dt_%r.png' % (date,
                                                                                      os.getpid(),
                                                                                      params.x_0[2],
                                                                                      params.decoherence_x,
                                                                                      params.decoherence_y,
                                                                                      params.dt
                                                                                      )
        plot_err_ekf_loglog(df, dir_name=dir_name, show=show, save=save, filename=filename)
    return


def plot_simple_model_avg(ddf,
                          dir_name,
                          num_reps,
                          simulation=True,
                          ekf=True,
                          err=True,
                          err_loglog=True,
                          show=False,
                          save=True):
    date = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')
    if simulation:
        filename = 'simulation_avg_%s_pid_%r_num_reps_%r.png' % (date,
                                                                 os.getpid(),
                                                                 num_reps
                                                                 )
        plot_simulation(ddf, dir_name=dir_name, show=show, save=save, filename=filename)
    if ekf:
        filename = 'simulation_ekf_avg_%s_pid_%rnum_reps_%r.png' % (date,
                                                                    os.getpid(),
                                                                    num_reps
                                                                    )
        plot_simulation_and_ekf(ddf, dir_name=dir_name, show=show, save=save, filename=filename)
    if err:
        filename = 'err_avg_%s_pid_%r_num_reps_%r.png' % (date,
                                                          os.getpid(),
                                                          num_reps
                                                          )
        plot_err_ekf(ddf, dir_name=dir_name, show=show, save=save, filename=filename)
    if err_loglog:
        filename = 'err_loglog_avg_%s_pid_%rnum_reps_%r.png' % (date,
                                                                os.getpid(),
                                                                num_reps
                                                                )
        plot_err_ekf_loglog(ddf, dir_name=dir_name, show=show, save=save, filename=filename)
    return
