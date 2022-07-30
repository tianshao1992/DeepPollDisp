import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy import stats
from matplotlib.animation import FuncAnimation

class matplotlib_vision(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        # sbn.set_style('ticks')
        # sbn.set()
        self.fields_name = ["p", "t", "u", "v"]
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30}


    def plot_loss(self, x, y, label, title=None):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)

        plt.plot(x, y, label=label)
        plt.semilogy()
        plt.grid(True)  # 添加网格
        plt.legend(loc="upper right", prop=self.font)
        plt.xlabel('iterations', self.font)
        plt.ylabel('loss value', self.font)
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.title(title, self.font)
        # plt.pause(0.001)


    def plot_scatter(self, true, pred, axis=0, title=None):
        # sbn.set(color_codes=True)

        plt.scatter(np.arange(true.shape[0]), true, marker='*')
        plt.scatter(np.arange(true.shape[0]), pred, marker='.')

        plt.ylabel('target value', self.font)
        plt.xlabel('samples', self.font)
        plt.xticks(fontproperties='Times New Roman', size=25)
        plt.yticks(fontproperties='Times New Roman', size=25)
        plt.grid(True)  # 添加网格
        plt.title(title, self.font)


    def plot_regression(self, true, pred, axis=0, title=None):
        # 所有功率预测误差与真实结果的回归直线
        # sbn.set(color_codes=True)

        max_value = max(true) # math.ceil(max(true)/100)*100
        min_value = min(true) # math.floor(min(true)/100)*100
        split_value = np.linspace(min_value, max_value, 11)

        split_dict = {}
        split_label = np.zeros(len(true), np.int)
        for i in range(len(split_value)):
            split_dict[i] = str(split_value[i])
            index = true >= split_value[i]
            split_label[index] = i + 1


        plt.scatter(true, pred, marker='.')

        plt.plot([min_value, max_value], [min_value, max_value], 'r-', linewidth=5.0)
        plt.fill_between([min_value, max_value], [0.95*min_value, 0.95*max_value], [1.05*min_value, 1.05*max_value],
                         alpha=0.2, color='b')

        # plt.ylim((min_value, max_value))
        plt.xlim((min_value, max_value))
        plt.ylabel('pred value', self.font)
        plt.xlabel('real value', self.font)
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.grid(True)  # 添加网格
        plt.title(title, self.font)
        # plt.ylim((-0.2, 0.2))
        # plt.pause(0.001)


    def plot_error(self, error, title=None):
        # sbn.set_color_codes()
        error = pd.DataFrame(error) * 100
        sbn.distplot(error, bins=20, norm_hist=True, rug=True, fit=stats.norm, kde=False,
                     rug_kws={"color": "g"}, fit_kws={"color": "r", "lw": 3}, hist_kws={"color": "b"})
        # plt.xlim([-1, 1])
        plt.xlabel("predicted relative error / %", self.font)
        plt.ylabel('distribution density', self.font)
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.grid(True)
        # plt.legend()
        plt.title(title, self.font)



    def plot_optimization(self, max_Ys, label=None):

        optim_step = max_Ys.shape[0]

        if len(max_Ys.shape) == 1:
            plt.plot(range(optim_step), max_Ys, 'r-', linewidth=5.0, label="mean_"+ label)  # 50条数据不能错

            plt.grid(True)
            plt.legend(loc="lower right", prop=self.font)
            plt.xlabel('iterations', self.font)
            plt.ylabel(label, self.font)
            plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
            plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
            # plt.title(title, self.font)

        if len(max_Ys.shape) == 2:

            mean = max_Ys.mean(axis=1)  # 计算开盘价的5期平均移动
            std = max_Ys.std(axis=1)
            plt.plot(range(optim_step), mean, 'r-', linewidth=5.0, label="mean_"+ label)  # 50条数据不能错

            plt.fill_between(range(optim_step), mean - std, mean + std,alpha=0.5, color='b', label="confidence")
            plt.fill_between(range(optim_step), max_Ys.min(axis=1), max_Ys.max(axis=1), alpha=0.2, color='g', label="min-max")

            plt.grid(True)
            plt.legend(loc="lower right", prop=self.font)
            plt.xlabel('iterations', self.font)
            plt.ylabel(label, self.font)
            plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
            plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
            # plt.title(title, self.font)

    def plot_fields_2d(self, fields_true, fields_pred, fmin_max=None):

        plt.clf()

        Nf = fields_pred.shape[-1]

        for fi in range(Nf):
            field_true, field_pred = fields_true[:, :, fi], fields_pred[:, :, fi]

            if fmin_max == None:
                fmin, fmax = fields_true.min(axis=(0, 1)), fields_true.max(axis=(0, 1))
            else:
                fmin, fmax = fmin_max[0], fmin_max[1]

            plt.subplot(Nf, 3, 3*fi + 1)
            plt.imshow(field_true, cmap='RdYlBu_r', aspect='auto', interpolation="spline16",)
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            plt.rcParams['font.family'] = 'Times New Roman'
            cb.set_label(self.fields_name[fi], rotation=0, fontdict=self.font, y=1.08)
            plt.grid(False)
            # plt.xlabel('foils', self.font)
            # plt.ylabel('time step', self.font)
            plt.yticks(fontproperties='Times New Roman', size=20)
            plt.xticks(fontproperties='Times New Roman', size=20)
            if fi == 0:
                plt.title('True field $' + self.fields_name[fi] + '(x,y)$', fontsize=20)

            plt.subplot(Nf, 3, 3*fi + 2)
            plt.imshow(field_pred, cmap='RdYlBu_r',  aspect='auto', interpolation="spline16",)
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            plt.rcParams['font.family'] = 'Times New Roman'
            cb.set_label(self.fields_name[fi], rotation=0, fontdict=self.font, y=1.08)
            plt.grid(False)
            # plt.xlabel('foils', self.font)
            # plt.ylabel('time step', self.font)
            plt.yticks(fontproperties='Times New Roman', size=20)
            plt.xticks(fontproperties='Times New Roman', size=20)
            if fi == 0:
                plt.title('Pred field $' + self.fields_name[fi] + '(x,y)$', fontsize=20)

            plt.subplot(Nf, 3, 3*fi + 3)
            plt.imshow(field_pred - field_true,  aspect='auto', cmap='coolwarm', interpolation="spline16",)
            plt.clim(vmin=-max(abs(fmin[fi]), abs(fmax[fi])), vmax=max(abs(fmin[fi]), abs(fmax[fi])))
            cb = plt.colorbar()
            plt.rcParams['font.family'] = 'Times New Roman'
            cb.set_label(self.fields_name[fi], rotation=0, fontdict=self.font, y=1.08)
            plt.grid(False)
            # plt.xlabel('foils', self.font)
            # plt.ylabel('time step', self.font)
            plt.yticks(fontproperties='Times New Roman', size=20)
            plt.xticks(fontproperties='Times New Roman', size=20)
            if fi == 0:
                plt.title('Error $' + self.fields_name[fi] + '(x,y)$', fontsize=20)



    def plot_fields_ms(self, out_true, out_pred, coord, fmin_max=None, if_contour=True):

        plt.clf()

        if fmin_max == None:
            fmin, fmax = out_true.min(axis=(0, 1)), out_true.max(axis=(0, 1))
        else:
            fmin, fmax = fmin_max[0] , fmin_max[1]


        x_pos = np.concatenate((coord[:, :, 0], coord[:, (0,), 0]), axis=1)
        y_pos = np.concatenate((coord[:, :, 1], coord[:, (0,), 1]), axis=1)

        Nf = out_pred.shape[-1]
        ############################# Plotting ###############################
        for fi in range(Nf):

            plt.rcParams['font.size'] = 20

            ########      Exact f(t,x,y)     ###########
            plt.subplot(Nf, 3, 3 * fi + 1)
            # plt.axis((-0.06, 0, 0, 0.18))
            f_true = np.concatenate((out_true[:, :, fi], out_true[:, (0,), fi]), axis=1)
            if if_contour:
                plt.contour(x_pos, y_pos, f_true, levels=20, linestyles='-', linewidths=0.4, colors='k')
            plt.pcolormesh(x_pos, y_pos, f_true, cmap='RdBu_r', shading='gouraud', antialiased=True, snap=True)
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=30)  # 设置色标刻度字体大小
            plt.rcParams['font.family'] = 'Times New Roman'
            # cb.set_label('value', rotation=0, fontdict=self.font, y=1.08)
            plt.rcParams['font.size'] = 20
            plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$y$', fontdict=self.font)
            if fi == 0:
                plt.title('True field $' + self.fields_name[fi] + '(x,y)$', fontsize=30)

            ########     Learned f(t,x,y)     ###########
            plt.subplot(Nf, 3, 3 * fi + 2)
            # plt.axis((-0.06, 0, 0, 0.18))
            f_pred = np.concatenate((out_pred[:, :, fi], out_pred[:, (0,), fi]), axis=1)
            if if_contour:
                plt.contour(x_pos, y_pos, f_pred, levels=20, linestyles='-', linewidths=0.4, colors='k')
            plt.pcolormesh(x_pos, y_pos, f_pred, cmap='RdBu_r', shading='gouraud', antialiased=True, snap=True)
            cb = plt.colorbar()
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb.ax.tick_params(labelsize=30)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 20
            plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$y$', fontdict=self.font)
            if fi == 0:
                plt.title('Pred field $' + self.fields_name[fi] + '(x,y)$', fontsize=30)

            ########     Error f(t,x,y)     ###########
            plt.subplot(Nf, 3, 3 * fi + 3)
            # plt.axis((-0.06, 0, 0, 0.18))
            err = f_true - f_pred
            plt.pcolormesh(x_pos, y_pos, err, cmap='coolwarm', shading='gouraud', antialiased=True, snap=True)
            cb = plt.colorbar()
            # plt.clim(vmin=-max(abs(fmin[fi]), abs(fmax[fi])), vmax=max(abs(fmin[fi]), abs(fmax[fi])))
            cb.ax.tick_params(labelsize=30)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 20
            plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$y$', fontdict=self.font)
            if fi == 0:
                plt.title('field error$' + self.fields_name[fi] + '(x,y)$', fontsize=30)



