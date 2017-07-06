import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from PynPoint2.util import detect_peaks
import numpy as np


class PeakDetection(object):
    def __init__(self,
                 data_in,
                 resolution=5):
        self.m_data = data_in
        self.m_grid = np.linspace(min(data_in), max(data_in),
                                  np.abs(max(data_in) - min(data_in))*resolution)
        self.m_resolution = 1./resolution

        self.m_pdf = []
        self.m_peak_list = []
        self.m_extrema_list = []

    @staticmethod
    def _kde_sklearn(x, x_grid, bandwidth=2.0, **kwargs):
        # Kernel Density Estimation with Scikit-learn
        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(x[:, np.newaxis])
        # score_samples() returns the log-likelihood of the samples
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        return np.exp(log_pdf)

    def calc_pdf(self,
                 bandwidth_in=4.0):
        print bandwidth_in
        print self.m_data
        print self.m_data.shape
        
        self.m_pdf = self._kde_sklearn(self.m_data,
                                       self.m_grid,
                                       bandwidth=bandwidth_in)

    def plot_pdf(self):
        print "max mode " + str(self.m_grid[np.argsort(self.m_pdf)[-1]])
        print "mean " + str(np.mean(self.m_data))
        print "median " + str(np.median(self.m_data))

        mean_line = plt.axvline(np.mean(self.m_data),
                                linestyle="--",
                                color="cyan",
                                label='Mean',
                                lw=3)
        median_line = plt.axvline(np.median(self.m_data),
                                  linestyle="--",
                                  color="magenta",
                                  label='Median',
                                  lw=3)

        if self.m_peak_list:
            for point in self.m_peak_list:
                if point[3] is "min":
                    point_color = "blue"
                else:
                    point_color = "red"
                plt.scatter(point[1], point[2], color=point_color, s=50)
            plt.plot(self.m_grid,
                     self.m_pdf,
                     color='blue',
                     alpha=0.5,
                     lw=3)
            plt.xlim(self.m_peak_list[0][1], self.m_peak_list[len(self.m_peak_list)-1][1])
        else:
            plt.plot(self.m_grid, self.m_pdf, color='blue', alpha=0.5, lw=3)
            plt.xlim(min(self.m_grid), max(self.m_grid))

        if self.m_extrema_list:
            colors = ["r", "b", "g", "y", "b", "m", "c"]
            i = 0
            for extrema in self.m_extrema_list:
                plt.fill_between(self.m_grid[extrema[3]:extrema[4]],
                                 0,
                                 self.m_pdf[extrema[3]:extrema[4]],
                                 facecolor=colors[i],
                                 alpha=0.4)
                plt.axvline(extrema[0],
                            linestyle="dotted",
                            alpha=0.4,
                            color=colors[i])
                plt.text(extrema[0],
                         self.m_pdf[extrema[5]] / 2.,
                         str(extrema[2])[0:4],
                         fontsize=10,
                         ha="center")
                i += 1

        plt.title("KDE - PDF")
        plt.ylim(0, )
        plt.legend(handles=[mean_line, median_line])
        plt.show()

    def get_num_peaks(self):
        n = 0
        for peak in self.m_peak_list:
            if peak[3] is "max":
                n += 1
        return n

    def get_best_peak_mode(self):
        best_peak = None
        if self.m_extrema_list:
            best_acc = 0

            for peak in self.m_extrema_list:
                if peak[2] > best_acc:
                    best_acc = peak[2]
                    best_peak = peak[1]

        return best_peak

    def get_best_peak_mean(self):
        best_peak = None
        if self.m_extrema_list:
            best_acc = 0

            for peak in self.m_extrema_list:
                if peak[2] > best_acc:
                    best_acc = peak[2]
                    best_peak = peak[0]

        return best_peak

    def get_lowest_peak_mode(self):
        if self.m_extrema_list:
            return self.m_extrema_list[0][1]

    def get_lowest_peak_mean(self):
        if self.m_extrema_list:
            return self.m_extrema_list[0][0]

    def get_highest_peak_mode(self):
        if self.m_extrema_list:
            return self.m_extrema_list[-1][1]

    def get_highest_peak_mean(self):
        if self.m_extrema_list:
            return self.m_extrema_list[-1][0]

    def get_peak_accuray(self):
        best_acc = 0
        if self.m_extrema_list:

            for peak in self.m_extrema_list:
                if peak[2] > best_acc:
                    best_acc = peak[2]

        return best_acc

    def detect_peaks(self,
                     bandwidth_space=np.linspace(1.0, 20, 50),
                     max_peaks=2):

        for bw in bandwidth_space:
            self.calc_pdf(bandwidth_in=bw)
            self._init_peaks()
            self._clean_peaks()

            if self.get_num_peaks() <= max_peaks:
                break

    def _init_peaks(self):

        # look for minima
        min_ind = detect_peaks(self.m_pdf,
                               show=False,
                               valley=True)

        # append last and first index
        min_ind = np.append(0, min_ind)
        min_ind = np.append(min_ind, len(self.m_grid) - 1)
        min_x = self.m_grid[min_ind]
        min_y = self.m_pdf[min_ind]
        self.m_peak_list = zip(min_ind, min_x, min_y, ["min"] * len(min_x))

        # look for maxima
        max_ind = detect_peaks(self.m_pdf,
                               show=False,
                               mph=np.max(self.m_pdf) / 10.,
                               valley=False)
        max_x = self.m_grid[max_ind]
        may_y = self.m_pdf[max_ind]
        self.m_peak_list += zip(max_ind, max_x, may_y, ["max"] * len(max_x))
        self.m_peak_list.sort(key=lambda tup: tup[0])

    def _clean_peaks(self,
                     height_threshold=20.):

        # search for minima with minima neighbors
        new_peaks = []

        i = 0
        while i < len(self.m_peak_list):
            n = 1.
            tmp_type = self.m_peak_list[i][3]
            tmp_mean_ind = self.m_peak_list[i][0]
            tmp_mean_x = self.m_peak_list[i][1]
            tmp_mean_y = self.m_peak_list[i][2]
            i += 1

            while i < len(self.m_peak_list) and tmp_type == self.m_peak_list[i][3]:
                tmp_mean_ind += self.m_peak_list[i][0]
                tmp_mean_x += self.m_peak_list[i][1]
                tmp_mean_y += self.m_peak_list[i][2]
                n += 1.
                i += 1

            new_peaks.append((int(tmp_mean_ind / n), tmp_mean_x / n, tmp_mean_y / n, tmp_type))

        self.m_peak_list = new_peaks

        # search for maxima which have a similar height than their next minimum
        distances = []
        for i in range(len(self.m_peak_list) -1):
            distances.append(np.abs(self.m_peak_list[i][2] - self.m_peak_list[i + 1][2]))

        bad_list = [i for i, value in enumerate(distances)
                    if value < np.max(distances) / height_threshold]

        new_peaks = []
        i = 0
        while i < len(self.m_peak_list):

            if i not in bad_list:
                # keep important extrema
                new_peaks.append(self.m_peak_list[i])
                i += 1
            else:
                tmp_type = self.m_peak_list[i][3]
                tmp_mean_ind = self.m_peak_list[i][0]
                tmp_mean_x = self.m_peak_list[i][1]
                tmp_mean_y = self.m_peak_list[i][2]
                n = 1
                while i in bad_list:
                    tmp_mean_ind += self.m_peak_list[i+1][0]
                    tmp_mean_x += self.m_peak_list[i+1][1]
                    tmp_mean_y += self.m_peak_list[i+1][2]
                    i += 1
                    n += 1

                if n % 2 == 0:
                    i += 1
                    continue
                else:
                    new_peaks.append((int(tmp_mean_ind / n),
                                      tmp_mean_x / n,
                                      tmp_mean_y / n,
                                      tmp_type))
                    i += 1

        self.m_peak_list = new_peaks

    def calc_peak_support(self):

        for i in range(len(self.m_peak_list)):
            if self.m_peak_list[i][3] is "min":
                continue

            integral = np.sum(self.m_pdf[self.m_peak_list[i-1][0]:self.m_peak_list[i+1][0]]) \
                            * self.m_resolution

            mean = np.sum(self.m_pdf[self.m_peak_list[i - 1][0]:self.m_peak_list[i + 1][0]] * \
                          self.m_grid[self.m_peak_list[i - 1][0]:self.m_peak_list[i + 1][0]])

            mean /= integral
            mode = self.m_peak_list[i][1]

            self.m_extrema_list.append((mean* self.m_resolution,
                                        mode,
                                        integral,
                                        self.m_peak_list[i - 1][0],
                                        self.m_peak_list[i + 1][0],
                                        self.m_peak_list[i][0]))
