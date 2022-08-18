from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments in plotting RL training process')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--title', default=None, type=str)
    parser.add_argument('--x_label', default='Steps', type=str)
    parser.add_argument('--y_label', default='Success Rate', choices=['Reward', 'Success Rate', 'Goal'], type=str)
    parser.add_argument('--save_path', default=None, type=bool)

    config = parser.parse_args()
    return config


def xy_success_rate_fn(r):
    x = np.cumsum(r.monitor.l)
    y = pu.smooth(r.monitor.is_success, radius=50)

    if MAX_STEPS < 0 or x.max() < MAX_STEPS:
        return x, y

    x_within_range = x[x<MAX_STEPS]
    x = np.append(x_within_range, [MAX_STEPS])
    y = y[:x.shape[0]]

    return x, y


def xy_goal_fn(r):
    x = np.cumsum(r.monitor.l)
    y = pu.smooth(r.monitor.goal, radius=50)

    if MAX_STEPS < 0 or x.max() < MAX_STEPS:
        return x, y * -1

    x_within_range = x[x<MAX_STEPS]
    x = np.append(x_within_range, [MAX_STEPS])
    y = y[:x.shape[0]]

    return x, y * -1


def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))


    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0 # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(- 1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    return xnews, ys, count_ys


def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    xs, ys1, count_ys1 = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0)
    _,  ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold=0)
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys, count_ys


def plot_results(
    allresults, *,
    xy_fn=pu.default_xy_fn,
    split_fn=pu.default_split_fn,
    group_fn=pu.default_split_fn,
    average_group=False,
    shaded_std=True,
    shaded_err=True,
    figsize=None,
    legend_outside=False,
    resample=0,
    smooth_step=1.0,
    tiling='vertical',
    xlabel=None,
    ylabel=None
):

    if split_fn is None:
        split_fn = lambda _: ''
    if group_fn is None:
        group_fn = lambda _: ''
    sk2r = defaultdict(list) # splitkey2results
    for result in allresults:
        splitkey = split_fn(result)
        sk2r[splitkey].append(result)
    assert len(sk2r) > 0
    assert isinstance(resample, int), "0: don't resample. <integer>: that many samples"

    if tiling == 'vertical' or tiling is None:
        nrows = len(sk2r)
        ncols = 1
    elif tiling == 'horizontal':
        ncols = len(sk2r)
        nrows = 1
    elif tiling == 'symmetric':
        import math
        N = len(sk2r)
        largest_divisor = 1
        for i in range(1, int(math.sqrt(N))+1):
            if N % i == 0:
                largest_divisor = i
        ncols = largest_divisor
        nrows = N // ncols
    figsize = figsize or (8 * ncols, 4.8 * nrows)

    f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize)

    groups = list(set(group_fn(result) for result in allresults))

    default_samples = 512
    if average_group:
        resample = resample or default_samples

    for (isplit, sk) in enumerate(sorted(sk2r.keys())):
        g2l = {}
        g2c = defaultdict(int)
        sresults = sk2r[sk]
        gresults = defaultdict(list)
        idx_row = isplit // ncols
        idx_col = isplit % ncols
        ax = axarr[idx_row][idx_col]
        for result in sresults:
            group = group_fn(result)
            g2c[group] += 1
            x, y = xy_fn(result)
            if x is None: x = np.arange(len(y))
            x, y = map(np.asarray, (x, y))
            if average_group:
                gresults[group].append((x,y))
            else:
                if resample:
                    x, y, counts = symmetric_ema(x, y, x[0], x[-1], resample, decay_steps=smooth_step)
                l, = ax.plot(x, y, color=COLORS[groups.index(group) % len(COLORS)])
                g2l[group] = l
        if average_group:
            used_color_index = []
            for group in sorted(groups):
                xys = gresults[group]
                if not any(xys):
                    continue

                # Assign each group a fixed color
                def group2color_idx(group):
                    color_idx = 0
                    group = group.lower()
                    for letter in group:
                        color_idx += ord(letter) - ord('a')
                    return color_idx
                color_idx = group2color_idx(group) % len(COLORS)
                while color_idx in used_color_index:
                    color_idx = (color_idx + 1) % len(COLORS)
                used_color_index.append(color_idx)
                color = COLORS[color_idx]

                origxs = [xy[0] for xy in xys]
                minxlen = min(map(len, origxs))
                def allequal(qs):
                    return all((q==qs[0]).all() for q in qs[1:])
                if resample:
                    low  = max(x[0] for x in origxs)
                    high = min(x[-1] for x in origxs)
                    usex = np.linspace(low, high, resample)
                    ys = []
                    for (x, y) in xys:
                        ys.append(symmetric_ema(x, y, low, high, resample, decay_steps=smooth_step)[1])
                else:
                    assert allequal([x[:minxlen] for x in origxs]),\
                        'If you want to average unevenly sampled data, set resample=<number of samples you want>'
                    usex = origxs[0]
                    ys = [xy[1][:minxlen] for xy in xys]
                y_maxs = []
                for y in ys:
                    y_maxs.append(y.max())
                y_maxs = np.array(y_maxs)
                ymean = np.mean(ys, axis=0)
                ystd = np.std(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                l, = axarr[idx_row][idx_col].plot(usex, ymean, color=color, label=group)
                g2l[group] = l
                if shaded_err:
                    ax.fill_between(usex, ymean - ystderr, ymean + ystderr, color=color, alpha=.3)
                if shaded_std:
                    ax.fill_between(usex, ymean - ystd,    ymean + ystd,    color=color, alpha=.2)
                print('Group: {}\tMax: {:.6f}\tMax_std: {:.6f}\tFinal: {:.6f}\tFinal_std: {:.6f}'.format(
                    group, y_maxs.mean(), y_maxs.std() / np.sqrt(len(y_maxs)), ymean.tolist()[-1], ystderr.tolist()[-1]
                ))

        plt.tight_layout()
        ax.set_title(sk)

        y_font_dict = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        x_font_dict = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

        plt.legend(bbox_to_anchor=(1,1))

        if xlabel is not None:
            for ax in axarr[-1]:
                plt.sca(ax)
                plt.xlabel(xlabel, fontdict=x_font_dict)
        # add ylabels, but only to left column
        if ylabel is not None:
            for ax in axarr[:,0]:
                plt.sca(ax)
                plt.ylabel(ylabel, fontdict=y_font_dict)

    return f, axarr, (usex, ymean, ystd)


MAX_STEPS = -1
# COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
#          'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
#          'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue']
COLORS = ['#001871', '#ff585d', '#ffb549', '#41b6e6']


if __name__ == '__main__':
    config = parse_args()
    log_dir = config.log_dir

    # The gray background with axis meshes
    # seaborn.set()

    results = pu.load_results(log_dir, enable_progress=False, verbose=True)

    if config.title == None:
        config.title = results[0].dirname.split('/')[-2].split('_')[0]

    x_label = config.x_label
    y_label = config.y_label

    if y_label == 'Reward':
        xy_fn = pu.default_xy_fn
    elif y_label == 'Goal':
        xy_fn = xy_goal_fn
    else:
        xy_fn = xy_success_rate_fn

    group_fn = lambda result: result.dirname.split('/')[-2].split('_')[1]

    f, _, data = plot_results(
        results,
        xy_fn=xy_fn,
        split_fn=lambda _: '',
        group_fn=group_fn,
        average_group=True,
        shaded_std=False,
        shaded_err=True,
        xlabel=x_label,
        ylabel=y_label,
        legend_outside=False
    )

    font_dict = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

    # plt.legend(prop={'family': 'Times New Roman', 'weight': 'normal', 'size': 12})
    plt.title(config.title, fontdict=font_dict)
    plt.tight_layout()

    if config.save_path is not None:
        plt.savefig(config.save_path)
    plt.show()
