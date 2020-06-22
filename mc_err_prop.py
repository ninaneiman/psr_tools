import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.visualization import quantity_support
from astropy import constants as const


def get_samples_gauss(mu, sigma, nmc):
    # generate samples from normal distribution
    samples = mu + sigma * np.random.randn(nmc)
    return samples


def get_samples_two_half_gauss(mu, sigma, nmc):
    # generate samples from two half-normal distributions
    errs = np.random.randn(nmc)
    ipos = np.asarray( errs > 0 ).nonzero()
    ineg = np.asarray( errs < 0 ).nonzero()
    errs[ipos] = errs[ipos] * sigma[0].value
    errs[ineg] = errs[ineg] * np.abs(sigma[1].value)
    samples = ( errs + mu.value ) * mu.unit
    return samples


def find_out_of_bounds(samples, val_min, val_max):
    # check for any out-of-bounds samples
    i_oob = np.where( (samples < val_min) | (samples > val_max) )[0]
    n_oob = len(i_oob)
    return (i_oob, n_oob)


class parameter(object):

    def __init__(self, is_input=False,
                 label='', unit=u.dimensionless_unscaled,
                 val_min=None, val_max=None, samples=None,
                 mu=None, sigma=None,
                 func=None, inputs=None):

        self.is_input = is_input
        self.label = label
        self.unit = unit

        # check if parameter is bounded
        self.bounded = False
        if (val_min is not None) or (val_max is not None):
            self.bounded = True

            # if only one bound is given, set other to -/+ infinity
            if val_min is None:
                self.val_min = -np.Inf * self.unit
            else:
                self.val_min = val_min * self.unit
            if val_max is None:
                self.val_max = np.Inf * self.unit
            else:
                self.val_max = val_max * self.unit

        if samples is not None:
            self.samples = samples * self.unit

        if is_input:

            # for input parameters, store mean and uncertainties

            self.mu = mu * self.unit

            # interpret possible asymmetric uncertainties
            if np.size(sigma) == 2:
                # assume/sort sigma to be [positive, negative]
                if np.sign(sigma[0]) == np.sign(sigma[1]):
                    print('Assuming sigma values of %s are given as +/-'
                         % self.label )
                    sigma = np.abs(sigma) * [1., -1.]
                else:
                    sigma = np.sort(sigma)[::-1]
            self.sigma = sigma * self.unit

        else:

            # for calculated parameters, store function that calculates them
            # and the associated inputs

            self.func = func
            if not isinstance(inputs, list):
                inputs = [inputs]
            self.inputs = inputs


    def clear_mc(self):
        # clear previously generated MC samples
        if hasattr(self, 'samples'):
            del self.samples
        # for calculated parameters, also recursively clear samples of inputs
        if not self.is_input:
            for inp in self.inputs:
                inp.clear_mc()


    def gen_mc(self, nmc):
        # generate MC samples

        # prevent overwriting already generated MC samples
        if hasattr(self, 'samples'):
            return None

        if self.is_input:
            # for input parameters, generate MC samples from stat. distribution

            # determine which stat. distribution to use
            if np.size(self.sigma) == 1:
                # symmetric uncertainties with normal distribution
                sampling_method = get_samples_gauss
            elif np.size(self.sigma) == 2:
                # asymmetric uncertainties
                print('Asymmetric uncertainties of %s treated using'
                      'two half-normal distributions' % self.label)
                sampling_method = get_samples_two_half_gauss

            # generate samples
            samples = sampling_method(self.mu, self.sigma, nmc)

            if self.bounded:
                # check for any out-of-bounds samples
                (i_oob, n_oob) = find_out_of_bounds(samples,
                                                    self.val_min,
                                                    self.val_max)
                if n_oob:
                    print('Warning:\t%4d out-of-bounds samples found' % n_oob)
                # replace out-of-bounds samples until none are left
                while n_oob:
                    print('Resampling...\t', end=''),
                    samples_update = sampling_method(self.mu,
                                                     self.sigma,
                                                     n_oob)
                    samples[i_oob] = samples_update
                    (i_oob, n_oob) = find_out_of_bounds(samples,
                                                        self.val_min,
                                                        self.val_max)
                    print('%4d out-of-bounds samples left' % n_oob)

            # store samples
            self.samples = samples.to(self.unit)

        else:
            # for calculated parameters, calculate samples from input samples

            # first recursively generate MC samples of inputs
            for inp in self.inputs:
                inp.gen_mc(nmc)

            # gather input MC samples and pass to output-calculating function
            input_samples = [ inp.samples for inp in self.inputs ]
            self.samples = self.func(*input_samples).to(self.unit)


def do_mc(outp, nmc, print_result=False):

    # reset by removing existing samples
    outp.clear_mc()

    # generate MC samples
    outp.gen_mc(nmc)

    # compute quantiles and uncertainties
    q_16, q_50, q_84 = np.percentile(outp.samples, [16, 50, 84])
    q_p, q_m = q_84-q_50, q_50-q_16

    # formating to print right amount of significant digits
    scaling_exp = int( np.floor( np.log10( np.abs(q_50) ) ) )
    scaling = 10.**scaling_exp

    q_50s, q_ps, q_ms = q_50/scaling, q_p/scaling, q_m/scaling

    err_exp = np.floor( np.log10( min( np.abs( [q_ms, q_ps] ) ) ) )
    sigdig = max(-int(err_exp)+1, 1)

    results_str = '%.*f +%.*f / -%.*f' % (sigdig, q_50s,
        sigdig, q_ps, sigdig, q_ms)

    if (scaling_exp != 0):
        results_str = '( %s ) x 10^%d' % (results_str, scaling_exp)
    if print_result is True:
        print('\nResult:')
        print('%s = %s %s' % (outp.label, results_str, outp.unit) )

    return [q_50, q_p, q_m] * outp.unit


def do_mc_multi(outputs, nmc):

    if not isinstance(outputs, list):
        outputs = [outputs]

    # reset by removing existing samples
    for outp in outputs:
        outp.clear_mc()

    # generate MC samples
    for outp in outputs:
        outp.gen_mc(nmc)


def err_prop_hist(outp, bins='auto'):

    ninputs = len(outp.inputs)

    x = outp.samples.value
    x_label_string = '%s [%s]' % (outp.label, outp.unit.to_string() )

    # Set up the axes with gridspec
    fig = plt.figure(figsize=(8, 4 * (ninputs + 1)),
                     facecolor='w', edgecolor='k')
    grid = plt.GridSpec(2 * (ninputs + 1), 3, hspace=0, wspace=0)

    x_hist = fig.add_subplot(grid[:2, :-1], yticklabels=[])

    # histogram on the top
    x_hist.hist(x, bins, histtype='stepfilled',
                orientation='vertical', color='gray')
    x_hist.set_xlabel(x_label_string)
    x_hist.xaxis.tick_top()
    x_hist.xaxis.set_label_position('top')

    for iinp in range(ninputs):

        inp = outp.inputs[iinp]

        y = inp.samples.value
        y_label_string = '%s [%s]' % (inp.label, inp.unit.to_string() )

        grid_pos = 2 * (iinp + 1)
        main_ax = fig.add_subplot(grid[grid_pos:grid_pos+2, :-1], sharex=x_hist)
        y_hist = fig.add_subplot(grid[grid_pos:grid_pos+2, -1:],
                                 xticklabels=[], sharey=main_ax)

        # scatter points on the main axes
        main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2, rasterized=True)
        main_ax.set_xlabel(x_label_string)
        main_ax.set_ylabel(y_label_string)

        # histogram on the attached axes
        y_hist.hist(y, bins, histtype='stepfilled',
                    orientation='horizontal', color='gray')
        y_hist.set_ylabel(y_label_string)
        y_hist.yaxis.tick_right()
        y_hist.yaxis.set_label_position('right')


    plt.show()


def err_prop_hist_single(inp, outp, bins='auto'):

    x = outp.samples.value
    y = inp.samples.value
    x_label_string = '%s [%s]' % (outp.label, outp.unit.to_string() )
    y_label_string = '%s [%s]' % (inp.label, inp.unit.to_string() )

    # Set up the axes with gridspec
    fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
    grid = plt.GridSpec(4, 4, hspace=0, wspace=0)
    main_ax = fig.add_subplot(grid[2:, :-2])
    y_hist = fig.add_subplot(grid[2:, -2:], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[:2, :-2], yticklabels=[], sharex=main_ax)

    # scatter points on the main axes
    main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2, rasterized=True)
    main_ax.set_xlabel(x_label_string)
    main_ax.set_ylabel(y_label_string)

    # histogram on the attached axes
    x_hist.hist(x, bins, histtype='stepfilled',
                orientation='vertical', color='gray')
    x_hist.set_xlabel(x_label_string)
    x_hist.xaxis.tick_top()
    x_hist.xaxis.set_label_position('top')

    y_hist.hist(y, bins, histtype='stepfilled',
                orientation='horizontal', color='gray')
    y_hist.set_ylabel(y_label_string)
    y_hist.yaxis.tick_right()
    y_hist.yaxis.set_label_position('right')


    plt.show()
