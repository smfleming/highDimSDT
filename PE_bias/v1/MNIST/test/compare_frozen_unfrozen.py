import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import ttest_rel, ttest_ind
import argparse


def plot(args):
    all_frozen_test_acc = []
    all_frozen_test_conf = []
    all_unfrozen_test_acc = []
    all_unfrozen_test_conf = []
    for r in range(args.N_runs):
        # Run directory
        run_dir = f'./classes{args.n_classes}/run' + str(r + 1) + '/'
        run_dir_frozen = f'./classes10_{args.n_classes}/run' + str(r + 1) + '/'
        # Load test results
        test_results = np.load(run_dir + 'test_results.npz')
        test_results_frozen = np.load(run_dir_frozen + 'test_results.npz')
        signal_test_vals = test_results['signal_test_vals']
        noise_test_vals = test_results['noise_test_vals']
        unfrozen_test_acc = test_results['all_test_acc'] / 100
        unfrozen_test_conf = test_results['all_test_conf'] / 100
        frozen_test_acc = test_results_frozen['all_test_acc'] / 100
        frozen_test_conf = test_results_frozen['all_test_conf'] / 100
        # Collect results for all runs
        all_frozen_test_acc.append(frozen_test_acc)
        all_frozen_test_conf.append(frozen_test_conf)
        all_unfrozen_test_acc.append(unfrozen_test_acc)
        all_unfrozen_test_conf.append(unfrozen_test_conf)
    # Convert to arrays
    all_frozen_test_acc = np.array(all_frozen_test_acc)
    all_frozen_test_conf = np.array(all_frozen_test_conf)
    all_unfrozen_test_acc = np.array(all_unfrozen_test_acc)
    all_unfrozen_test_conf = np.array(all_unfrozen_test_conf)
    # Summary statistics
    all_frozen_test_acc_mean = all_frozen_test_acc.mean(0)
    all_frozen_test_acc_se = sem(all_frozen_test_acc, 0)
    all_frozen_test_conf_mean = all_frozen_test_conf.mean(0)
    all_frozen_test_conf_se = sem(all_frozen_test_conf, 0)
    all_unfrozen_test_acc_mean = all_unfrozen_test_acc.mean(0)
    all_unfrozen_test_acc_se = sem(all_unfrozen_test_acc, 0)
    all_unfrozen_test_conf_mean = all_unfrozen_test_conf.mean(0)
    all_unfrozen_test_conf_se = sem(all_unfrozen_test_conf, 0)
    # Find PE conditions
    target_acc = 0.55
    # Low PE
    frozen_low_PE_ind = np.abs(all_frozen_test_acc_mean[0, :] - target_acc).argmin()
    frozen_low_PE_signal = signal_test_vals[frozen_low_PE_ind]
    frozen_low_PE_test_acc = all_frozen_test_acc[:, 0, frozen_low_PE_ind]
    frozen_low_PE_test_acc_mean = all_frozen_test_acc_mean[0, frozen_low_PE_ind]
    frozen_low_PE_test_acc_se = all_frozen_test_acc_se[0, frozen_low_PE_ind]
    frozen_low_PE_test_conf = all_frozen_test_conf[:, 0, frozen_low_PE_ind]
    frozen_low_PE_test_conf_mean = all_frozen_test_conf_mean[0, frozen_low_PE_ind]
    frozen_low_PE_test_conf_se = all_frozen_test_conf_se[0, frozen_low_PE_ind]
    
    unfrozen_low_PE_ind = np.abs(all_unfrozen_test_acc_mean[0, :] - target_acc).argmin()
    unfrozen_low_PE_signal = signal_test_vals[unfrozen_low_PE_ind]
    unfrozen_low_PE_test_acc = all_unfrozen_test_acc[:, 0, unfrozen_low_PE_ind]
    unfrozen_low_PE_test_acc_mean = all_unfrozen_test_acc_mean[0, unfrozen_low_PE_ind]
    unfrozen_low_PE_test_acc_se = all_unfrozen_test_acc_se[0, unfrozen_low_PE_ind]
    unfrozen_low_PE_test_conf = all_unfrozen_test_conf[:, 0, unfrozen_low_PE_ind]
    unfrozen_low_PE_test_conf_mean = all_unfrozen_test_conf_mean[0, unfrozen_low_PE_ind]
    unfrozen_low_PE_test_conf_se = all_unfrozen_test_conf_se[0, unfrozen_low_PE_ind]
    # High PE
    frozen_high_PE_ind = np.abs(all_frozen_test_acc_mean[1, :] - target_acc).argmin()
    frozen_high_PE_signal = signal_test_vals[frozen_high_PE_ind]
    frozen_high_PE_test_acc = all_frozen_test_acc[:, 1, frozen_high_PE_ind]
    frozen_high_PE_test_acc_mean = all_frozen_test_acc_mean[1, frozen_high_PE_ind]
    frozen_high_PE_test_acc_se = all_frozen_test_acc_se[1, frozen_high_PE_ind]
    frozen_high_PE_test_conf = all_frozen_test_conf[:, 1, frozen_high_PE_ind]
    frozen_high_PE_test_conf_mean = all_frozen_test_conf_mean[1, frozen_high_PE_ind]
    frozen_high_PE_test_conf_se = all_frozen_test_conf_se[1, frozen_high_PE_ind]
    
    unfrozen_high_PE_ind = np.abs(all_unfrozen_test_acc_mean[1, :] - target_acc).argmin()
    unfrozen_high_PE_signal = signal_test_vals[unfrozen_high_PE_ind]
    unfrozen_high_PE_test_acc = all_unfrozen_test_acc[:, 1, unfrozen_high_PE_ind]
    unfrozen_high_PE_test_acc_mean = all_unfrozen_test_acc_mean[1, unfrozen_high_PE_ind]
    unfrozen_high_PE_test_acc_se = all_unfrozen_test_acc_se[1, unfrozen_high_PE_ind]
    unfrozen_high_PE_test_conf = all_unfrozen_test_conf[:, 1, unfrozen_high_PE_ind]
    unfrozen_high_PE_test_conf_mean = all_unfrozen_test_conf_mean[1, unfrozen_high_PE_ind]
    unfrozen_high_PE_test_conf_se = all_unfrozen_test_conf_se[1, unfrozen_high_PE_ind]

    # Stats
    # Open file
    stats_fname = f'./PE_stats_comparison_{args.n_classes}.txt'
    fid = open(stats_fname, 'w')
    # Noiseless test accuracy

    # Low and high PE conditions
    fid.write('Low PE frozen condition: sigma = ' + str(noise_test_vals[0]) + ', mu = ' + str(frozen_low_PE_signal) + '\n')
    fid.write('High PE frozen condition: sigma = ' + str(noise_test_vals[1]) + ', mu = ' + str(frozen_high_PE_signal) + '\n')
    fid.write(
        'Low PE unfrozen condition: sigma = ' + str(noise_test_vals[0]) + ', mu = ' + str(unfrozen_low_PE_signal) + '\n')
    fid.write(
        'High PE unfrozen condition: sigma = ' + str(noise_test_vals[1]) + ', mu = ' + str(unfrozen_high_PE_signal) + '\n')
    # Accuracy difference
    frozen_acc_diff = frozen_high_PE_test_acc_mean - frozen_low_PE_test_acc_mean
    unfrozen_acc_diff = unfrozen_high_PE_test_acc_mean - unfrozen_low_PE_test_acc_mean
    fid.write('Acc. diff. frozen = ' + str(frozen_acc_diff) + '\n')
    fid.write('Acc. diff. unfrozen = ' + str(unfrozen_acc_diff) + '\n')

    # Confidence difference frozen
    frozen_conf_diff = frozen_high_PE_test_conf_mean - frozen_low_PE_test_conf_mean
    fid.write('Conf. diff. frozen = ' + str(frozen_conf_diff) + '\n')
    # Confidence difference unfrozen
    unfrozen_conf_diff = unfrozen_high_PE_test_conf_mean - unfrozen_low_PE_test_conf_mean
    fid.write('Conf. diff. unfrozen = ' + str(unfrozen_conf_diff) + '\n')
    # Confidence difference between
    conf_diff_between = unfrozen_conf_diff - frozen_conf_diff
    fid.write('Conf. diff. between = ' + str(conf_diff_between) + '\n')
    # T-test for confidence
    conf_t, conf_p = ttest_ind(unfrozen_high_PE_test_conf- unfrozen_low_PE_test_conf, frozen_high_PE_test_conf-frozen_low_PE_test_conf)
    fid.write('t-test: t = ' + str(conf_t) + ', p = ' + str(conf_p))
    # Close file
    fid.close()
    # Significance symbols
    # if conf_p >= 0.05: conf_p_symb = 'ns'
    # if conf_p < 0.05: conf_p_symb = '*'
    # if conf_p < 0.01: conf_p_symb = '**'
    # if conf_p < 0.001: conf_p_symb = '***'
    # if conf_p < 0.0001: conf_p_symb = '****'
    # if acc_p >= 0.05: acc_p_symb = 'ns'
    # if acc_p < 0.05: acc_p_symb = '*'
    # if acc_p < 0.01: acc_p_symb = '**'
    # if acc_p < 0.001: acc_p_symb = '***'
    # if acc_p < 0.0001: acc_p_symb = '****'
    #
    # # Font sizes
    # axis_label_font_size = 22
    # tick_font_size = 20
    # significance_font_size = 20
    # title_font_size = 30
    #
    # # Combined plot
    # ax1 = plt.subplot(111)
    # ax1.bar([0, 1], [low_PE_test_acc_mean, high_PE_test_acc_mean], yerr=[low_PE_test_acc_se, high_PE_test_acc_se],
    #         width=0.8, color='gray')
    # ax1.set_ylabel('P(Correct)', fontsize=axis_label_font_size)
    # plt.ylim([0.45, 0.8])
    # plt.xticks([0, 1, 2.5, 3.5], ['Low', 'High', 'Low', 'High'], fontsize=tick_font_size)
    # ax1.set_xlabel('Positive evidence', fontsize=axis_label_font_size)
    # plt.yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], ['0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75'],
    #            fontsize=tick_font_size)
    # ax1.spines['top'].set_visible(False)
    # ax2 = ax1.twinx()
    # ax2.bar([2.5, 3.5], [low_PE_test_conf_mean, high_PE_test_conf_mean],
    #         yerr=[low_PE_test_conf_se, high_PE_test_conf_se], width=0.8, color='black')
    # ax2.set_ylabel('Confidence', fontsize=axis_label_font_size)
    # plt.ylim([0.45, 0.8])
    # plt.yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], ['0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75'],
    #            fontsize=tick_font_size)
    # ax2.spines['top'].set_visible(False)
    # # Significance
    # max_y_val = np.max([low_PE_test_acc_mean + low_PE_test_acc_se, high_PE_test_acc_mean + high_PE_test_acc_se])
    # y_start = max_y_val + 0.01
    # y_end = max_y_val + 0.015
    # ax1.plot([0, 0, 1, 1], [y_start, y_end, y_end, y_start], color='black')
    # ax1.text(0.5, y_end + 0.005, acc_p_symb, fontsize=significance_font_size, horizontalalignment='center')
    # max_y_val = np.max([low_PE_test_conf_mean + low_PE_test_conf_se, high_PE_test_conf_mean + high_PE_test_conf_se])
    # y_start = max_y_val + 0.01
    # y_end = max_y_val + 0.015
    # ax2.plot([2.5, 2.5, 3.5, 3.5], [y_start, y_end, y_end, y_start], color='black')
    # ax2.text(3, y_end + 0.005, conf_p_symb, fontsize=significance_font_size, horizontalalignment='center')
    # # Title
    # plt.title('MNIST', fontsize=title_font_size)
    # # Save plot
    # plot_fname = f'./PE_bias_MNIST{args.n_classes}.png'
    # plt.savefig(plot_fname, bbox_inches='tight', dpi=300)
    # plt.close()
    #

def main():
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_runs', type=int, default=100)
    parser.add_argument('--n_classes', type=int, default=10)
    args = parser.parse_args()

    # Plot
    plot(args)


if __name__ == '__main__':
    main()