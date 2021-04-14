from argparse import ArgumentParser
from run import get_x_y
import matplotlib.pyplot as plt
import math
import numpy as np
import copy

# Use a function for the X axis
def area(config, results):
    # In this example, I'm setting the total area in the X axis
    return config["width"] * config["height"]


# Another function for the X axis
def n_states(config, results):
    # In this example, I'm using the number of states
    return results["Earth Observation Ground MDP"]["Number of States"]

# 
def reward_density(config, results):
    # Number of POIs divided by number of locations (all weather configs have reward at poi losc)
    return float(results["Earth Observation Ground MDP"]["POIs"]) / float(results["Earth Observation Ground MDP"]["Width"] * results["Earth Observation Ground MDP"]["Height"])

# Use a function for the Y axis
def cumulative_sketch_refine_time(config, results):
    return sum(s.get("Policy Sketch-Refine Time", 0) for s in results["Simulation"]["Steps"])

# Use a function for the Y axis
def abstraction_time_cumulative_sketch_refine_time(config, results):
    return results["Abstract MDP"]["Abstraction Time"] + sum(s.get("Policy Sketch-Refine Time", 0) for s in results["Simulation"]["Steps"])

# Use a function for the Y axis
def abstraction_time(config, results):
    return results["Abstract MDP"]["Abstraction Time"]

# Use a function for the Y axis
def ground_mdp_solve_time(config, results):
    return results["Earth Observation Ground MDP"]["Solving Time"]


# Another function for the Y axis
def cumulative_reward(config, results):
    return results["Simulation"]["Cumulative Reward"]

# Use a function for the Y axis
def percent_states_expanded(config, results):
    num_abstract_states = results["Abstract MDP"]["Abstraction Number of States"]
    list_abstract_states = [s.get("Current Abstract State", 0) for s in results["Simulation"]["Steps"]]
    set_abstract_states = set(list_abstract_states)
    return float(len(set_abstract_states)) / float(num_abstract_states)

def calculate_confidence_interval(mean, conf):
    upper_bound = []
    lower_bound = []
    for i in range(len(mean)):
        lower_bound.append(mean[i] - conf[i])
        upper_bound.append(mean[i] + conf[i])

    return upper_bound, lower_bound

def calculate_statistics(independent, dependent):
    independent_var = []
    dependent_var_mean = []
    dependent_var_var = []
    conf_interval_95 = []
    
    if len(independent) != len(dependent):
        print("Mismatched data vectors")
        return independent_var, dependent_var_mean, dependent_var_var

    totals = {}
    counts = {}
    spread = {}
    for key in independent:
        totals[key] = 0
        counts[key] = 0
        spread[key] = []
    
    for i in range(len(independent)):
        totals[independent[i]] += dependent[i]
        counts[independent[i]] += 1
        tmp = spread[independent[i]]
        tmp.append(dependent[i])
        spread[independent[i]] = tmp

    means = {}
    for key in totals:
        means[key] = totals[key] / float(counts[key])
 
    variances = {}
    for key in spread:
        total_deviation = 0
        entry = spread[key]
        for i in range(len(entry)):
            total_deviation += pow((entry[i] - means[key]), 2)
        variances[key] = total_deviation / float(counts[key])

    for key in means:
        independent_var.append(key)
        dependent_var_mean.append(means[key])
        dependent_var_var.append(variances[key])
        conf_interval_95.append(2.0 * math.sqrt(variances[key]) / math.sqrt(float(counts[key])))

    return independent_var, dependent_var_mean, dependent_var_var, conf_interval_95

def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("baseline_config_file")
    arg_parser.add_argument("config_file_0")
    arg_parser.add_argument("config_file_1")
    arg_parser.add_argument("config_file_2")
    arg_parser.add_argument("config_file_a")
    arg_parser.add_argument("data_dir")
    
    args = arg_parser.parse_args()
    baseline_config_file = args.baseline_config_file
    config_file_0 = args.config_file_0
    config_file_1 = args.config_file_1
    config_file_2 = args.config_file_2
    config_file_a = args.config_file_a
    data_dir = args.data_dir

# TODO: update plots with final experiments / red dashed line of hypothetical MDP times



#TODO: Things to do for camera-ready:
#TODO: are we logging PAMDP construction time as well??
#TODO: hook up AMDP experiments
#TODO: run AMDP experiments
#TODO: run final PAMDP experiments
#TODO: run MDP version of big problems
#      More experiments - 
#          possibly an analysis of the types of paths taken 
#          statistics of actual wait times for actions
#          does homotopic class of trajectory depend on method?
#          how do different abstraction schemes change performance?
#    Intro could emphasize experimental results more, and possibly current text can be shortened to accommodate
#    To accommodate more empirical analysis, probably section 5 can be shortened even further
#    To accommodate more empirical analysis, probably section 6 can be shortened slightly
#    Can we investigate replacing proof sketches with proper proofs? perhaps move them to an appendix if it gets ugly? 


##### Some other submission...(s) #####
#TODO: Plot some freaking histograms dude.... for:
#    run final 3 problem sizes...
#    chache hits / misses
#    %states visited
#    ????
#TODO: add negative reward for moving north and south?
#TODO: experiment with transition function perturbations?
#TODO: one thing I'm noticing is that 60%-80% of the time spent on PAMDPs is spent building the PAMDPs... 
#      I'm wondering if there is additional caching we can do to speed that up. For example, many PAMDPs are 
#      are identical except for the reward of one of the abstract states. In this case we could load the already constructed 
#      PAMDP basically instantly and just change one number....


    plt.rcParams["font.family"] = "FreeSerif"
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 18

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    plt.rc('legend', fontsize=16)   # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    """
    # time vs. number of states

    b_x, b_y = get_x_y(data_dir, baseline_config_file, x_func=n_states, y_func=ground_mdp_solve_time, sort=True)
    x0, y0 = get_x_y(data_dir, config_file_0, x_func=n_states, y_func=cumulative_sketch_refine_time, sort=True)
    x1, y1 = get_x_y(data_dir, config_file_1, x_func=n_states, y_func=cumulative_sketch_refine_time, sort=True)
    x2, y2 = get_x_y(data_dir, config_file_2, x_func=n_states, y_func=cumulative_sketch_refine_time, sort=True)
    #x0, y0 = get_x_y(data_dir, config_file_0, x_func=n_states, y_func=abstraction_time_cumulative_sketch_refine_time, sort=True)
    #x1, y1 = get_x_y(data_dir, config_file_1, x_func=n_states, y_func=abstraction_time_cumulative_sketch_refine_time, sort=True)
    #x2, y2 = get_x_y(data_dir, config_file_2, x_func=n_states, y_func=abstraction_time_cumulative_sketch_refine_time, sort=True)
    x0_abstract, y0_abstract = get_x_y(data_dir, config_file_0, x_func=n_states, y_func=abstraction_time, sort=True)

    # calculate mean and variance 
    b_x, b_y_mean, b_y_var, b_conf_95 = calculate_statistics(b_x, b_y)
    x0, y0_mean, y0_var, conf0_95 = calculate_statistics(x0, y0)
    x1, y1_mean, y1_var, conf1_95 = calculate_statistics(x1, y1)
    x2, y2_mean, y2_var, conf2_95 = calculate_statistics(x2, y2)
    
    x0_abstract, y0_abstract_mean, _, _ = calculate_statistics(x0_abstract, y0_abstract)
    
    # calculate confidence intervals 
    b_ub, b_lb = calculate_confidence_interval(b_y_mean, b_conf_95)
    ub0, lb0 = calculate_confidence_interval(y0_mean, conf0_95)
    ub1, lb1 = calculate_confidence_interval(y1_mean, conf1_95)
    ub2, lb2 = calculate_confidence_interval(y2_mean, conf2_95)
 

    # since we have some single samples...
    ub1[-1] = 1.05 * y1_mean[-1]
    ub1[-2] = 1.05 * y1_mean[-2]
    ub1[-3] = 1.05 * y1_mean[-3]
    ub2[-1] = 1.05 * y2_mean[-1]
    ub2[-2] = 1.05 * y2_mean[-2]
    ub2[-3] = 1.05 * y2_mean[-3]
    lb1[-1] = 0.95 * y1_mean[-1]
    lb1[-2] = 0.95 * y1_mean[-2]
    lb1[-3] = 0.95 * y1_mean[-3]
    lb2[-1] = 0.95 * y2_mean[-1]
    lb2[-2] = 0.95 * y2_mean[-2]
    lb2[-3] = 0.95 * y2_mean[-3]

    # since we're missing some ground MDPs
    b_x_est = copy.deepcopy(b_x)
    b_y_mean_est = copy.deepcopy(b_y_mean)
    b_x_est.append(27648)
    b_x_est.append(73728)
    b_x_est.append(110592)
    b_y_mean_est.append(24200)
    b_y_mean_est.append(170000)
    b_y_mean_est.append(380000)

    figure = plt.figure(figsize=(7, 5))

    # Plot
    plt.plot(b_x, b_y_mean, 'r', label='Ground MDP')
    plt.plot(x0, y0_mean, label='Naive Strategy')
    plt.plot(x1, y1_mean, label='Greedy Strategy')
    plt.plot(x2, y2_mean, label='Proactive Strategy')
    plt.plot(x0_abstract, y0_abstract_mean, 'k--', label='Abstract MDP')
    plt.plot(b_x_est, b_y_mean_est, 'r:', label='Estimated MDP')
    plt.fill_between(b_x, b_lb, b_ub, alpha=0.3, color='r')
    plt.fill_between(x0, lb0, ub0, alpha=0.3)
    plt.fill_between(x1, lb1, ub1, alpha=0.3)
    plt.fill_between(x2, lb2, ub2, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    # plt.title('Time to Compute Policy vs. Number of States')
    #plt.xlabel('Ground State Space Size', fontsize=20, fontweight='ultralight')
    #plt.ylabel('Cumulative Planning Time [sec] ', fontsize=20, fontweight='ultralight')
    plt.xlabel('Ground State Space Size', fontsize=20)
    plt.ylabel('Cumulative Planning Time [sec] ', fontsize=20)
    #plt.legend(ncol=2, loc='lower right', handletextpad=0.3, columnspacing=0.6, labelspacing=0.15)
    plt.legend(ncol=2, loc='upper left', handletextpad=0.3, columnspacing=0.6, labelspacing=0.15)
    plt.tight_layout()
    plt.margins(x=0.01, y=0.05)
   
    FILENAME = 'time_vs_size_log_log.pdf'
    figure.savefig(FILENAME, bbox_inches="tight")
    plt.show()
    """

    # cum reward ratio vs. number of states

    b_x, b_y = get_x_y(data_dir, baseline_config_file, x_func=n_states, y_func=cumulative_reward, sort=True)
    x0, y0 = get_x_y(data_dir, config_file_0, x_func=n_states, y_func=cumulative_reward, sort=True)
    x1, y1 = get_x_y(data_dir, config_file_1, x_func=n_states, y_func=cumulative_reward, sort=True)
    x2, y2 = get_x_y(data_dir, config_file_2, x_func=n_states, y_func=cumulative_reward, sort=True)
    xa, ya = get_x_y(data_dir, config_file_a, x_func=n_states, y_func=cumulative_reward, sort=True)

    y0 = [i / j for i, j in zip(y0, b_y)]
    y1 = [i / j for i, j in zip(y1, b_y)]
    y2 = [i / j for i, j in zip(y2, b_y)]
    ya = [i / j for i, j in zip(ya, b_y)]
    b_y = [1.0 for _ in range(len(y0))]
    
    """
    figure = plt.figure(figsize=(7, 3))
    # Plot a histogram
    bins = np.arange(0.4, 1.01, 0.01)
    plt.hist(y0, bins=bins, density=True, alpha=0.5, histtype='stepfilled', cumulative=True, label='Naive Strategy')
    plt.hist(y1, bins=bins, density=True, alpha=0.5, histtype='stepfilled', cumulative=True, label='Greedy Strategy')
    plt.hist(y2, bins=bins, density=True, alpha=0.5, histtype='stepfilled', cumulative=True, label='Proactive Strategy')
    #plt.title('Cumulative Reward Ratio Frequencies')
    plt.xlabel('Cumulative Reward Ratio', fontsize=17)
    plt.ylabel('Cumulative Frequency', fontsize=17)
    plt.legend(loc='upper left', handletextpad=0.3, columnspacing=0.6, labelspacing=0.15)
    plt.tight_layout()
    plt.margins(x=0.0, y=0.05)
    FILENAME = 'histogram_of_reward_ratio.pdf'
    figure.savefig(FILENAME, bbox_inches="tight")
    plt.show()
    """
    
    # calculate mean and variance 
    b_x, b_y_mean, b_y_var, b_conf_95 = calculate_statistics(b_x, b_y)
    x0, y0_mean, y0_var, conf0_95 = calculate_statistics(x0, y0)
    x1, y1_mean, y1_var, conf1_95 = calculate_statistics(x1, y1)
    x2, y2_mean, y2_var, conf2_95 = calculate_statistics(x2, y2)
    xa, ya_mean, ya_var, confa_95 = calculate_statistics(xa, ya)
    
    # calculate confidence intervals 
    b_ub, b_lb = calculate_confidence_interval(b_y_mean, b_conf_95)
    ub0, lb0 = calculate_confidence_interval(y0_mean, conf0_95)
    ub1, lb1 = calculate_confidence_interval(y1_mean, conf1_95)
    ub2, lb2 = calculate_confidence_interval(y2_mean, conf2_95)
    uba, lba = calculate_confidence_interval(ya_mean, confa_95)

    figure = plt.figure(figsize=(7, 5))
    # Plot
    plt.plot(b_x, b_y_mean, 'r', label='Ground MDP')
    plt.plot(x0, y0_mean, label='Naive Strategy')
    plt.plot(x1, y1_mean, label='Greedy Strategy')
    plt.plot(x2, y2_mean, label='Proactive Strategy')
    plt.plot(xa, ya_mean, 'k--', label='Abstract MDP')
    plt.fill_between(x0, lb0, ub0, alpha=0.3)
    plt.fill_between(x1, lb1, ub1, alpha=0.3)
    plt.fill_between(x2, lb2, ub2, alpha=0.3)
    plt.fill_between(xa, lba, uba, facecolor='black', alpha=0.3)
    #plt.ylim(0.6, 1.03)
    plt.ylim(0.05, 1.03)
    #plt.xscale('log')
    #plt.title('Cumulative Reward Ratio vs. Number of States')
    plt.xlabel('Ground State Space Size', fontsize=20)
    plt.ylabel('Cumulative Reward Ratio', fontsize=20)
    plt.legend(ncol=2, loc='center right', handletextpad=0.3, columnspacing=0.6, labelspacing=0.15, bbox_to_anchor=[1.0, 0.58])
    plt.tight_layout()
    plt.margins(x=0.01, y=0.05)
    FILENAME = 'reward_ratio_vs_size.pdf'
    figure.savefig(FILENAME, bbox_inches="tight")
    plt.show()

    # cum reward ratio vs. reward density
    
    b_x, b_y = get_x_y(data_dir, baseline_config_file, x_func=reward_density, y_func=cumulative_reward, sort=True)
    x0, y0 = get_x_y(data_dir, config_file_0, x_func=reward_density, y_func=cumulative_reward, sort=True)
    x1, y1 = get_x_y(data_dir, config_file_1, x_func=reward_density, y_func=cumulative_reward, sort=True)
    x2, y2 = get_x_y(data_dir, config_file_2, x_func=reward_density, y_func=cumulative_reward, sort=True)
    xa, ya = get_x_y(data_dir, config_file_a, x_func=reward_density, y_func=cumulative_reward, sort=True)

    y0 = [i / j for i, j in zip(y0, b_y)]
    y1 = [i / j for i, j in zip(y1, b_y)]
    y2 = [i / j for i, j in zip(y2, b_y)]
    ya = [i / j for i, j in zip(ya, b_y)]
    b_y = [1.0 for _ in range(len(y0))]

    # calculate mean and variance 
    b_x, b_y_mean, b_y_var, b_conf_95 = calculate_statistics(b_x, b_y)
    x0, y0_mean, y0_var, conf0_95 = calculate_statistics(x0, y0)
    x1, y1_mean, y1_var, conf1_95 = calculate_statistics(x1, y1)
    x2, y2_mean, y2_var, conf2_95 = calculate_statistics(x2, y2)
    xa, ya_mean, ya_var, confa_95 = calculate_statistics(xa, ya)
    
    # calculate confidence intervals 
    b_ub, b_lb = calculate_confidence_interval(b_y_mean, b_conf_95)
    ub0, lb0 = calculate_confidence_interval(y0_mean, conf0_95)
    ub1, lb1 = calculate_confidence_interval(y1_mean, conf1_95)
    ub2, lb2 = calculate_confidence_interval(y2_mean, conf2_95)
    uba, lba = calculate_confidence_interval(ya_mean, confa_95)

    figure = plt.figure(figsize=(7, 5))
    # Plot
    plt.plot(b_x, b_y_mean, 'r', label='Ground MDP')
    plt.plot(x0, y0_mean, label='Naive Strategy')
    plt.plot(x1, y1_mean, label='Greedy Strategy')
    plt.plot(x2, y2_mean, label='Proactive Strategy')
    plt.plot(xa, ya_mean, 'k--',  label='Abstract MDP')
    plt.fill_between(x0, lb0, ub0, alpha=0.3)
    plt.fill_between(x1, lb1, ub1, alpha=0.3)
    plt.fill_between(x2, lb2, ub2, alpha=0.3)
    plt.fill_between(xa, lba, uba, facecolor='black', alpha=0.3)
    #plt.ylim(0.6, 1.03)
    plt.ylim(0.12, 1.03)
    #plt.title('Cumulative Reward Ratio vs. Reward Sparsity')
    plt.xlabel('Reward Density', fontsize=20)
    plt.ylabel('Cumulative Reward Ratio', fontsize=20)
    plt.legend(ncol=2, loc='center right', handletextpad=0.3, columnspacing=0.6, labelspacing=0.15, bbox_to_anchor=[1.0, 0.58])
    plt.tight_layout()
    plt.margins(x=0.01, y=0.05)
    FILENAME = 'reward_ratio_vs_reward_density.pdf'
    figure.savefig(FILENAME, bbox_inches="tight")
    plt.show()

    



    """
    # percentage of states expanded vs. reward density
    
    x0, y0 = get_x_y(data_dir, config_file_0, x_func=reward_density, y_func=percent_states_expanded, sort=True)
    
    # calculate mean and variance 
    x0, y0_mean, y0_var, conf0_95 = calculate_statistics(x0, y0)
    
    # calculate confidence intervals 
    ub0, lb0 = calculate_confidence_interval(y0_mean, conf0_95)

    # Plot
    plt.plot(x0, y0_mean, 'b')
    #plt.fill_between(x0, lb0, ub0, alpha=0.5, color='b', label='')
    plt.fill_between(x0, lb0, ub0, alpha=0.5, color='b')
    #plt.ylim(0.0, 1.1)
    plt.title('Fraction of Abstract States Expanded vs. Reward Sparsity')
    plt.xlabel('Reward Density', fontsize=18)
    plt.ylabel('Fraction of States Expanded', fontsize=18)
    #plt.legend(loc='upper left')
    #plt.legend(ncol=2)
    plt.tight_layout()
    FILENAME = 'exploration.pdf'
    figure.savefig(FILENAME, bbox_inches="tight")
    plt.show()
    """

if __name__ == '__main__':
    main()
