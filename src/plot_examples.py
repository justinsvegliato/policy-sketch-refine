from argparse import ArgumentParser
from run import get_x_y
import matplotlib.pyplot as plt
import math

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
    arg_parser.add_argument("data_dir")
    
    args = arg_parser.parse_args()
    baseline_config_file = args.baseline_config_file
    config_file_0 = args.config_file_0
    config_file_1 = args.config_file_1
    config_file_2 = args.config_file_2
    data_dir = args.data_dir

##### Before Secondary Batch #####
#TODO: ask for Justin's plotting / formatting code

#TODO: Plot some freaking histograms dude.... for:
#    chache hits / misses
#    %states visited
#    ????

    plt.rcParams["font.family"] = "Times New Roman"
    # font size of 16 is far too large
    #plt.rcParams["font.size"] = 16

##### Possible if given time #####
#TODO: add negative reward for moving north and south?
#TODO: experiment with transition function perturbations?

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
 
    figure = plt.figure(figsize=(7, 3))

    # Plot
    plt.plot(b_x, b_y_mean, 'r', label='Ground MDP')
    plt.plot(x0, y0_mean, 'b', label='Naive Strategy')
    plt.plot(x1, y1_mean, 'g', label='Greedy Strategy')
    plt.plot(x2, y2_mean, 'k', label='Proactive Strategy')
    plt.plot(x0_abstract, y0_abstract_mean, 'b--', label='Abstraction Time')
    plt.fill_between(b_x, b_lb, b_ub, alpha=0.5, color='r')
    plt.fill_between(x0, lb0, ub0, alpha=0.5, color='b')
    plt.fill_between(x1, lb1, ub1, alpha=0.5, color='g')
    plt.fill_between(x2, lb2, ub2, alpha=0.3, color='k')
    plt.yscale('log')
    # plt.title('Time to Compute Policy vs. Number of States')
    plt.xlabel('Ground State Space Size')
    plt.ylabel('Comulative Planning Time [seconds]')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()



    # cum reward ratio vs. number of states

    b_x, b_y = get_x_y(data_dir, baseline_config_file, x_func=n_states, y_func=cumulative_reward, sort=True)
    x0, y0 = get_x_y(data_dir, config_file_0, x_func=n_states, y_func=cumulative_reward, sort=True)
    x1, y1 = get_x_y(data_dir, config_file_1, x_func=n_states, y_func=cumulative_reward, sort=True)
    x2, y2 = get_x_y(data_dir, config_file_2, x_func=n_states, y_func=cumulative_reward, sort=True)

    y0 = [i / j for i, j in zip(y0, b_y)]
    y1 = [i / j for i, j in zip(y1, b_y)]
    y2 = [i / j for i, j in zip(y2, b_y)]
    b_y = [1.0 for _ in range(len(y0))]

    figure = plt.figure(figsize=(7, 3))
    # Plot a histogram
    plt.hist(y0, alpha=0.5, color='b', label='Naive Strategy', density=True)
    plt.hist(y1, alpha=0.5, color='g', label='Greedy Strategy', density=True)
    plt.hist(y2, alpha=0.5, color='k', label='Proactive Strategy', density=True)
    #plt.title('Cumulative Reward Ratio Frequencies')
    plt.xlabel('Cumulative Reward Ratio')
    plt.ylabel('Frequency')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # calculate mean and variance 
    b_x, b_y_mean, b_y_var, b_conf_95 = calculate_statistics(b_x, b_y)
    x0, y0_mean, y0_var, conf0_95 = calculate_statistics(x0, y0)
    x1, y1_mean, y1_var, conf1_95 = calculate_statistics(x1, y1)
    x2, y2_mean, y2_var, conf2_95 = calculate_statistics(x2, y2)
    
    # calculate confidence intervals 
    b_ub, b_lb = calculate_confidence_interval(b_y_mean, b_conf_95)
    ub0, lb0 = calculate_confidence_interval(y0_mean, conf0_95)
    ub1, lb1 = calculate_confidence_interval(y1_mean, conf1_95)
    ub2, lb2 = calculate_confidence_interval(y2_mean, conf2_95)

    figure = plt.figure(figsize=(7, 3))
    # Plot
    plt.plot(b_x, b_y_mean, 'r--', label='Ground MDP')
    plt.plot(x0, y0_mean, 'b', label='Naive Strategy')
    plt.plot(x1, y1_mean, 'g', label='Greedy Strategy')
    plt.plot(x2, y2_mean, 'k', label='Proactive Strategy')
    plt.fill_between(x0, lb0, ub0, alpha=0.5, color='b')
    plt.fill_between(x1, lb1, ub1, alpha=0.5, color='g')
    plt.fill_between(x2, lb2, ub2, alpha=0.3, color='k')
    plt.ylim(0.5, 1.1)
    #plt.yscale('symlog')
    #plt.xscale('log')
    #plt.title('Cumulative Reward Ratio vs. Number of States')
    plt.xlabel('Number of States')
    plt.ylabel('Cumulative Reward Ratio')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()



    # cum reward ratio vs. reward density
    
    b_x, b_y = get_x_y(data_dir, baseline_config_file, x_func=reward_density, y_func=cumulative_reward, sort=True)
    x0, y0 = get_x_y(data_dir, config_file_0, x_func=reward_density, y_func=cumulative_reward, sort=True)
    x1, y1 = get_x_y(data_dir, config_file_1, x_func=reward_density, y_func=cumulative_reward, sort=True)
    x2, y2 = get_x_y(data_dir, config_file_2, x_func=reward_density, y_func=cumulative_reward, sort=True)

    y0 = [i / j for i, j in zip(y0, b_y)]
    y1 = [i / j for i, j in zip(y1, b_y)]
    y2 = [i / j for i, j in zip(y2, b_y)]
    b_y = [1.0 for _ in range(len(y0))]

    # calculate mean and variance 
    b_x, b_y_mean, b_y_var, b_conf_95 = calculate_statistics(b_x, b_y)
    x0, y0_mean, y0_var, conf0_95 = calculate_statistics(x0, y0)
    x1, y1_mean, y1_var, conf1_95 = calculate_statistics(x1, y1)
    x2, y2_mean, y2_var, conf2_95 = calculate_statistics(x2, y2)
    
    # calculate confidence intervals 
    b_ub, b_lb = calculate_confidence_interval(b_y_mean, b_conf_95)
    ub0, lb0 = calculate_confidence_interval(y0_mean, conf0_95)
    ub1, lb1 = calculate_confidence_interval(y1_mean, conf1_95)
    ub2, lb2 = calculate_confidence_interval(y2_mean, conf2_95)

    figure = plt.figure(figsize=(7, 3))
    # Plot
    plt.plot(b_x, b_y_mean, 'r--', label='Ground MDP')
    plt.plot(x0, y0_mean, 'b', label='Naive Strategy')
    plt.plot(x1, y1_mean, 'g', label='Greedy Strategy')
    plt.plot(x2, y2_mean, 'k', label='Proactive Strategy')
    plt.fill_between(x0, lb0, ub0, alpha=0.5, color='b')
    plt.fill_between(x1, lb1, ub1, alpha=0.5, color='g')
    plt.fill_between(x2, lb2, ub2, alpha=0.3, color='k')
    #plt.ylim(0.0, 1.1)
    plt.ylim(0.5, 1.1)
    #plt.title('Cumulative Reward Ratio vs. Reward Sparsity')
    plt.xlabel('Reward Density')
    plt.ylabel('Cumulative Reward Ratio')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()



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
    plt.xlabel('Reward Density')
    plt.ylabel('Fraction of States Expanded')
    #plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
