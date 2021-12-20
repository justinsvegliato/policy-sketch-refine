import logging
import time

import cplex_mdp_solver
import utils
from partially_abstract_mdp import PartiallyAbstractMDP
import numpy as np
from collections import Counter
import  matplotlib.pyplot as plt
from MP_Features import Features
import torch

logging.basicConfig(format='[%(asctime)s|%(module)-30s|%(funcName)-10s|%(levelname)-5s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


def sketch(abstract_mdp, gamma):
    return cplex_mdp_solver.solve(abstract_mdp, gamma, constant_state_values={}, relax_infeasible=False)


def refine(ground_mdp, ground_state, abstract_mdp, abstract_state, sketched_solution, expand_points_of_interest, expansion_level, gamma, ML_params):
    if expansion_level == 'ML':
        # Unpack ML params
        choices = ML_params['expansion_choices']
        model = ML_params['model']
        config = ML_params['config']

        # Get features for current abstract state
        features = Features(ground_mdp)
        curr_features = features.state_features(ground_state, abstract_mdp, abstract_state, gamma)
        curr_features_flat = features.flatten_state_features(curr_features)
        curr_features_flat.update(config)
        del curr_features_flat['expansion_level']
        del curr_features_flat['abstract_aggregate']
        # print(f'Features flat: {curr_features_flat}\n{len(curr_features_flat)}')
        inputs = torch.Tensor(curr_features_flat.values())

        # Get prediction from model
        # with torch.no_grad():
        logits = model(inputs)
        print(logits)
        expansion_level_idx = torch.argmax(logits)
        expansion_level = choices[expansion_level_idx]
        # print(f'Expansion level: {expansion_level}')
        


        # in_deg_cache = Counter()
        # out_deg_cache = Counter()

        # # Calculat in/out degree on abstract MDP for all states
        # p_thresh = 0
        # abs_states = abstract_mdp.states()
        # abs_trans = abstract_mdp.abstract_transition_probabilities
        # for state, trans_probs in abs_trans.items():
        #     out_counts = Counter()
        #     for action, probs in trans_probs.items():
        #         for s2, prob in probs.items():
        #             if prob > p_thresh:
        #                 out_counts[s2] += 1
        #     out_deg_cache[state] += sum(out_counts.values())
        #     in_deg_cache += out_counts

        # plt.clf()
        # plt.hist(in_deg_cache.values())
        # plt.savefig('in_deg.png')
        # plt.clf()
        # plt.hist(out_deg_cache.values())
        # plt.savefig('out_deg.png')
        # plt.clf()


        # # Single outcome calculation
        # p_thresh = 0
        # abs_states = abstract_mdp.states()
        # abs_trans = abstract_mdp.abstract_transition_probabilities
        # for state, trans_probs in abs_trans.items():
        #     out_counts = Counter()
        #     for action, probs in trans_probs.items():
        #         most_lik = list(probs.items())[0]
        #         for s2, prob in probs.items():
        #             if prob > most_lik[1]:
        #                 most_lik = (s2, prob)
        #         for s2, prob in probs.items():
        #             if s2 is not most_lik[0]:
        #                 probs[s2] = 0
        #     out_deg_cache[state] += sum(out_counts.values())
        #     in_deg_cache += out_counts
        # pass

        # curr_features = [in_deg_cache.values(), out_deg_cache.values()]
        # # Select the best expansion strategy
        # # Load model
        # # with open('clf.pkl', 'wb') as clf:
        # #     expansion_level = clf.predict(curr_features)
        # choices = [2, 'a', 1, 0]
        # expansion_level = np.random.choice(choices, 1)[0]

    if expansion_level == 'r':
        print('using this expansion level!!!!')
        # Randomly choose an expansion level
        choices = [2, 'a', 1, 0]
        expansion_level = np.random.choice(choices, 1)[0]
        print(f'new expansion level: {expansion_level}')
        # print(expansion_level)
        # expansion_level = 2
        # print(expansion_level[0])
    if expansion_level == 'a':
        #values = {}
        #for state in ground_mdp.states():
        #   corr_abstract_stat= abstract_mdp.get_abstract_state(state)
        #   values[state] = sketched_solution['values'][corr_abstract_state]

        #return values
        return sketched_solution


    # print(expansion_level)
    start = time.time()

    # TODO Definitely move this code to anywhere but here
    point_of_interest_locations = []
    point_of_interest_abstract_state_set = set()
    if expand_points_of_interest:
        current_location, current_weather_status = ground_mdp.get_state_factors_from_state(ground_state)
        for point_of_interest_location in current_weather_status:
            if expansion_level == "inf":
                #TODO: finish maybe
                point_of_interest_ground_state = ground_mdp.get_state_from_state_factors(point_of_interest_location, current_weather_status)
                point_of_interest_abstract_state = abstract_mdp.get_abstract_state(point_of_interest_ground_state)
                point_of_interest_abstract_state_set.add(point_of_interest_abstract_state)
                point_of_interest_locations.append(point_of_interest_location)
                
            vertical_distance = abs(current_location[0] - point_of_interest_location[0])
            horizontal_displacement = point_of_interest_location[1] - current_location[1]
            horizontal_distance = abs(horizontal_displacement) if horizontal_displacement >= 0 else ground_mdp.width() - abs(horizontal_displacement)
            # print(vertical_distance, abstract_mdp.abstract_state_height, expansion_level, horizontal_distance, abstract_mdp.abstract_state_width)
            # print(type(abstract_mdp.abstract_state_height * int(expansion_level)))
            # prin::::t(type(abstract_mdp.abstract_state_width * int(expansion_level)))
            if int(vertical_distance) > int(abstract_mdp.abstract_state_height) * int(expansion_level) or int(horizontal_distance) > int(abstract_mdp.abstract_state_width) * int(expansion_level):
                continue
            # if vertical_distance > abstract_mdp.abstract_state_height * expansion_level or horizontal_distance > abstract_mdp.abstract_state_width * expansion_level:
            #     continue

            point_of_interest_ground_state = ground_mdp.get_state_from_state_factors(point_of_interest_location, current_weather_status)
            point_of_interest_abstract_state = abstract_mdp.get_abstract_state(point_of_interest_ground_state)
            point_of_interest_abstract_state_set.add(point_of_interest_abstract_state)
            point_of_interest_locations.append(point_of_interest_location)



            if expansion_level == 2:
                x_range = []
                if current_location[1] < point_of_interest_location[1]:
                    x_range += range(current_location[1], point_of_interest_location[1]+1)
                else:
                    x_range += range(current_location[1], ground_mdp.width())
                    x_range += range(0, point_of_interest_location[1]+1)
                y_range = []
                if current_location[0] < point_of_interest_location[0]:
                    y_range += range(current_location[0], point_of_interest_location[0]+1)
                else:
                    y_range += range(point_of_interest_location[0], current_location[0]+1)
                for x in x_range:
                    for y in y_range:
                      point_of_interest_ground_state = ground_mdp.get_state_from_state_factors((y, x), current_weather_status)
                      point_of_interest_abstract_state = abstract_mdp.get_abstract_state(point_of_interest_ground_state)
                      point_of_interest_abstract_state_set.add(point_of_interest_abstract_state)

        logging.info("Enabled point of interest abstract state expansion: [abstract_states=%s]", point_of_interest_locations)

    # TODO Yikes...
    grounding_abstract_states = list(set([abstract_state] + list(point_of_interest_abstract_state_set)))
    partially_abstract_mdp = PartiallyAbstractMDP(ground_mdp, abstract_mdp, grounding_abstract_states)
    logging.info("Built the PAMDP: [states=%d, actions=%d, time=%f]", len(partially_abstract_mdp.states()), len(partially_abstract_mdp.actions()), time.time() - start)

    abstract_state_set = set(abstract_mdp.states())
    constant_abstract_state_set = abstract_state_set - {abstract_state} - point_of_interest_abstract_state_set
    variable_abstract_state_set = abstract_state_set - constant_abstract_state_set
    logging.info('Initialized state information: [constants=%d, variables=%d]', len(constant_abstract_state_set), len(variable_abstract_state_set))

    """
    while True:
        constant_state_values = {}
        for partially_abstract_state in partially_abstract_mdp.states():
            if partially_abstract_state in constant_abstract_state_set:
                constant_state_values[partially_abstract_state] = sketched_solution['values'][partially_abstract_state]

        start = time.time()
        refined_solution = cplex_mdp_solver.solve(partially_abstract_mdp, gamma, constant_state_values=constant_state_values, relax_infeasible=False)
        logging.info("Ran the CPLEX solver: [time=%f]", time.time() - start)

        if refined_solution:
            logging.info("Found a feasible solution to the PAMDP")
            for constant_abstract_state in constant_abstract_state_set:
                refined_solution['values'][constant_abstract_state] = sketched_solution['values'][constant_abstract_state]
                refined_solution['policy'][constant_abstract_state] = sketched_solution['policy'][constant_abstract_state]
            break

        if not constant_abstract_state_set:
            logging.error("Failed to find a feasible solution to the PAMDP")
            refined_solution = cplex_mdp_solver.solve(partially_abstract_mdp, gamma, constant_state_values=constant_state_values, relax_infeasible=True)
            break

        logging.info('Could not find a feasible solution to the PAMDP')

        successor_abstract_state_set = utils.get_successor_state_set(abstract_mdp, variable_abstract_state_set)
        constant_abstract_state_set -= successor_abstract_state_set
        variable_abstract_state_set = abstract_state_set - constant_abstract_state_set
        logging.info('Updated state information: [successors=%d, constants=%d, variables=%d]',
                     len(successor_abstract_state_set), len(constant_abstract_state_set),
                     len(variable_abstract_state_set))
        
    return refined_solution
    """

    constant_state_values = {}

    start = time.time()
    refined_solution = cplex_mdp_solver.solve(partially_abstract_mdp, gamma, constant_state_values=constant_state_values, relax_infeasible=False)
    logging.info("Ran the CPLEX solver: [time=%f]", time.time() - start)

    if refined_solution:
        logging.info("Found a feasible solution to the PAMDP")
        return refined_solution
    else:
        logging.error("Failed to find a feasible solution to the PAMDP")
        refined_solution = cplex_mdp_solver.solve(partially_abstract_mdp, gamma, constant_state_values=constant_state_values, relax_infeasible=True)
        if refined_solution:        
            logging.info('Found a feasible solution to the PAMDP after relaxing some constraints')
            return refined_solution
        else:
            logging.info('Could not find a feasible solution to the PAMDP')
            return refined_solution

def solve(ground_mdp, ground_state, abstract_mdp, abstract_state, expand_points_of_interest, expansion_level, gamma, ML_params):
    logging.info("Starting the sketch phase...")
    start = time.time()
    sketched_solution = sketch(abstract_mdp, gamma)
    logging.info("Finished the sketch phase: [time=%f]", time.time() - start)

    logging.info("Starting the refine phase...")
    start = time.time()
    refined_solution = refine(ground_mdp, ground_state, abstract_mdp, abstract_state, sketched_solution, expand_points_of_interest, expansion_level, gamma, ML_params)
    logging.info("Finished the refine phase: [time=%f]", time.time() - start)

    return refined_solution
