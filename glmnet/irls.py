import logging

import numpy as np
from copy import deepcopy

# for Gaussian check below

def quasi_newton_step(regularizer,
                      family,
                      design,
                      y,
                      offset,
                      weights,
                      state,
                      objective,
                      control):


    oldstate = deepcopy(state)
    
    pseudo_response, newton_weights = family.get_response_and_weights(state,
                                                                      y,
                                                                      offset,
                                                                      weights)

    state = regularizer.newton_step(design,
                                    pseudo_response,
                                    newton_weights,
                                    state)

    state.update(design,
                 family,
                 offset,
                 objective)


    # check to make sure it is a feasible descent step

    boundary = False
    halved = False  # did we have to halve the step size?

    # three checks we'll apply

    # FIX THESE 
    _valid = lambda state: True

    # not sure boundary / halved handled correctly

    def finite_objective(state):
        boundary = True
        halved = True
        return np.isfinite(state.obj_val) and state.obj_val < control.big, boundary, halved

    def valid(state):
        boundary = True
        halved = True
        return _valid(state), boundary, halved

    def decreased_obj(state):
        boundary = False
        halved = True

        return state.obj_val <= oldstate.obj_val + 1e-7, boundary, halved

    for test, msg in [(finite_objective,
                       "Non finite objective function! Step size truncated due to divergence."),
                      (valid,
                       "Invalid eta/mu! Step size truncated: out of bounds."),
                      (decreased_obj,
                       "Objective did not decrease!")]:

        if not test(state)[0]:
            if control.logging: logging.debug(msg)
            regularizer.check_state(oldstate)

            ii = 1
            check, boundary_, halved_ = test(state)
            if not check:
                boundary = boundary or boundary_
                halved = halved or halved_

            while not check:
                if ii > control.mxitnr:
                    raise ValueError(f"inner loop {test}; cannot correct step size")
                ii += 1

                state = regularizer.half_step(state,
                                              oldstate)
                state.update(design,
                             family,
                             offset,
                             objective)
                check, boundary_, halved_ = test(state)

    if control.logging: logging.debug(f'old value: {oldstate.obj_val}, new value: {state.obj_val}') 

    return state, boundary, halved, newton_weights

def IRLS(regularizer,
         family,
         design,
         y,
         offset,
         weights,
         state,
         objective,
         control):

    converged = False

    if control.logging:
        logging.info('Starting ISLR')
        logging.debug(f'{regularizer._debug_msg(state)}')

    for i in range(control.mxitnr):

        obj_val_old = state.obj_val

        (state,
         boundary,
         halved,
         newton_weights) = quasi_newton_step(regularizer,
                                             family,
                                             design,
                                             y,
                                             offset,
                                             weights,
                                             state,
                                             objective,
                                             control)

        if control.logging:
            logging.debug(f'Iteration {i}, {regularizer._debug_msg(state)}')
            logging.info(f'Objective: {state.obj_val}')

        is_gaussian = False
        if hasattr(family, 'base'):
            base_family = family.base

        # test for convergence
        if ((np.fabs(state.obj_val - obj_val_old)/(0.1 + abs(state.obj_val)) < control.epsnr) or
            family.is_gaussian):
            converged = True
            break

    if control.logging:
        logging.info(f'Terminating ISLR after {i+1} iterations.')
        logging.debug(f'{regularizer._debug_msg(state)}')
    return converged, boundary, state, newton_weights

