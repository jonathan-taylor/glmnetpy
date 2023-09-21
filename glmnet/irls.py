import logging

import numpy as np
from copy import deepcopy

from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links

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
    
    pseudo_response, newton_weights = get_response_and_weights(family,
                                                               state,
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
        # test for convergence
        if ((np.fabs(state.obj_val - obj_val_old)/(0.1 + abs(state.obj_val)) < control.epsnr) or
            (isinstance(family, sm_family.Gaussian) and isinstance(family.link, sm_links.Identity))):
            converged = True
            break

    if control.logging:
        logging.info(f'Terminating ISLR after {i+1} iterations.')
        logging.debug(f'{regularizer._debug_msg(state)}')
    return converged, boundary, state, newton_weights

# would be good to have the family know how to do this...
# maybe our family should be an sm.Family with this method (and others?)

def get_response_and_weights(family,
                             state,
                             y,
                             offset,
                             weights):

    if isinstance(family, sm_family.Family):
        # some checks for NAs/zeros
        varmu = family.variance(state.mu)
        if np.any(np.isnan(varmu)): raise ValueError("NAs in V(mu)")

        if np.any(varmu == 0): raise ValueError("0s in V(mu)")

        dmu_deta = family.link.inverse_deriv(state.eta)
        if np.any(np.isnan(dmu_deta)): raise ValueError("NAs in d(mu)/d(eta)")

        newton_weights = weights * dmu_deta**2 / varmu

        # compute working response and weights
        if offset is not None:
            pseudo_response = (state.eta - offset) + (y - state.mu) / dmu_deta
        else:
            pseudo_response = state.eta + (y - state.mu) / dmu_deta

        return pseudo_response, newton_weights
