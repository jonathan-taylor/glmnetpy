from dataclasses import dataclass

import numpy as np

from .glm import GLMFamilySpec

@dataclass
class GibbsGLMFamilySpec(GLMFamilySpec):

    def get_response_and_weights(self,
                                 state,
                                 y,
                                 offset,
                                 sample_weight):

        family = self.base

        # some checks for NAs/zeros
        varmu = family.variance(state.mu)
        if np.any(np.isnan(varmu)): raise ValueError("NAs in V(mu)")

        if np.any(varmu == 0): raise ValueError("0s in V(mu)")

        dmu_deta = family.link.inverse_deriv(state.eta)
        if np.any(np.isnan(dmu_deta)): raise ValueError("NAs in d(mu)/d(eta)")

        newton_weights = sample_weight * dmu_deta**2 / varmu

        # need to track dispersion!!!

        dbn = family.get_distribution(state.mu)
        y_pseudo = dbn.rvs()
        
        # compute working response and weights
        if offset is not None:
            pseudo_response = (state.eta - offset) + (y - state.mu + 2 * (y_pseudo - state.mu)) / dmu_deta
        else:
            pseudo_response = state.eta + (y - state.mu + 2 * (y_pseudo - state.mu)) / dmu_deta

        return pseudo_response, newton_weights


