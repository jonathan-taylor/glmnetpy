from dataclasses import dataclass

import numpy as np

from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             accuracy_score,
                             roc_auc_score,
                             average_precision_score)

@dataclass(frozen=True)
class Scorer(object):
    """Scorer for evaluating model performance.
    
    Parameters
    ----------
    name : str
        Name of the scorer.
    score : callable, optional
        Scoring function that takes (y_true, y_pred, sample_weight) and returns a score.
    maximize : bool, default=True
        Whether higher scores are better.
    use_full_data : bool, default=False
        Whether to use full data instead of split data.
    grouped : bool, default=True
        Whether the scorer operates on grouped data.
    normalize_weights : bool, default=True
        Whether to normalize sample weights.
    """

    name: str
    score: callable=None
    maximize: bool=True
    use_full_data: bool=False
    grouped: bool=True
    normalize_weights: bool=True

    def score_fn(self,
                 split,
                 response,
                 predictions,
                 sample_weight):
        """Compute score for a given split.
        
        Parameters
        ----------
        split : array-like
            Indices for the current split.
        response : array-like
            True response values.
        predictions : array-like
            Predicted values.
        sample_weight : array-like
            Sample weights.
            
        Returns
        -------
        tuple
            Tuple of (score, weight_sum).
        """

        W = np.asarray(sample_weight)[split]
        W_sum = W.sum()
        if self.normalize_weights:
            W = W / W.mean()
        return (self.score(response[split],
                           predictions[split],
                           sample_weight=W), W_sum)

mse_scorer = Scorer(name='Mean Squared Error',
                    score=mean_squared_error,
                    maximize=False)
mae_scorer = Scorer(name='Mean Absolute Error',
                    score=mean_absolute_error,
                    maximize=False)

def _accuracy_score(y, yhat, sample_weight):
    """Compute accuracy score for binary classification.
    
    Parameters
    ----------
    y : array-like
        True binary labels.
    yhat : array-like
        Predicted probabilities.
    sample_weight : array-like, optional
        Sample weights.
        
    Returns
    -------
    float
        Accuracy score.
    """
    return accuracy_score(y,
                          yhat>0.5,
                          sample_weight=sample_weight,
                          normalize=True)

accuracy_scorer = Scorer(name='Accuracy',
                         score=_accuracy_score,
                         maximize=True)
auc_scorer = Scorer('AUC',
                    score=roc_auc_score,
                    maximize=True)
aucpr_scorer = Scorer('AUC-PR',
                      score=average_precision_score,
                      maximize=True)

class UngroupedScorer(Scorer):
    """Scorer for ungrouped data.
    
    This scorer operates on individual observations rather than grouped data.
    """

    grouped: bool=False
    score: callable=None

    def score_fn(self,
                 split,
                 response,
                 predictions,
                 sample_weight):
        """Compute score for ungrouped data.
        
        Parameters
        ----------
        split : array-like
            Indices for the current split.
        response : array-like
            True response values.
        predictions : array-like
            Predicted values.
        sample_weight : array-like
            Sample weights.
            
        Returns
        -------
        tuple
            Tuple of (score, weights).
        """
        return self.score(response[split], predictions[split]), sample_weight[split]

ungrouped_mse_scorer = UngroupedScorer(name="Mean Squared Error (Ungrouped)",
                                       score=lambda y, pred: (y-pred)**2,
                                       maximize=False,
                                       grouped=False)

ungrouped_mae_scorer = UngroupedScorer(name="Mean Absolute Error (Ungrouped)",
                                       score=lambda y, pred: np.fabs(y-pred),
                                       maximize=False,
                                       grouped=False)
