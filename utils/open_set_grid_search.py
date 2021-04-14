from sklearn.model_selection import GridSearchCV
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.utils.validation import indexable, _check_fit_params, _num_samples
from sklearn.model_selection._validation import _score
from sklearn.utils import _message_with_time
from traceback import format_exc
from sklearn.exceptions import FitFailedWarning
from sklearn.base import clone
from joblib import Parallel, delayed
from itertools import product
import numpy as np
import warnings
import numbers
import time
from utils.open_set_split import OpenSetShuffleSplit
from sklearn.metrics._scorer import _BaseScorer


def _fit_and_score_open_set(
        estimator, X_train, y_train, X_test, y_test, scorer, verbose,
        parameters, fit_params, return_train_score=False,
        return_parameters=False, return_n_test_samples=False,
        return_times=False, return_estimator=False,
        error_score=np.nan):
    """
    Adapts the method sklearn.model_selection._validation._fit_and_score to the open-set setting, where some labels correspond to unknown camera models.

    Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X_train : array-like of shape (n_samples, n_features)
        The data to fit.

    y_train : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of supervised learning.

    X_test : array-like of shape (n_samples, n_features)
        The test data

    y_test : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The test labels

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    verbose : int
        The verbosity level.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : bool, default=False
        Compute and return score on training set.

    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.

    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``

    return_times : bool, default=False
        Whether to return the fit/score times.

    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    Returns
    -------
    train_scores : dict of scorer name -> float
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.

    test_scores : dict of scorer name -> float
        Score on testing set (for all the scorers).

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None
        The parameters that have been evaluated.

    estimator : estimator object
        The fitted estimator
    """
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v) for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X_train, fit_params)

    train_scores = {}
    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Estimator fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_score, format_exc()),
                          FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, y_test, scorer)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer)
    if verbose > 2:
        if isinstance(test_scores, dict):
            for scorer_name in sorted(test_scores):
                msg += ", %s=" % scorer_name
                if return_train_score:
                    msg += "(train=%.3f," % train_scores[scorer_name]
                    msg += " test=%.3f)" % test_scores[scorer_name]
                else:
                    msg += "%.3f" % test_scores[scorer_name]
        else:
            msg += ", score="
            msg += ("%.3f" % test_scores if not return_train_score else
                    "(train=%.3f, test=%.3f)" % (train_scores, test_scores))

    if verbose > 1:
        total_time = score_time + fit_time
        print(_message_with_time('CV', msg, total_time))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_estimator:
        ret.append(estimator)
    return ret


class PredictWithProbScorer(_BaseScorer):
    """
    Custom scorer for the PI-SVM that predicts both y_pred and y_prob.
    """

    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        if sample_weight:
            raise NotImplementedError("sample_weight not implemented")

        y_pred, y_prob = method_caller(estimator, "predict", X)
        return self._sign * self._score_func(y_true, y_pred, y_prob, **self._kwargs)


class OpenSetGridSearch(GridSearchCV):
    def __init__(self, estimator, param_grid, n_splits, scoring=None, n_jobs=None, verbose=0):
        super().__init__(estimator=estimator, param_grid=param_grid, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
        self.n_splits = n_splits
        self.cv = OpenSetShuffleSplit(n_splits, train_percentage=0.8)

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """
        Run fit with all sets of parameters
        :param X:
        :param y:
        :param groups:
        :param fit_params:
        :return:
        """
        scorers, self.multimetric_ = _check_multimetric_scoring(self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, str) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers) and not callable(self.refit):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key or a "
                                 "callable to refit an estimator with the "
                                 "best parameter setting on the whole "
                                 "data and make the best_* attributes "
                                 "available for that metric. If this is "
                                 "not needed, refit should be set to "
                                 "False explicitly. %r was passed."
                                 % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(scorer=scorers,
                                    fit_params=fit_params,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=False,
                                    error_score=self.error_score,
                                    verbose=self.verbose)
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []

            def evaluate_candidates(candidate_params):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print("Fitting {0} folds for each of {1} candidates, totalling {2} fits".format(self.n_splits,
                                                                                                    n_candidates,
                                                                                                    n_candidates * self.n_splits))

                out = parallel(delayed(_fit_and_score_open_set)(
                    clone(base_estimator),
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    parameters=parameters,
                    **fit_and_score_kwargs)
                               for parameters, (X_train, y_train, X_test, y_test)
                               in product(candidate_params, self.cv.split(X, y)))

                if len(out) < 1:
                    raise ValueError('No fits were performed. '
                                     'Was the CV iterator empty? '
                                     'Were there no candidates?')
                elif len(out) != n_candidates * self.n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned '
                                     'inconsistent results. Expected {} '
                                     'splits, got {}'
                                     .format(self.n_splits, len(out) // n_candidates))

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, scorers, self.n_splits, all_out)
                return results

            self._run_search(evaluate_candidates)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or
                        self.best_index_ >= len(results["params"])):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results["rank_test_%s"
                                           % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(clone(base_estimator).set_params(
                **self.best_params_))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results

        return self
