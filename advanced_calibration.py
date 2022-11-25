import sklearn
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import binom
from sklearn import linear_model
from sklearn.metrics import log_loss, make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold

def _natural_cubic_spline_basis_expansion(xpts, knots):
    num_knots = len(knots)
    num_pts = len(xpts)
    outmat = np.zeros((num_pts, num_knots))
    outmat[:, 0] = np.ones(num_pts)
    outmat[:, 1] = xpts

    def make_func_H(k):
        def make_func_d(k):
            def func_d(x):
                denom = knots[-1] - knots[k-1]
                numer = np.maximum(x-knots[k-1], 
                                   np.zeros(len(x))) ** 3 - np.maximum(x-knots[-1], 
                                                                       np.zeros(len(x))) ** 3
                return numer/denom
            return func_d

        def func_H(x):
            d_fun_k = make_func_d(k)
            d_fun_Km1 = make_func_d(num_knots-1)
            return d_fun_k(x) -  d_fun_Km1(x)
        return func_H
    for i in range(1, num_knots-1):
        curr_H_fun = make_func_H(i)
        outmat[:, i+1] = curr_H_fun(xpts)
    return outmat


def prob_calibration_function(truthvec, scorevec, reg_param_vec='default', knots='sample',
                              method='logistic', force_prob=True, eps=1e-15, max_knots=200,
                              transform_fn='none', random_state=42, verbose=True, cv_folds=5,
                              unity_prior_weight=1, unity_prior_gridsize=20):
    """This function takes an uncalibrated set of scores and the true 0/1 values and returns a calibration function.
    This calibration function can then be applied to other scores from the same model and will return an accurate probability
    based on the data it has seen.  For best results, the calibration should be done on a separate validation set (not used
    to train the model).
    Parameters
    ----------
    truthvec : A numpy array containing the true values that is the target of the calibration.  For binary
     classification these are typically your 0/1 values.
    scorevec : A numpy array containing the scores that are not appropriate to be used as probabilities.  These do not
     necessarily need to be between 0 and 1, though that is the typical usage.
    reg_param_vec:  The vector of C-values (if method = 'logistic') or alpha values (if method = 'ridge') that the calibration should
      search across.  If reg_param_vec = 'default' (which is the default) then it picks a reasonable set of values to search across.
    knots: Default is 'sample', which means it will randomly pick a subset of size max_knots from the unique values in scorevec (while always
        keeping the largest and smallest value).  If knots='all' it will use all unique values of scorevec as knots.  This may yield a
        better calibration, but will be slower.
    method : 'logistic' or 'ridge'
        The default is 'logistic', which is best if you plan to use log-loss as your metric.  You can also use "ridge" which may
        be better if Brier Score is your metic of interest.  However, "ridge" may be less stable and robust, especially when used
        for probabilities.
    force_prob: This is ignored for method = 'logistic'.  For method = 'ridge', if set to True (the default), it will ensure that the
        values coming out of the calibration are between eps and 1-eps.
    eps: default is 1e-15.  Applies only if force_prob = True and method = 'ridge'.  See force_prob above.
    max_knots:  The number of knots to use when knots='sample'.  See knots above.
    random_state: default is 942 (a particular value to ensure consistency when running multiple times).  User can supply a different value
        if they want a different random seed.
    Returns
    ---------------------
    A function object which takes a numpy array (or a single number) and returns the output of the calculated calibration function.
    """

    if (unity_prior_weight>0):
        scorevec_coda, truthvec_coda = create_yeqx_bias_vectors(unity_prior_gridsize)
        coda_wt = unity_prior_weight/unity_prior_gridsize
        weightvec = np.concatenate((np.ones(len(scorevec)), coda_wt * np.ones(len(scorevec_coda))))
        scorevec = np.concatenate((scorevec, scorevec_coda))
        truthvec = np.concatenate((truthvec, truthvec_coda))

    if transform_fn != 'none':
        scorevec = transform_fn(scorevec)

    knot_vec = np.unique(scorevec)
    if (knots == 'sample'):
        num_unique = len(knot_vec)
        if (num_unique > max_knots):
            smallest_knot, biggest_knot = knot_vec[0], knot_vec[-1]
            inter_knot_vec = knot_vec[1:-1]
            random.seed(random_state)
            random.shuffle(inter_knot_vec)
            reduced_knot_vec = inter_knot_vec[:(max_knots-2)]
            reduced_knot_vec = np.concatenate((reduced_knot_vec, [smallest_knot, biggest_knot]))
            reduced_knot_vec = np.concatenate((reduced_knot_vec, np.linspace(0, 1, 21)))
            if (unity_prior_weight>0):
                reduced_knot_vec = np.concatenate((reduced_knot_vec, scorevec_coda))
            knot_vec = np.unique(reduced_knot_vec)
        if verbose:
            print("Originally there were {} knots.  Reducing to {} while preserving first and last knot.".format(num_unique, len(knot_vec)))
    X_mat = _natural_cubic_spline_basis_expansion(scorevec, knot_vec)

    if (method == 'logistic'):
        if ((type(reg_param_vec) == str) and (reg_param_vec == 'default')):
            reg_param_vec = 10**np.linspace(-7, 5, 61)
        if verbose:
            print("Trying {} values of C between {} and {}".format(len(reg_param_vec), 
                                                                   np.min(reg_param_vec), 
                                                                   np.max(reg_param_vec)))
        reg = linear_model.LogisticRegressionCV(Cs=reg_param_vec, 
                                                solver='liblinear', 
                                                cv=StratifiedKFold(cv_folds, 
                                                                   shuffle=True, 
                                                                   random_state=42),
                                                scoring=make_scorer(log_loss, 
                                                                    needs_proba=True, 
                                                                    greater_is_better=False))
        if (unity_prior_weight>0):
            reg.fit(X_mat, truthvec, weightvec)
        else:
            reg.fit(X_mat, truthvec)
        if verbose:
            print("Best value found C = {}".format(reg.C_))

    if (method == 'ridge'):
        if ((type(reg_param_vec) == str) and (reg_param_vec == 'default')):
            reg_param_vec = 10**np.linspace(-7, 7, 71)
        if verbose:
            print("Trying {} values of alpha between {} and {}".format(len(reg_param_vec), 
                                                                       np.min(reg_param_vec),
                                                                       np.max(reg_param_vec)))
        reg = linear_model.RidgeCV(alphas=reg_param_vec, cv=KFold(cv_folds, 
                                                                  shuffle=True, 
                                                                  random_state=42), 
                                   scoring=make_scorer(mean_squared_error_trunc,
                                                       needs_proba=False, 
                                                       greater_is_better=False))
        reg.fit(X_mat, truthvec)
        if verbose:
            print("Best value found alpha = {}".format(reg.alpha_))

    def calibrate_scores(new_scores):
        new_scores = np.maximum(new_scores,knot_vec[0]*np.ones(len(new_scores)))
        new_scores = np.minimum(new_scores,knot_vec[-1]*np.ones(len(new_scores)))
        if transform_fn != 'none':
            new_scores = transform_fn(new_scores)
        basis_exp = _natural_cubic_spline_basis_expansion(new_scores,knot_vec)
        if (method == 'logistic'):
            outvec = reg.predict_proba(basis_exp)[:,1]
        if (method == 'ridge'):
            outvec = reg.predict(basis_exp)
            if force_prob:
                outvec = np.where(outvec < eps, eps, outvec)
                outvec = np.where(outvec > 1-eps, 1-eps, outvec)
        return outvec

    return calibrate_scores


def mean_squared_error_trunc(y_true, y_pred,eps=1e-15):
    y_pred = np.where(y_pred<eps,eps,y_pred)
    y_pred = np.where(y_pred>1-eps,1-eps,y_pred)
    return np.average((y_true-y_pred)**2)


def prob_calibration_function_multiclass(truthvec, scoremat, verbose=False, **kwargs):
    """This function takes an uncalibrated set of scores and the true 0/1 values and returns a calibration function.
    This calibration function can then be applied to other scores from the same model and will return an accurate probability
    based on the data it has seen.  For best results, the calibration should be done on a separate validation set (not used
    to train the model).
    Parameters
    ----------
    truthvec : A numpy array containing the true values that is the target of the calibration.  For binary
     classification these are typically your 0/1 values.
    scorevec : A numpy array containing the scores that are not appropriate to be used as probabilities.  These do not
     necessarily need to be between 0 and 1, though that is the typical usage.
    reg_param_vec:  The vector of C-values (if method = 'logistic') or alpha values (if method = 'ridge') that the calibration should
      search across.  If reg_param_vec = 'default' (which is the default) then it picks a reasonable set of values to search across.
    knots: Default is 'sample', which means it will randomly pick a subset of size max_knots from the unique values in scorevec (while always
        keeping the largest and smallest value).  If knots='all' it will use all unique values of scorevec as knots.  This may yield a
        better calibration, but will be slower.
    method : 'logistic' or 'ridge'
        The default is 'logistic', which is best if you plan to use log-loss as your metric.  You can also use "ridge" which may
        be better if Brier Score is your metic of interest.  However, "ridge" may be less stable and robust, especially when used
        for probabilities.
    force_prob: This is ignored for method = 'logistic'.  For method = 'ridge', if set to True (the default), it will ensure that the
        values coming out of the calibration are between eps and 1-eps.
    eps: default is 1e-15.  Applies only if force_prob = True and method = 'ridge'.  See force_prob above.
    max_knots:  The number of knots to use when knots='sample'.  See knots above.
    random_state: default is 942 (a particular value to ensure consistency when running multiple times).  User can supply a different value
        if they want a different random seed.
    Returns
    ---------------------
    A function object which takes a numpy array (or a single number) and returns the output of the calculated calibration function.
    """

    num_classes = scoremat.shape[1]
    function_list = []
    for i in range(num_classes):
        scorevec = scoremat[:,i]
        curr_truthvec = (truthvec==i).astype(int)
        function_list.append(prob_calibration_function(curr_truthvec,
                                                       scorevec,
                                                       verbose=verbose,
                                                       **kwargs))

    def calibrate_scores_multiclass(new_scoremat):
        a,b = new_scoremat.shape
        pre_probmat = np.zeros((a,b))
        for i in range(num_classes):
            pre_probmat[:,i] = function_list[i](new_scoremat[:,i])
        probmat = (pre_probmat.T/np.sum(pre_probmat,axis=1)).T
        #if (not extrapolate):
        #    new_scores = np.maximum(new_scores,knot_vec[0]*np.ones(len(new_scores)))
        #    new_scores = np.minimum(new_scores,knot_vec[-1]*np.ones(len(new_scores)))
        return probmat
    return calibrate_scores_multiclass, function_list

def plot_prob_calibration(calib_fn, 
                          show_baseline=True, 
                          ax=None, 
                          **kwargs):
    if ax is None:
        ax = plt.gca()
        fig = ax.get_figure()
    ax.plot(np.linspace(0,1,100),
            calib_fn(np.linspace(0,1,100)),
            **kwargs)
    if show_baseline:
        ax.plot(np.linspace(0,1,100),
                (np.linspace(0,1,100)), 'k--')
    ax.axis([-0.1,1.1,-0.1,1.1])

def plot_reliability_diagram(y,
                             x,
                             bins=np.linspace(0,1,21),
                             size_points=False, 
                             show_baseline=True,
                             error_bars=True, 
                             error_bar_alpha=.05, 
                             marker='+',
                             c='red', 
                             **kwargs):
    # if ax is None:
    #     ax = plt.gca()
    #     fig = ax.get_figure()
    digitized_x = np.digitize(x, bins)
    mean_count_array = np.array([[np.mean(y[digitized_x == i]),
                                  len(y[digitized_x == i]),
                                  np.mean(x[digitized_x==i])] for i in np.unique(digitized_x)])
    x_pts_to_graph = mean_count_array[:,2]
    y_pts_to_graph = mean_count_array[:,0]
    pt_sizes = mean_count_array[:,1]
    plt.subplot(1,2,1)

    if show_baseline:
        plt.plot(np.linspace(0,1,100),(np.linspace(0,1,100)),'k--')
#        ax.loglog(np.linspace(0,1,100),(np.linspace(0,1,100)),'k--')
    for i in range(len(y_pts_to_graph)):
        if size_points:
            plt.scatter(x_pts_to_graph,y_pts_to_graph,s=pt_sizes,marker=marker,c=c, **kwargs)
        else:
            plt.scatter(x_pts_to_graph,y_pts_to_graph, c=c, **kwargs)
    plt.axis([-0.1,1.1,-0.1,1.1])
    
    if error_bars:
        yerr_mat = binom.interval(1-error_bar_alpha,pt_sizes,x_pts_to_graph)/pt_sizes - x_pts_to_graph
        yerr_mat[0,:] = -yerr_mat[0,:]
        plt.errorbar(x_pts_to_graph, x_pts_to_graph, yerr=yerr_mat, capsize=5)
    plt.subplot(1,2,2)
    plt.hist(x,bins=bins)

    return(x_pts_to_graph,y_pts_to_graph,pt_sizes)

def compact_logit(x, eps=.00001):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        return np.nansum(((x<=eps)*x, (x>=(1-eps))*x, ((x>eps)&(x<(1-eps)))*((1-2*eps)*(np.log(x/(1-x)))/(2*np.log((1-eps)/eps))+.5)),axis=0)

def inverse_compact_logit(x, eps=.00001):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        return np.nansum(((x<=eps)*x, (x>=(1-eps))*x,
                          ((x>eps)&(x<(1-eps)))*
                          (1/(1+np.exp(-(x-.5)*((2/(1-2*eps))*np.log((1-eps)/eps)))))),axis=0)

def create_yeqx_bias_vectors(gridsize=10):
    scorevec_coda = np.sort(np.tile(np.arange(gridsize + 1)/gridsize, reps = (gridsize)))
    truthvec_coda = np.array([])
    for i in range(gridsize + 1):
        added_bit = np.concatenate((np.zeros(gridsize - i), np.ones(i)))
        truthvec_coda = np.concatenate((truthvec_coda, added_bit))
    return scorevec_coda, truthvec_coda

class SplineCalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    """Probability calibration using cubic splines.
    With this class, the base_estimator is fit on each of the cross-validation
    training set folds in order to generate scores on the (cross-validated)
    test set folds.  The test set scores are accumulated into a final vector
    (the size of the full set) which is used to calibrate the answers.
    The model is then fit on the full data set.  The predict, and predict_proba
    methods are then updated to use the combination of the predictions from the 
    full model and the calibration function computed as above.
    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. If cv='prefit', the
        classifier must have been fit already on data.
    method : 'logistic' or 'ridge'
        The default is 'logistic', which is best if you plan to use log-loss as your
        performance metric.  This method is relatively robust and will typically do
        well on brier score as well.  The 'ridge' method calibrates using an L2 loss,
        and therefore should do better for brier score, but may do considerably worse
        on log-loss.
    cv : integer, cross-validation generator, iterable or "prefit", optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - 'prefit', if you wish to use the data only for calibration
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
        neither binary nor multiclass, :class:`sklearn.model_selection.KFold`
        is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        If "prefit" is passed, it is assumed that base_estimator has been
        fitted already and all data is used for calibration.
    Attributes
    ----------
    uncalibrated_classifier: this gives the uncalibrated version of the classifier, fit on the entire data set
    calib_func: this is the calibration function that has been learned from the cross-validation.  Applying this function
     to the results of the uncalibrated classifier (via model.predict_proba(X_test)[:,1]) gives the fully calibrated classifier
    References
    ----------
   """
    def __init__(self, base_estimator=None, method='logistic', 
                 cv=5, verbose=True, transform_type='none',
                 cl_eps = .000001, **calib_kwargs):
        self.base_estimator = base_estimator
        self.uncalibrated_classifier = None
        self.calib_func = None
        self.method = method
        self.cv = cv
        self.verbose = verbose
        self.cl_eps = cl_eps
        self.calib_kwargs = calib_kwargs
        self.fit_on_multiclass = False
        self.transform_type = transform_type
        self.pre_transform = lambda x: x
        if type(self.transform_type) == str:
            if self.transform_type == 'cl':
                self.pre_transform = lambda x: compact_logit(x, eps=self.cl_eps)
        if callable(self.transform_type):
            self.pre_transform = self.transform_type

    def fit(self, X, y, verbose=True):
        """Fit the calibrated model
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        Returns
        -------
        self : object
            Returns an instance of self.
        """
        
        if len(np.unique(y)) > 2:
            self.fit_on_multiclass = True
            return self._fit_multiclass(X, y, verbose=verbose)

        self.fit_on_multiclass=False
        if ((type(self.cv)==str) and (self.cv=='prefit')):
            self.uncalibrated_classifier = self.base_estimator
            y_pred = self.uncalibrated_classifier.predict_proba(X)[:,1]

        else:
            y_pred = np.zeros(len(y))
            
            if sklearn.__version__ < '0.18':
                if type(self.cv)==int:
                    skf = StratifiedKFold(y, n_folds=self.cv, shuffle=True, random_state=42)
                else:
                    skf = self.cv
            else:
                if type(self.cv)==int:
                    skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42).split(X, y)
                else:
                    skf = self.cv.split(X, y)
            for idx, (train_idx, test_idx) in enumerate(skf):
                if verbose:
                    print("training fold {} of {}".format(idx + 1, self.cv))
                X_train = np.array(X)[train_idx,:]
                X_test = np.array(X)[test_idx,:]
                y_train = np.array(y)[train_idx]
                # We could also copy the model first and then fit it
                this_estimator = clone(self.base_estimator)
                this_estimator.fit(X_train,y_train)
                y_pred[test_idx] = this_estimator.predict_proba(X_test)[:,1]
            
            if verbose:
                print("Training Full Model")
            self.uncalibrated_classifier = clone(self.base_estimator)
            self.uncalibrated_classifier.fit(X, y)

        # calibrating function
        if verbose:
            print("Determining Calibration Function")
        if self.method=='logistic':
            self.calib_func = prob_calibration_function(y, 
                                                        self.pre_transform(y_pred), 
                                                        verbose=verbose, 
                                                        **self.calib_kwargs)
        if self.method=='ridge':
            self.calib_func = prob_calibration_function(y, 
                                                        self.pre_transform(y_pred), 
                                                        method='ridge', 
                                                        verbose=verbose, 
                                                        **self.calib_kwargs)
        # training full model

        return self

    def _fit_multiclass(self, X, y, verbose=False):
        """Fit the calibrated model in multiclass setting
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        Returns
        -------
        self : object
            Returns an instance of self.
        """
        class_list = np.unique(y)
        num_classes = len(class_list)
        y_mod = np.zeros(len(y))
        for i in range(num_classes):
            y_mod[y==class_list[i]]=i

        y_mod = y_mod.astype(int)
        if ((type(self.cv)==str) and (self.cv=='prefit')):
            self.uncalibrated_classifier = self.base_estimator
            y_pred = self.uncalibrated_classifier.predict_proba(X)

        else:
            y_pred = np.zeros((len(y_mod),num_classes))
            if sklearn.__version__ < '0.18':
                skf = StratifiedKFold(y_mod, n_folds=self.cv,shuffle=True)
            else:
                skf = StratifiedKFold(n_splits=self.cv, shuffle=True).split(X, y)
            for idx, (train_idx, test_idx) in enumerate(skf):
                if verbose:
                    print("training fold {} of {}".format(idx+1, self.cv))
                X_train = np.array(X)[train_idx,:]
                X_test = np.array(X)[test_idx,:]
                y_train = np.array(y_mod)[train_idx]
                # We could also copy the model first and then fit it
                this_estimator = clone(self.base_estimator)
                this_estimator.fit(X_train,y_train)
                y_pred[test_idx,:] = this_estimator.predict_proba(X_test)
            
            if verbose:
                print("Training Full Model")
            self.uncalibrated_classifier = clone(self.base_estimator)
            self.uncalibrated_classifier.fit(X, y_mod)

        # calibrating function
        if verbose:
            print("Determining Calibration Function")
        if self.method == 'logistic':
            self.calib_func, self.cf_list = prob_calibration_function_multiclass(
                y_mod, 
                self.pre_transform(y_pred), 
                verbose=verbose, 
                **self.calib_kwargs)
        if self.method == 'ridge':
            self.calib_func, self.cf_list = prob_calibration_function_multiclass(
                y_mod, 
                self.pre_transform(y_pred), 
                verbose=verbose, 
                method='ridge', 
                **self.calib_kwargs)
        # training full model

        return self


    def predict_proba(self, X):
        """Posterior probabilities of classification
        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.
        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """
        # check_is_fitted(self, ["classes_", "calibrated_classifier"])
        if self.fit_on_multiclass:
            return self.calib_func(self.pre_transform(
                self.uncalibrated_classifier.predict_proba(X)))
        
        col_1 = self.calib_func(self.pre_transform(
            self.uncalibrated_classifier.predict_proba(X)[:,1]))
        col_0 = 1-col_1
        return np.vstack((col_0,col_1)).T
                  
    def predict(self, X):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.
        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        # check_is_fitted(self, ["classes_", "calibrated_classifier"])
        return self.uncalibrated_classifier.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def classes_(self):
        return self.uncalibrated_classifier.classes_

    
class SplineCalibratedClassifierMulticlassCV(BaseEstimator, ClassifierMixin):
    """Probability calibration using cubic splines.
    With this class, the base_estimator is fit on each of the cross-validation
    training set folds in order to generate scores on the (cross-validated)
    test set folds.  The test set scores are accumulated into a final vector
    (the size of the full set) which is used to calibrate the answers.
    The model is then fit on the full data set.  The predict, and predict_proba
    methods are then updated to use the combination of the predictions from the 
    full model and the calibration function computed as above.
    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. If cv='prefit', the
        classifier must have been fit already on data.
    method : 'logistic' or 'ridge'
        The default is 'logistic', which is best if you plan to use log-loss as your
        performance metric.  This method is relatively robust and will typically do
        well on brier score as well.  The 'ridge' method calibrates using an L2 loss,
        and therefore should do better for brier score, but may do considerably worse
        on log-loss.
    cv : integer, cross-validation generator, iterable or "prefit", optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - 'prefit', if you wish to use the data only for calibration
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
        neither binary nor multiclass, :class:`sklearn.model_selection.KFold`
        is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        If "prefit" is passed, it is assumed that base_estimator has been
        fitted already and all data is used for calibration.
    Attributes
    ----------
    uncalibrated_classifier: this gives the uncalibrated version of the classifier, fit on the entire data set
    calib_func: this is the calibration function that has been learned from the cross-validation.  Applying this function
     to the results of the uncalibrated classifier (via model.predict_proba(X_test)[:,1]) gives the fully calibrated classifier
    References
    ----------
   """
    def __init__(self, base_estimator=None, method='logistic', cv=5, **calib_kwargs):
        self.base_estimator = base_estimator
        self.uncalibrated_classifier = None
        self.calib_func = None
        self.method = method
        self.cv = cv
        self.calib_kwargs = calib_kwargs

    def fit(self, X, y, verbose=False):
        """Fit the calibrated model
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        Returns
        -------
        self : object
            Returns an instance of self.
        """
        class_list = np.unique(y)
        num_classes = len(class_list)
        y_mod = np.zeros(len(y))

        for i in range(num_classes):
            y_mod[np.where(y==class_list[i])]=i

        y_mod = y_mod.astype(int)
        if ((type(self.cv)==str) and (self.cv=='prefit')):
            self.uncalibrated_classifier = self.base_estimator
            y_pred = self.uncalibrated_classifier.predict_proba(X)[:,1]

        else:
            y_pred = np.zeros((len(y_mod),num_classes))
            if sklearn.__version__ < '0.18':
                skf = StratifiedKFold(y_mod, n_folds=self.cv,shuffle=True)
            else:
                skf = StratifiedKFold(n_splits=self.cv, shuffle=True).split(X, y)
            for idx, (train_idx, test_idx) in enumerate(skf):
                if verbose:
                    print("training fold {} of {}".format(idx+1, self.cv))
                X_train = np.array(X)[train_idx,:]
                X_test = np.array(X)[test_idx,:]
                y_train = np.array(y_mod)[train_idx]
                # We could also copy the model first and then fit it
                this_estimator = clone(self.base_estimator)
                this_estimator.fit(X_train,y_train)
                y_pred[test_idx,:] = this_estimator.predict_proba(X_test)
            
            if verbose:
                print("Training Full Model")
            self.uncalibrated_classifier = clone(self.base_estimator)
            self.uncalibrated_classifier.fit(X, y_mod)

        # calibrating function
        if verbose:
            print("Determining Calibration Function")
        if self.method=='logistic':
            self.calib_func = prob_calibration_function_multiclass(y_mod, 
                                                                   y_pred, 
                                                                   verbose=verbose, 
                                                                   **self.calib_kwargs)
        if self.method=='ridge':
            self.calib_func = prob_calibration_function_multiclass(y_mod, 
                                                                   y_pred, 
                                                                   verbose=verbose, 
                                                                   method='ridge', 
                                                                   **self.calib_kwargs)
        # training full model

        return self

    def predict_proba(self, X):
        """Posterior probabilities of classification
        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.
        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """
        # check_is_fitted(self, ["classes_", "calibrated_classifier"])
        return self.calib_func(self.uncalibrated_classifier.predict_proba(X))


    def predict(self, X):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.
        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        # check_is_fitted(self, ["classes_", "calibrated_classifier"])
        return self.uncalibrated_classifier.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def classes_(self):
        return self.uncalibrated_classifier.classes_