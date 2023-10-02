import lightgbm
from lightgbm import LGBMModel
from lightgbm.compat import LGBMNotFittedError, _LGBMClassifierBase
from lightgbm.callback import _EvalResultDict, record_evaluation

import numpy as np
from scipy import optimize
from scipy import special


class FocalLoss:

    def __init__(self, gamma=0, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better


class CustomLGBMClassifier(_LGBMClassifierBase, LGBMModel):
    """" Implementation of LGBMClassifier with Focal Loss 
    """

    def __is_fitted__(self):
        return getattr(self, "fitted_", False)

    def get_params(self, deep: bool = True):
        
        params = super().get_params(deep=deep)

        if hasattr(self, "obj_alpha"):
            params.update({"obj_alpha": self.obj_alpha})
        if hasattr(self, "obj_gamma"):
            params.update({"obj_gamma": self.obj_gamma})        
     
        return params

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
            self._other_params[key] = value

        return self
        
    def _process_params(self, stage):

        assert stage in {"fit", "predict"}
        params = self.get_params()

        if "obj_alpha" not in params.keys():
            self.obj_alpha = params["obj_alpha"] = None
        if "obj_gamma" not in params.keys():
            self.obj_gamma = params["obj_gamma"] = 0

        if stage == "fit":
            params["objective"] = FocalLoss(alpha=params["obj_alpha"], gamma=params["obj_gamma"]).lgb_obj
            params["random_seed"] = 0
               
        self.eval_metric = FocalLoss(alpha=params["obj_alpha"], gamma=params["obj_gamma"]).lgb_eval

        return params

    def fit(self, X, y, eval_set=None, callbacks=None):
        
        params = self._process_params(stage="fit")

        self.init_score = FocalLoss(alpha=self.obj_alpha, gamma=self.obj_gamma).init_score(y)
        init_scores = np.full_like(y, self.init_score, dtype=float)
        train_set = lightgbm.Dataset(data=X, label=y, init_score=init_scores)

        valid_sets = []
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, valid_data in enumerate(eval_set):
                if valid_data[0] is X and valid_data[1] is y:
                    valid_set = train_set
                else:
                    valid_init_scores = np.full_like(valid_data[1], self.init_score, dtype=float)
                    valid_set = lightgbm.Dataset(data=valid_data[0], label=valid_data[1], init_score=valid_init_scores)
                valid_sets.append(valid_set)

        if callbacks is None:
            callbacks = []

        evals_result: _EvalResultDict = {}
        callbacks.append(record_evaluation(evals_result))
        
        self._Booster = lightgbm.train(
            params=params,
            train_set=train_set,
            valid_sets=valid_sets,
            feval=self.eval_metric,
            callbacks=callbacks)
        
        self._evals_result = evals_result
        self.fitted_ = True

        return self
    
    def predict_proba(self, X):

        result = self._Booster.predict(X)
        preds = special.expit(self.init_score + result)
        # return preds
        return np.vstack((1.-preds, preds)).transpose()
    
    @property
    def classes_(self) -> np.ndarray:
        """:obj:`array` of shape = [n_classes]: The class label array."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No classes found. Need to call fit beforehand.')
        return self._classes  # type: ignore[return-value]
  