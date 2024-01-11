import lightgbm
from lightgbm import LGBMModel
from lightgbm.compat import LGBMNotFittedError, _LGBMClassifierBase, _LGBMLabelEncoder
from lightgbm.callback import _EvalResultDict, record_evaluation
from loss_fn import FocalLoss
import numpy as np
from scipy import special


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

    def fit(self, X, y, eval_set=None, categorical_feature=None, callbacks=None):

        self._le = _LGBMLabelEncoder().fit(y)
        self._classes = self._le.classes_
        
        params = self._process_params(stage="fit")

        self.init_score = FocalLoss(alpha=self.obj_alpha, gamma=self.obj_gamma).init_score(y)
        init_scores = np.full_like(y, self.init_score, dtype=float)
        train_set = lightgbm.Dataset(data=X, label=y, init_score=init_scores, categorical_feature=categorical_feature)

        valid_sets = []
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, valid_data in enumerate(eval_set):
                if valid_data[0] is X and valid_data[1] is y:
                    valid_set = train_set
                else:
                    valid_init_scores = np.full_like(valid_data[1], self.init_score, dtype=float)
                    valid_set = lightgbm.Dataset(data=valid_data[0], label=valid_data[1], init_score=valid_init_scores, categorical_feature=categorical_feature)
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
        self._best_iteration = self._Booster.best_iteration
        self._best_score = self._Booster.best_score
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

    @property
    def classes_(self) -> np.ndarray:
        """:obj:`array` of shape = [n_classes]: The class label array."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No classes found. Need to call fit beforehand.')
        return self._classes  # type: ignore[return-value]
  