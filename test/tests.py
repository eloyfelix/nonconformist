import unittest 

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, load_diabetes, load_boston
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.base import OobClassifierAdapter, OobRegressorAdapter

from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler, CrossSampler, RandomSubSampler
from nonconformist.acp import BootstrapConformalClassifier
from nonconformist.acp import CrossConformalClassifier

from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.icp import OobCpClassifier, OobCpRegressor

from nonconformist.nc import MarginErrFunc
from nonconformist.nc import ClassifierNc, RegressorNc, RegressorNormalizer
from nonconformist.nc import AbsErrorErrFunc, SignErrorErrFunc
from nonconformist.cp import TcpClassifier

from nonconformist.evaluation import cross_val_score
from nonconformist.evaluation import ClassIcpCvHelper, RegIcpCvHelper
from nonconformist.evaluation import class_avg_c, class_mean_errors
from nonconformist.evaluation import reg_mean_errors, reg_median_size


class TestNonconformist(unittest.TestCase):
    """
    Tests that only check that examples are not broken, need to write new tests for it.
    """

    def test_acp_classification_tree(self):

        # -----------------------------------------------------------------------------
        # Experiment setup
        # -----------------------------------------------------------------------------
        data = load_iris()

        idx = np.random.permutation(data.target.size)
        train = idx[: int(2 * idx.size / 3)]
        test = idx[int(2 * idx.size / 3) :]

        truth = data.target[test].reshape(-1, 1)
        columns = ["C-{}".format(i) for i in np.unique(data.target)] + ["truth"]
        significance = 0.1

        # -----------------------------------------------------------------------------
        # Define models
        # -----------------------------------------------------------------------------

        models = {
            "ACP-RandomSubSampler": AggregatedCp(
                IcpClassifier(ClassifierNc(ClassifierAdapter(DecisionTreeClassifier()))),
                RandomSubSampler(),
            ),
            "ACP-CrossSampler": AggregatedCp(
                IcpClassifier(ClassifierNc(ClassifierAdapter(DecisionTreeClassifier()))),
                CrossSampler(),
            ),
            "ACP-BootstrapSampler": AggregatedCp(
                IcpClassifier(ClassifierNc(ClassifierAdapter(DecisionTreeClassifier()))),
                BootstrapSampler(),
            ),
            "CCP": CrossConformalClassifier(
                IcpClassifier(ClassifierNc(ClassifierAdapter(DecisionTreeClassifier())))
            ),
            "BCP": BootstrapConformalClassifier(
                IcpClassifier(ClassifierNc(ClassifierAdapter(DecisionTreeClassifier())))
            ),
        }

        # -----------------------------------------------------------------------------
        # Train, predict and evaluate
        # -----------------------------------------------------------------------------
        for name, model in models.items():
            model.fit(data.data[train, :], data.target[train])
            prediction = model.predict(data.data[test, :], significance=significance)
            table = np.hstack((prediction, truth))
            df = pd.DataFrame(table, columns=columns)
            print("\n{}".format(name))
            print("Error rate: {}".format(class_mean_errors(prediction, truth, significance)))
            print(df)

        self.assertTrue(True)



    def test_acp_regression_tree(self):
        # -----------------------------------------------------------------------------
        # Experiment setup
        # -----------------------------------------------------------------------------
        data = load_diabetes()

        idx = np.random.permutation(data.target.size)
        train = idx[: int(2 * idx.size / 3)]
        test = idx[int(2 * idx.size / 3) :]

        truth = data.target[test]
        columns = ["min", "max", "truth"]
        significance = 0.1

        # -----------------------------------------------------------------------------
        # Define models
        # -----------------------------------------------------------------------------

        models = {
            "ACP-RandomSubSampler": AggregatedCp(
                IcpRegressor(RegressorNc(RegressorAdapter(DecisionTreeRegressor()))),
                RandomSubSampler(),
            ),
            "ACP-CrossSampler": AggregatedCp(
                IcpRegressor(RegressorNc(RegressorAdapter(DecisionTreeRegressor()))),
                CrossSampler(),
            ),
            "ACP-BootstrapSampler": AggregatedCp(
                IcpRegressor(RegressorNc(RegressorAdapter(DecisionTreeRegressor()))),
                BootstrapSampler(),
            ),
        }

        # -----------------------------------------------------------------------------
        # Train, predict and evaluate
        # -----------------------------------------------------------------------------
        for name, model in models.items():
            model.fit(data.data[train, :], data.target[train])
            prediction = model.predict(data.data[test, :])
            prediction_sign = model.predict(data.data[test, :], significance=significance)
            table = np.vstack((prediction_sign.T, truth)).T
            df = pd.DataFrame(table, columns=columns)
            print("\n{}".format(name))
            print("Error rate: {}".format(reg_mean_errors(prediction, truth, significance)))
            print(df)


    def test_confidence_credibility(self):
        
        data = load_iris()
        x, y = data.data, data.target

        for i, y_ in enumerate(np.unique(y)):
            y[y == y_] = i

        n_instances = y.size
        idx = np.random.permutation(n_instances)

        train_idx = idx[: int(n_instances / 3)]
        cal_idx = idx[int(n_instances / 3) : 2 * int(n_instances / 3)]
        test_idx = idx[2 * int(n_instances / 3) :]

        nc = ClassifierNc(ClassifierAdapter(RandomForestClassifier()))
        icp = IcpClassifier(nc)

        icp.fit(x[train_idx, :], y[train_idx])
        icp.calibrate(x[cal_idx, :], y[cal_idx])

        print(
            pd.DataFrame(
                icp.predict_conf(x[test_idx, :]), columns=["Label", "Confidence", "Credibility"]
            )
        )

    def test_cross_validation(self):
        # -----------------------------------------------------------------------------
        # Classification
        # -----------------------------------------------------------------------------
        data = load_iris()

        icp = IcpClassifier(
            ClassifierNc(
                ClassifierAdapter(RandomForestClassifier(n_estimators=100)), MarginErrFunc()
            )
        )
        icp_cv = ClassIcpCvHelper(icp)

        scores = cross_val_score(
            icp_cv,
            data.data,
            data.target,
            iterations=5,
            folds=5,
            scoring_funcs=[class_mean_errors, class_avg_c],
            significance_levels=[0.05, 0.1, 0.2],
        )

        print("Classification: iris")
        scores = scores.drop(["fold", "iter"], axis=1)
        print(scores.groupby(["significance"]).mean())

        # -----------------------------------------------------------------------------
        # Regression, absolute error
        # -----------------------------------------------------------------------------
        data = load_diabetes()

        icp = IcpRegressor(
            RegressorNc(
                RegressorAdapter(RandomForestRegressor(n_estimators=100)), AbsErrorErrFunc()
            )
        )
        icp_cv = RegIcpCvHelper(icp)

        scores = cross_val_score(
            icp_cv,
            data.data,
            data.target,
            iterations=5,
            folds=5,
            scoring_funcs=[reg_mean_errors, reg_median_size],
            significance_levels=[0.05, 0.1, 0.2],
        )

        print("Absolute error regression: diabetes")
        scores = scores.drop(["fold", "iter"], axis=1)
        print(scores.groupby(["significance"]).mean())

        # -----------------------------------------------------------------------------
        # Regression, normalized absolute error
        # -----------------------------------------------------------------------------
        data = load_diabetes()

        underlying_model = RegressorAdapter(RandomForestRegressor(n_estimators=100))
        normalizer_model = RegressorAdapter(RandomForestRegressor(n_estimators=100))
        normalizer = RegressorNormalizer(underlying_model, normalizer_model, AbsErrorErrFunc())
        nc = RegressorNc(underlying_model, AbsErrorErrFunc(), normalizer)

        icp = IcpRegressor(nc)
        icp_cv = RegIcpCvHelper(icp)

        scores = cross_val_score(
            icp_cv,
            data.data,
            data.target,
            iterations=5,
            folds=5,
            scoring_funcs=[reg_mean_errors, reg_median_size],
            significance_levels=[0.05, 0.1, 0.2],
        )

        print("Normalized absolute error regression: diabetes")
        scores = scores.drop(["fold", "iter"], axis=1)
        print(scores.groupby(["significance"]).mean())

        # -----------------------------------------------------------------------------
        # Regression, normalized signed error
        # -----------------------------------------------------------------------------
        data = load_diabetes()

        icp = IcpRegressor(
            RegressorNc(
                RegressorAdapter(RandomForestRegressor(n_estimators=100)), SignErrorErrFunc()
            )
        )
        icp_cv = RegIcpCvHelper(icp)

        scores = cross_val_score(
            icp_cv,
            data.data,
            data.target,
            iterations=5,
            folds=5,
            scoring_funcs=[reg_mean_errors, reg_median_size],
            significance_levels=[0.05, 0.1, 0.2],
        )

        print("Signed error regression: diabetes")
        scores = scores.drop(["fold", "iter"], axis=1)
        print(scores.groupby(["significance"]).mean())

        # -----------------------------------------------------------------------------
        # Regression, signed error
        # -----------------------------------------------------------------------------
        data = load_diabetes()

        underlying_model = RegressorAdapter(RandomForestRegressor(n_estimators=100))
        normalizer_model = RegressorAdapter(RandomForestRegressor(n_estimators=100))

        # The normalization model can use a different error function than is
        # used to measure errors on the underlying model
        normalizer = RegressorNormalizer(underlying_model, normalizer_model, AbsErrorErrFunc())
        nc = RegressorNc(underlying_model, SignErrorErrFunc(), normalizer)

        icp = IcpRegressor(nc)
        icp_cv = RegIcpCvHelper(icp)

        scores = cross_val_score(
            icp_cv,
            data.data,
            data.target,
            iterations=5,
            folds=5,
            scoring_funcs=[reg_mean_errors, reg_median_size],
            significance_levels=[0.05, 0.1, 0.2],
        )

        print("Normalized signed error regression: diabetes")
        scores = scores.drop(["fold", "iter"], axis=1)
        print(scores.groupby(["significance"]).mean())


    def test_icp_classification_tree(self):
        # -----------------------------------------------------------------------------
        # Setup training, calibration and test indices
        # -----------------------------------------------------------------------------
        data = load_iris()

        idx = np.random.permutation(data.target.size)
        train = idx[: int(idx.size / 3)]
        calibrate = idx[int(idx.size / 3) : int(2 * idx.size / 3)]
        test = idx[int(2 * idx.size / 3) :]

        # -----------------------------------------------------------------------------
        # Train and calibrate
        # -----------------------------------------------------------------------------
        icp = IcpClassifier(
            ClassifierNc(ClassifierAdapter(DecisionTreeClassifier()), MarginErrFunc())
        )
        icp.fit(data.data[train, :], data.target[train])
        icp.calibrate(data.data[calibrate, :], data.target[calibrate])

        # -----------------------------------------------------------------------------
        # Predict
        # -----------------------------------------------------------------------------
        prediction = icp.predict(data.data[test, :], significance=0.1)
        header = np.array(["c0", "c1", "c2", "Truth"])
        table = np.vstack([prediction.T, data.target[test]]).T
        df = pd.DataFrame(np.vstack([header, table]))
        print(df)


    def test_icp_regression_tree(self):
        # -----------------------------------------------------------------------------
        # Setup training, calibration and test indices
        # -----------------------------------------------------------------------------
        data = load_boston()

        idx = np.random.permutation(data.target.size)
        train = idx[: int(idx.size / 3)]
        calibrate = idx[int(idx.size / 3) : int(2 * idx.size / 3)]
        test = idx[int(2 * idx.size / 3) :]

        # -----------------------------------------------------------------------------
        # Without normalization
        # -----------------------------------------------------------------------------
        # Train and calibrate
        # -----------------------------------------------------------------------------
        underlying_model = RegressorAdapter(DecisionTreeRegressor(min_samples_leaf=5))
        nc = RegressorNc(underlying_model, AbsErrorErrFunc())
        icp = IcpRegressor(nc)
        icp.fit(data.data[train, :], data.target[train])
        icp.calibrate(data.data[calibrate, :], data.target[calibrate])

        # -----------------------------------------------------------------------------
        # Predict
        # -----------------------------------------------------------------------------
        prediction = icp.predict(data.data[test, :], significance=0.1)
        header = ["min", "max", "truth", "size"]
        size = prediction[:, 1] - prediction[:, 0]
        table = np.vstack([prediction.T, data.target[test], size.T]).T
        df = pd.DataFrame(table, columns=header)
        print(df)

        # -----------------------------------------------------------------------------
        # With normalization
        # -----------------------------------------------------------------------------
        # Train and calibrate
        # -----------------------------------------------------------------------------
        underlying_model = RegressorAdapter(DecisionTreeRegressor(min_samples_leaf=5))
        normalizing_model = RegressorAdapter(KNeighborsRegressor(n_neighbors=1))
        normalizer = RegressorNormalizer(underlying_model, normalizing_model, AbsErrorErrFunc())
        nc = RegressorNc(underlying_model, AbsErrorErrFunc(), normalizer)
        icp = IcpRegressor(nc)
        icp.fit(data.data[train, :], data.target[train])
        icp.calibrate(data.data[calibrate, :], data.target[calibrate])

        # -----------------------------------------------------------------------------
        # Predict
        # -----------------------------------------------------------------------------
        prediction = icp.predict(data.data[test, :], significance=0.1)
        header = ["min", "max", "truth", "size"]
        size = prediction[:, 1] - prediction[:, 0]
        table = np.vstack([prediction.T, data.target[test], size.T]).T
        df = pd.DataFrame(table, columns=header)
        print(df)


    def test_oob_calibration(self):
        # -----------------------------------------------------------------------------
        # Classification
        # -----------------------------------------------------------------------------
        data = load_iris()

        icp = OobCpClassifier(
            ClassifierNc(
                OobClassifierAdapter(RandomForestClassifier(n_estimators=100, oob_score=True))
            )
        )
        icp_cv = ClassIcpCvHelper(icp)

        scores = cross_val_score(
            icp_cv,
            data.data,
            data.target,
            iterations=5,
            folds=5,
            scoring_funcs=[class_mean_errors, class_avg_c],
            significance_levels=[0.05, 0.1, 0.2],
        )

        print("Classification: iris")
        scores = scores.drop(["fold", "iter"], axis=1)
        print(scores.groupby(["significance"]).mean())

        # -----------------------------------------------------------------------------
        # Regression, absolute error
        # -----------------------------------------------------------------------------
        data = load_diabetes()

        icp = OobCpRegressor(
            RegressorNc(
                OobRegressorAdapter(RandomForestRegressor(n_estimators=100, oob_score=True))
            )
        )
        icp_cv = RegIcpCvHelper(icp)

        scores = cross_val_score(
            icp_cv,
            data.data,
            data.target,
            iterations=5,
            folds=5,
            scoring_funcs=[reg_mean_errors, reg_median_size],
            significance_levels=[0.05, 0.1, 0.2],
        )

        print("Absolute error regression: diabetes")
        scores = scores.drop(["fold", "iter"], axis=1)
        print(scores.groupby(["significance"]).mean())


    def test_tcp_classification_svm(self):
        # -----------------------------------------------------------------------------
        # Setup training, calibration and test indices
        # -----------------------------------------------------------------------------
        data = load_iris()

        idx = np.random.permutation(data.target.size)
        train = idx[: int(idx.size / 2)]
        test = idx[int(idx.size / 2) :]

        # -----------------------------------------------------------------------------
        # Train and calibrate
        # -----------------------------------------------------------------------------
        tcp = TcpClassifier(
            ClassifierNc(ClassifierAdapter(SVC(probability=True)), MarginErrFunc())
        )
        tcp.fit(data.data[train, :], data.target[train])

        # -----------------------------------------------------------------------------
        # Predict
        # -----------------------------------------------------------------------------
        prediction = tcp.predict(data.data[test, :], significance=0.1)
        header = np.array(["c0", "c1", "c2", "Truth"])
        table = np.vstack([prediction.T, data.target[test]]).T
        df = pd.DataFrame(np.vstack([header, table]))
        print(df)