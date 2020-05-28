[![Build Status](https://travis-ci.org/eloyfelix/nonconformist.svg?branch=master)](https://travis-ci.org/eloyfelix/nonconformist)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](http://mybinder.org/badge.svg)](http://beta.mybinder.org/v2/gh/eloyfelix/nonconformist/master?filepath=README.ipynb)

# cbl-nonconformist

This is a fork of [nonconformist](https://github.com/donlnz/nonconformist) keeping it up to date with new Python / scikit-learn versions.

Python implementation of the conformal prediction framework [1].

Primarily to be used as an extension to the scikit-learn library.

# Installation

## Dependencies

cbl-nonconformist requires:

* Python >= 3.6
* scikit-learn >= 0.20
* numpy
* scipy

## User installation

```bash
git clone https://github.com/eloyfelix/cbl-nonconformist
cd cbl-nonconformist/
python setup.py install
```


# TODO

* Exchangeability testing [2].
* Interpolated p-values [3,4].
* Conformal prediction trees [5].
* Venn predictors [?]
* Venn-ABERS predictors [?]
* Nonparametric distribution prediction [?]

[1] Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning in a random world.
Springer Science & Business Media.

[2] Fedorova, V., Gammerman, A., Nouretdinov, I., & Vovk, V. (2012).
Plug-in martingales for testing exchangeability on-line. In Proceedings
of the 29th International Conference on Machine Learning (ICML-12) (pp. 1639-1646).

[3] Carlsson, L., Ahlberg, E., Boström, H., Johansson, U., Linusson, & H. (2015).
Modifications to p-values of Conformal Predictors. In Proceedings of the 3rd
International Symposium on Statistical Learning and Data Sciences (SLDS 2015). (In press).

[4] Johansson, U., Ahlberg, E., Boström, H., Carlsson, L., Linusson, H., Sönströd, C. (2015).
Handling Small Calibration Sets in Mondrian Inductive Conformal Regressors. In Proceedings of
the 3rd International Symposium on Statistical Learning and Data Sciences (SLDS 2015). (In press).

[5] Johansson, U., Sönströd, C., Linusson, H., & Boström, H. (2014, October).
Regression trees for streaming data with local performance guarantees.
In Big Data (Big Data), 2014 IEEE International Conference on (pp. 461-470). IEEE.