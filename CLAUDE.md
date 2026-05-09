# CLAUDE.md

Operating instructions for any Claude session working in this repository. These rules are binding; follow them on every change.

## 1. Project context

**Yggdrasil** is a Python library of scikit-learn-compatible tree ensembles. The package lives under `src/yggdrasil/` and is organized by estimator family (e.g. `src/yggdrasil/clustering/`). Every public estimator must conform to the scikit-learn estimator API so it composes cleanly with `Pipeline`, `GridSearchCV`, and `cross_validate`.

Authoritative external reference: <https://scikit-learn.org/stable/developers/develop.html>. When this file and that page disagree, **prefer the upstream page** and update this file.

## 2. Environment & toolchain — `uv` only

The project is pinned to **Python 3.14.4** via `.python-version`, with `requires-python = ">=3.14"` in `pyproject.toml`.

**Never** run `pip`, `python -m venv`, `python setup.py`, `pip install -e .`, or activate `.venv` manually in commands you suggest or execute. Always use `uv`:

| Task | Command |
| --- | --- |
| Install / refresh environment from lockfile | `uv sync` |
| Add a runtime dependency | `uv add <pkg>` |
| Add a dev dependency | `uv add --dev <pkg>` |
| Remove a dependency | `uv remove <pkg>` |
| Refresh lockfile after a manual `pyproject.toml` edit | `uv lock` |
| Run any command in the project env | `uv run <cmd>` (e.g. `uv run pytest`, `uv run ruff check .`) |

Do **not** invent version pins — let `uv add` resolve the version. Always commit `uv.lock` after any dependency change.

## 3. Code style & linting

Run both before declaring a change complete:

```bash
uv run ruff format .
uv run ruff check --fix .
```

Configuration that is already in `pyproject.toml` and must be respected:

- Line length **100**, target **`py314`**.
- Active rule sets: `E, F, W, I, B, UP, N, SIM, RUF`.
- Per-file ignores already in place:
  - `src/yggdrasil/*`: `N803`, `N806`, `RUF012` — uppercase identifiers like `X`, `Xt`, `Y` are allowed, and class-level mutable defaults (e.g. sklearn-style `_parameter_constraints = {...}`) are allowed.
  - `tests/*`: `N803`, `N806`.

Additional style rules (from PEP 8 and the scikit-learn coding guidelines):

- **Absolute imports only.** Never `import *`.
- One statement per line. Always break after `if` / `for` / `while`.
- Use `_` to separate words in non-class names: `n_samples`, not `nsamples`.
- Prefer `np.asarray` / `sklearn.utils.check_array` over `np.asanyarray` or `np.atleast_2d` (these let `np.matrix` through and have surprising semantics).

## 4. Docstrings — numpydoc with shape-aware annotation

All public modules, classes, and functions get a **numpydoc**-style docstring (<https://numpydoc.readthedocs.io/en/latest/format.html>). Use these sections, in this order, only when applicable: `Parameters`, `Attributes`, `Returns`, `Yields`, `Raises`, `See Also`, `Notes`, `References`, `Examples`.

### Where things go on an estimator class

- Constructor (`__init__`) arguments are documented in the **class** docstring under **Parameters**, not under `__init__`.
- Learned, public attributes (those ending in `_`) are documented under **Attributes** on the class docstring.
- Leading-underscore attributes (e.g. `_intermediate_coefs`) are **not** documented in the public docstring.

### Shape-aware annotation (mandatory)

Every array-valued parameter, return value, and fitted attribute must declare its **dtype family and shape**, using scikit-learn's conventions and named axes:

| Situation | Annotation form |
| --- | --- |
| 2-D feature matrix | `array-like of shape (n_samples, n_features)` |
| 1-D target / labels | `array-like of shape (n_samples,)` (note trailing comma) |
| Multi-output target | `array-like of shape (n_samples,) or (n_samples, n_outputs)` |
| Sample weights | `array-like of shape (n_samples,), default=None` |
| Sparse-accepting input | `{array-like, sparse matrix} of shape (n_samples, n_features)` |
| Concrete output array | `ndarray of shape (n_samples,)` |
| Class labels attribute | `ndarray of shape (n_classes,)` |
| Tree / ensemble structure | `ndarray of shape (n_estimators,)` or document the precise shape |
| Scalar fitted attribute | `int` / `float` (no shape) |
| Optional / conditional shape | use `or` (e.g. `ndarray of shape (n_samples,) or (n_samples, n_classes)`) |
| Random-state parameter | `int, RandomState instance or None, default=None` and link `:term:\`Glossary <random_state>\`` |

Use the same axis names that scikit-learn uses (`n_samples`, `n_features`, `n_classes`, `n_outputs`, `n_components`, `n_estimators`, `n_features_in_`). Do not invent new axis names without documenting them in **Notes**.

### `docstring-code-format = true`

`pyproject.toml` enables `[tool.ruff.format] docstring-code-format = true`. This means **`uv run ruff format` rewrites code inside docstrings**, including `>>>` doctest blocks and fenced code blocks. Practical rules for any example you put in a docstring:

- It must be **syntactically valid Python** so ruff can parse it. Pseudo-code and partial snippets go in **Notes** as prose, not in code blocks.
- Prefer `>>>` doctest style for short, self-contained examples (matches scikit-learn).
- Keep example lines short (~75 chars) so ruff doesn't have to wrap them awkwardly to the 100-char limit.
- Imports inside an example are fine and encouraged for runnable examples.

## 5. Scikit-learn estimator conventions

These are non-negotiable for any class meant to be used as an estimator. Every public estimator must pass `sklearn.utils.estimator_checks.check_estimator` (see section 6).

### Inheritance order

Mixins on the **left**, `BaseEstimator` on the **right**, so MRO resolves correctly:

```python
class MyClusterer(ClusterMixin, BaseEstimator): ...
class MyClassifier(ClassifierMixin, BaseEstimator): ...
class MyRegressor(RegressorMixin, BaseEstimator): ...
class MyTransformer(TransformerMixin, BaseEstimator): ...
```

For transformers that produce named output columns, also inherit from `OneToOneFeatureMixin` (one-to-one input/output) or `ClassNamePrefixFeaturesOutMixin` (generated names).

### `__init__` rules

- Every parameter is a keyword argument with a default, unless there is genuinely no sane default (rare; in this codebase, prefer always-defaulted).
- **No logic, no validation, no mutation.** The body is just `self.param = param` for every argument.
- The instance attribute name **must equal** the constructor argument name.
- No mutable defaults (`list`, `dict`, etc.); if a mutable parameter is needed, copy it inside `fit`.
- **Never** set a trailing-underscore (learned) attribute in `__init__`.

### `fit` rules

- Signature: `fit(self, X, y=None, sample_weight=None, ...)`. Even unsupervised estimators accept `y=None` in second position and ignore it (so they compose in `Pipeline`).
- First validation step: `X, y = validate_data(self, X, y, ...)` from `sklearn.utils.validation`. This sets `n_features_in_` and, for dataframes, `feature_names_in_`.
- Validate hyperparameters here (not in `__init__`).
- Calling `fit` again must overwrite all learned state, unless `warm_start=True` is supported.
- Always `return self`.

### Learned state

- Public learned attributes end in `_` (e.g. `labels_`, `classes_`, `coef_`, `n_features_in_`, `feature_names_in_`, `random_state_`).
- Internal-only learned attributes start with `_` and are not documented in the public docstring.

### `predict` / `transform` / `score` / `predict_proba` / `decision_function`

First two lines are always:

```python
check_is_fitted(self)
X = validate_data(self, X, reset=False)
```

### Family-specific contracts

- **`ClassifierMixin`** — set `self.classes_` (e.g. via `self.classes_, y = np.unique(y, return_inverse=True)`); `predict` must return values drawn from `self.classes_`. Implement `predict_proba` / `decision_function` / `predict_log_proba` when supported.
- **`RegressorMixin`** — accept numeric `y`; default `score` is R².
- **`ClusterMixin`** — set `self.labels_` (cluster index per training sample); accept and ignore `y`. Optionally implement `predict` for new samples.
- **`TransformerMixin`** — implement `transform` (and optionally `fit_transform` if it can be more efficient). Implement `get_feature_names_out` so `set_output(transform="pandas"|"polars")` works.

### Random number handling

Estimators that use randomness:

1. Take `random_state` as an `__init__` keyword (default `None`).
2. Store it **unmodified** as `self.random_state`.
3. In `fit`, do `self.random_state_ = check_random_state(self.random_state)` and use `self.random_state_` thereafter.

Never call `numpy.random.random()` or seed the global RNG.

### Estimator tags

When defaults don't fit (e.g. multi-output, non-deterministic, sparse support, requires positive `X`), override `__sklearn_tags__`, **always calling `super` first**:

```python
def __sklearn_tags__(self):
    tags = super().__sklearn_tags__()
    tags.target_tags.single_output = False
    tags.non_deterministic = True
    return tags
```

### Custom fitted-check

Only implement `__sklearn_is_fitted__` when the default trailing-underscore heuristic is wrong (rare).

### Utilities to use

From `sklearn.utils` / `sklearn.utils.validation`: `validate_data`, `check_array`, `check_is_fitted`, `check_random_state`, `unique_labels`. From `sklearn.utils._testing` (in tests only): `assert_allclose`.

## 6. Testing

Run with `uv run pytest`.

Layout: tests mirror the source tree under `tests/yggdrasil/<family>/test_<estimator>.py`. Tests import from the **public** path (`from yggdrasil.clustering import Foo`), never from internal modules.

Every public estimator needs:

1. A common-checks test using `parametrize_with_checks`:

   ```python
   from sklearn.utils.estimator_checks import parametrize_with_checks
   from yggdrasil.clustering import MyClusterer

   @parametrize_with_checks([MyClusterer()])
   def test_sklearn_compatible(estimator, check):
       check(estimator)
   ```

2. Targeted unit tests for behavior specific to the estimator (correctness on known data, edge cases, hyperparameter effects).

For float-array equality use `sklearn.utils._testing.assert_allclose`; provide a non-zero `atol` when comparing arrays of zeros.

## 7. Repository layout convention

```text
src/yggdrasil/
    __init__.py              # re-export top-level public API
    <family>/
        __init__.py          # re-export the family's public estimators
        <estimator>.py       # one estimator (or tightly-coupled set) per file
tests/yggdrasil/
    <family>/
        test_<estimator>.py
```

Public estimators are re-exported from `src/yggdrasil/<family>/__init__.py` and, when top-level, from `src/yggdrasil/__init__.py`.

## 8. Definition of done

A change is not done until **all** of the following hold:

- [ ] `uv sync` succeeds; `uv.lock` is updated and committed if dependencies changed.
- [ ] `uv run ruff format .` produces no diff.
- [ ] `uv run ruff check .` passes.
- [ ] `uv run pytest` passes, including `parametrize_with_checks` for any new or modified estimator.
- [ ] Every new or modified public symbol has a numpydoc docstring with shape-aware annotations on all array parameters, return values, and fitted attributes.
- [ ] Every example in a docstring is valid Python and survives `uv run ruff format` unchanged.

## Appendix: template estimator skeleton

Use this as the starting point for any new estimator. It demonstrates the mandatory inheritance order, no-logic `__init__`, `validate_data` / `check_is_fitted` placement, trailing-underscore learned attributes, `random_state` handling, and a numpydoc docstring with shape-aware annotations and a doctest example.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import (
    check_is_fitted,
    check_random_state,
    validate_data,
)


class TemplateClusterer(ClusterMixin, BaseEstimator):
    """One-line summary of the estimator.

    Longer description of what the estimator does, the algorithm it
    implements, and any important caveats.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of initialization. Pass an int for
        reproducible results across multiple calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster index assigned to each training sample.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of the learned cluster centers.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during :meth:`fit`. Defined only when `X`
        has feature names that are all strings.
    random_state_ : RandomState
        The random number generator instantiated from ``random_state``.

    Examples
    --------
    >>> import numpy as np
    >>> from yggdrasil.clustering import TemplateClusterer
    >>> X = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
    >>> est = TemplateClusterer(n_clusters=2, random_state=0).fit(X)
    >>> est.labels_.shape
    (4,)
    """

    def __init__(self, n_clusters=8, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Present for API consistency with supervised estimators.

        Returns
        -------
        self : TemplateClusterer
            Fitted estimator.
        """
        X = validate_data(self, X)
        self.random_state_ = check_random_state(self.random_state)
        # ... real fitting logic here ...
        self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))
        self.labels_ = np.zeros(X.shape[0], dtype=np.int64)
        return self

    def predict(self, X):
        """Assign each sample in `X` to the nearest cluster.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to assign.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster index for each sample.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        # ... real prediction logic here ...
        return np.zeros(X.shape[0], dtype=np.int64)
```

Swap `ClusterMixin` for `ClassifierMixin` / `RegressorMixin` / `TransformerMixin` and adjust the learned attributes (`classes_`, `coef_`, etc.) according to section 5.
