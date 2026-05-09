"""Forest-level diagnostics about how informative the leaf kernel is.

The discriminator OOB-AUC answers a different question from the
spectrum-shape diagnostics: it asks whether the random-vs-synthetic
classifier has learned anything at all. When AUC ≈ 0.5 the resulting
leaf kernel is essentially random co-occurrence noise and any
clustering on it should be reported with low confidence regardless of
spectral structure. The selector uses this signal as an early gate.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

__all__ = [
    "discriminator_oob_auc",
    "is_kernel_informative",
]

_ForestClassifier = RandomForestClassifier | ExtraTreesClassifier


def discriminator_oob_auc(
    classifier: _ForestClassifier,
    y_true: np.ndarray,
) -> float:
    """Compute the out-of-bag AUC of a fitted real-vs-synthetic forest.

    Reads ``classifier.oob_decision_function_`` (which exists when the
    forest was fit with ``oob_score=True`` and ``bootstrap=True``) and
    scores it against ``y_true``. The OOB-AUC is a pessimistic
    estimate of how separable the real samples are from the synthetic
    ones under the trained forest; it is the right "is the forest
    doing anything?" signal for the leaf kernel.

    Parameters
    ----------
    classifier : RandomForestClassifier or ExtraTreesClassifier
        A fitted scikit-learn forest classifier with two classes and
        ``oob_decision_function_`` populated.
    y_true : array-like of shape (n_samples,)
        Binary class labels in the same order as the rows of
        ``classifier.oob_decision_function_``. Values must match the
        classifier's two ``classes_`` entries.

    Returns
    -------
    auc : float
        Out-of-bag ROC AUC in ``[0, 1]``. Values near ``0.5`` indicate
        the discriminator failed to learn anything useful; values near
        ``1`` indicate a sharp real-vs-synthetic boundary and a
        well-resolved leaf kernel. Returns ``nan`` if no rows have
        valid OOB predictions or only one class is present after
        masking NaN rows.

    Raises
    ------
    AttributeError
        If the classifier does not expose ``oob_decision_function_``.
    ValueError
        If the classifier was not trained on a binary task or if
        ``y_true`` has the wrong length.

    Notes
    -----
    When some samples never appear out-of-bag, scikit-learn fills the
    corresponding rows of ``oob_decision_function_`` with NaN. Those
    rows are dropped before AUC is computed so the metric is defined
    on the OOB-evaluable subset.
    """
    if not hasattr(classifier, "oob_decision_function_"):
        raise AttributeError(
            "classifier was not fit with oob_score=True; oob_decision_function_ is unavailable."
        )

    classes = np.asarray(classifier.classes_)
    if classes.size != 2:
        raise ValueError(
            f"discriminator_oob_auc expects a binary classifier; got {classes.size} classes."
        )

    oob_scores = np.asarray(classifier.oob_decision_function_, dtype=np.float64)
    if oob_scores.ndim != 2 or oob_scores.shape[1] != 2:
        raise ValueError(
            f"oob_decision_function_ must be 2-D with two columns; got shape {oob_scores.shape}."
        )

    y_arr = np.asarray(y_true).ravel()
    if y_arr.shape[0] != oob_scores.shape[0]:
        raise ValueError(
            "y_true length must match oob_decision_function_ rows; "
            f"got {y_arr.shape[0]} and {oob_scores.shape[0]}."
        )

    valid = ~np.any(np.isnan(oob_scores), axis=1)
    if not np.any(valid):
        return float("nan")

    pos_class = classes[1]
    pos_scores = oob_scores[:, 1]
    y_bin = (y_arr == pos_class).astype(np.int64)
    if np.unique(y_bin[valid]).size < 2:
        return float("nan")
    return float(roc_auc_score(y_bin[valid], pos_scores[valid]))


def is_kernel_informative(auc: float, threshold: float = 0.55) -> bool:
    """Return whether an OOB-AUC value clears an informativeness threshold.

    Parameters
    ----------
    auc : float
        Discriminator out-of-bag AUC; see :func:`discriminator_oob_auc`.
    threshold : float, default=0.55
        Minimum AUC required to consider the leaf kernel informative.
        The default is calibrated against the random-vs-uniform-noise
        case where AUC concentrates around ``0.5`` with small
        finite-sample fluctuations.

    Returns
    -------
    informative : bool
        ``True`` when ``auc >= threshold`` and ``auc`` is finite,
        ``False`` otherwise.

    Examples
    --------
    >>> from yggdrasil.clustering.diagnostics import is_kernel_informative
    >>> is_kernel_informative(0.92)
    True
    >>> is_kernel_informative(0.50)
    False
    """
    return bool(np.isfinite(auc) and auc >= threshold)
