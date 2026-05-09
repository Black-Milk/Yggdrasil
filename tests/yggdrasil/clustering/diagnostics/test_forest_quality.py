import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier

from yggdrasil.clustering import DiscriminativeForestEmbedding
from yggdrasil.clustering.diagnostics.forest_quality import (
    discriminator_oob_auc,
    is_kernel_informative,
)


@pytest.fixture
def fitted_separable_embedding():
    X, _ = make_blobs(n_samples=80, centers=4, n_features=4, cluster_std=0.4, random_state=0)
    emb = DiscriminativeForestEmbedding(n_estimators=80, oob_score=True, random_state=0).fit(X)
    return emb


@pytest.fixture
def fitted_noise_embedding():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(150, 4))
    emb = DiscriminativeForestEmbedding(n_estimators=80, oob_score=True, random_state=0).fit(X)
    return emb


def test_oob_auc_high_on_separable_blobs(fitted_separable_embedding):
    auc = discriminator_oob_auc(
        fitted_separable_embedding.forest_, fitted_separable_embedding.y_disc_
    )

    assert 0.7 <= auc <= 1.0


def test_oob_auc_near_half_on_uniform_noise(fitted_noise_embedding):
    auc = discriminator_oob_auc(fitted_noise_embedding.forest_, fitted_noise_embedding.y_disc_)

    assert auc < 0.6


def test_oob_auc_raises_when_oob_score_disabled():
    X, y = make_blobs(n_samples=40, centers=2, n_features=3, random_state=0)
    clf = RandomForestClassifier(n_estimators=5, oob_score=False, random_state=0).fit(X, y)

    with pytest.raises(AttributeError, match="oob_score"):
        discriminator_oob_auc(clf, y)


def test_oob_auc_rejects_multiclass_classifier():
    X, y = make_blobs(n_samples=60, centers=3, n_features=3, random_state=0)
    clf = RandomForestClassifier(
        n_estimators=20, oob_score=True, bootstrap=True, random_state=0
    ).fit(X, y)

    with pytest.raises(ValueError, match="binary"):
        discriminator_oob_auc(clf, y)


def test_oob_auc_rejects_y_length_mismatch(fitted_separable_embedding):
    with pytest.raises(ValueError, match="length"):
        discriminator_oob_auc(
            fitted_separable_embedding.forest_,
            np.zeros(fitted_separable_embedding.y_disc_.shape[0] - 1),
        )


def test_is_kernel_informative_threshold():
    assert is_kernel_informative(0.92) is True
    assert is_kernel_informative(0.55) is True
    assert is_kernel_informative(0.50) is False
    assert is_kernel_informative(float("nan")) is False
