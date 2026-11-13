import numpy as np
from scipy.optimize import check_grad

import routers as R


def _toy(E=200, J=5, m=3, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(E, J))
    u = P[:, 1] + 0.1 * rng.normal(size=E)                 # expert 2 is the good one
    A = rng.normal(size=(E, J, m)); A[:, 1] *= 0.1
    rho = rng.uniform(-1, 1, size=(E, m))
    skill = np.array([0, -1.0, 0.3, 0.5, 0.2])
    Phi, C = R.audit_features(P, A, rho, skill)
    return P, u, A, rho, skill, Phi, C


def test_router_gradient_matches_finite_difference():
    P, u, A, rho, skill, Phi, C = _toy()
    avail = np.array([True] * 5)
    w = np.ones(len(u)) / len(u)
    x0 = np.random.default_rng(1).normal(size=5 + 7 + 25) * 0.1
    f = lambda x: R.router_loss_grad(x, Phi, C, P, u, w, avail)[0]
    g = lambda x: R.router_loss_grad(x, Phi, C, P, u, w, avail)[1]
    assert check_grad(f, g, x0) < 1e-5


def test_router_learns_the_better_expert_and_unavailable_expert_gets_no_weight():
    P, u, A, rho, skill, Phi, C = _toy()
    avail = np.array([True, True, True, True, False])
    rec = R.Records(Phi, C, P, u, np.ones(len(u)), avail)
    params = R.fit_router(rec)
    w, Pp, f = R.router_predict(params, Phi, C, P)
    assert (w[:, 4] == 0).all()
    assert np.abs(f - u).mean() < np.abs(P[:, 0] - u).mean()
    assert w[:, 1].mean() > 0.3


def test_feature_shapes_and_invariance_to_support_permutation():
    P, u, A, rho, skill, Phi, C = _toy(E=50)
    Phi2, C2 = R.audit_features(P, A[:, :, ::-1], rho[:, ::-1], skill)
    np.testing.assert_allclose(Phi, Phi2, atol=1e-12)
    np.testing.assert_allclose(C, C2, atol=1e-12)
    assert Phi.shape == (50, 5, 7) and C.shape == (50, 5, 5)


def test_aligned_features_canonical_order_under_joint_permutation_including_ties():
    rng = np.random.default_rng(3)
    E, J, m = 60, 7, 3
    P = rng.normal(size=(E, J)); A = rng.normal(size=(E, J, m))
    rho = rng.uniform(-1, 1, size=(E, m)); rho[:10, 1] = rho[:10, 0]           # exact |rho| ties
    rho[5:10, 1] *= -1
    r = rng.normal(size=(E, m)); r[:10, 1] = r[:10, 0]                         # tie on residual as well
    fp, pr, q = rng.normal(size=E), rng.normal(size=E), rng.integers(0, 5, size=E)
    base = R.aligned_features(P, A, rho, r, fp, pr, q)
    for perm in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        out = R.aligned_features(P, A[:, :, perm], rho[:, perm], r[:, perm], fp, pr, q)
        np.testing.assert_allclose(base, out, atol=1e-12)
    assert base.shape[1] == J + 1 + 2 * m + 3 * J * m + 3


def test_laplace_mixture_median_properties():
    mu = np.array([[0.0, 0.0], [-1.0, 3.0], [2.0, 2.0]])
    a = np.full((3, 2), 0.5)
    d = np.ones((3, 2))
    M = R.laplace_mixture_median(mu, a, d)
    np.testing.assert_allclose(M[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(M[1], 1.0, atol=1e-6)         # symmetric mixture
    np.testing.assert_allclose(M[2], 2.0, atol=1e-8)
    assert ((M >= mu.min(axis=1) - 1e-9) & (M <= mu.max(axis=1) + 1e-9)).all()
    # heavier weight pulls the median
    M2 = R.laplace_mixture_median(np.array([[0.0, 4.0]]), np.array([[0.9, 0.1]]), np.ones((1, 2)))
    assert M2[0] < 1.0


def test_group_equal_mass_and_stack_records():
    g = np.array(["a"] * 6 + ["b"] * 2)
    w = R.group_equal_mass(g)
    np.testing.assert_allclose([w[g == "a"].sum(), w[g == "b"].sum()], [0.5, 0.5])
