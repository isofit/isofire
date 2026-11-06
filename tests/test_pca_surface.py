"""Synthetic checks of the spaceborne solver: a smooth fake atmosphere, a PCA basis
with water-band gaps, and pixels generated with the same forward model.
Run: uv run pytest tests/test_pca_surface.py -q
"""
import numpy as np
import pytest

from isofire.pca_surface import PCASurfaceBasis, planck_stable, ratio_temperature, run_pca_fire_retrieval
from isofire.retrieval import BAD_RANGES_NM_DEFAULT, MixtureRetrievalConfig, make_bitwise_bad_range_mask_1d, ranges_to_mask

WL = np.linspace(381, 2493, 285)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rng = np.random.default_rng(0)
    good = make_bitwise_bad_range_mask_1d(WL, np.array([[1340, 1450], [1800, 1960]]))
    X = np.array([rng.uniform(.02, .4) + rng.uniform(-.2, .2) * (WL - 1400) / 1100
                  + rng.uniform(0, .15) * np.exp(-((WL - 800) / 150) ** 2) for _ in range(3000)])[:, good]
    mu = X.mean(0)
    V = np.linalg.eigh(np.cov(X, rowvar=False))[1][:, ::-1].T[:8]
    S = (X - mu) @ V.T
    M = np.full(285, np.nan); M[good] = mu
    C = np.full((8, 285), np.nan); C[:, good] = V
    f = tmp_path_factory.mktemp("b") / "basis.npz"
    np.savez(f, wavelengths=WL, good_bands=good, mean=M, components=C, prior_mean=S.mean(0),
             prior_cov=np.cov(S, rowvar=False), approach="test")
    basis = PCASurfaceBasis.from_npz(str(f), WL)
    t = 0.85 - 0.6 * np.exp(-((WL - 1380) / 40) ** 2) - 0.6 * np.exp(-((WL - 1880) / 50) ** 2)
    tu = np.sqrt(t) * 0.95
    k = lambda v: (lambda x: np.asarray(v)[None, :])
    rte = dict(rhoatm=k(0.05 * np.exp(-(WL - 400) / 300)), sphalb=k(0.1 * np.exp(-(WL - 400) / 500)),
               transm_up_dir=k(tu), transm_up_dif=k(0 * tu), transm_down_dir=k(tu), transm_down_dif=k(0 * tu))
    solar = 150 * np.exp(-((WL - 500) / 700) ** 2) + 20
    bad = make_bitwise_bad_range_mask_1d(WL, BAD_RANGES_NM_DEFAULT)
    win = lambda r: bad & ranges_to_mask(WL, r)
    return dict(rng=rng, basis=basis, rte=rte, tu=tu, solar=solar,
                surf=win([[350, 2500]]), hot=win([[900, 1800]]), smol=win([[1200, 2500]]))


def _run(s, obs, **kw):
    n = len(obs)
    return run_pca_fire_retrieval(obs, np.full(n, 1.5), np.ones_like(obs, bool), cfg=kw.pop("cfg", MixtureRetrievalConfig()),
                                  rte=s["rte"], solar_irr=s["solar"], wl_nm=WL, surface_1d=s["surf"], hot_1d=s["hot"],
                                  smolder_1d=s["smol"], eta_lo=np.zeros(n, np.float32), eta_hi=np.ones(n, np.float32), **kw)


def test_day_single_temperature(setup):
    s, b, mu0 = setup, setup["basis"], 0.8
    truth = [(1e-3, 1000.), (3e-3, 800.), (2e-4, 1500.)]
    obs, rfl = [], []
    for a, T in truth:
        r = b.mean + s["rng"].multivariate_normal(b.prior_mean, b.prior_cov) @ b.components
        r[~b.valid_1d] = 0.2
        base = s["solar"] * mu0 / np.pi * (s["rte"]["rhoatm"](0)[0] + s["tu"] ** 2 * r / (1 - s["rte"]["sphalb"](0)[0] * r))
        obs.append((base + a * np.asarray(planck_stable(T, WL)) * s["tu"]) * (1 + 0.005 * s["rng"].standard_normal(285)))
        rfl.append(r)
    o = _run(s, np.array(obs), mu0=mu0, basis=b)
    for k, (a, T) in enumerate(truth):
        assert abs(o["T1"][k] - T) < 60
        assert np.nanmean(np.abs(o["rfl"][k][b.valid_1d] - rfl[k][b.valid_1d])) < 0.005
        assert o["chi2_stage2"][k] < 2.0


def test_night_flares(setup):
    s = setup
    truth = [(1e-4, 1800.), (5e-3, 1150.), (2e-5, 1900.)]
    obs = np.array([a * np.asarray(planck_stable(T, WL)) * s["tu"] * (1 + .01 * s["rng"].standard_normal(285))
                    + .003 * s["rng"].standard_normal(285) for a, T in truth])
    hi, lo = np.argmin(abs(WL - 2249.4)), np.argmin(abs(WL - 1096.9))
    T0 = ratio_temperature(obs[:, hi], obs[:, lo], WL[hi], WL[lo])
    la0 = np.log10(obs[:, hi] / (np.array([float(planck_stable(t, WL[hi])) for t in T0]) * s["tu"][hi]))
    o = _run(s, obs, mu0=1.0, basis=None, include_solar=False, T1_init=T0, log_alpha1_init=la0,
             cfg=MixtureRetrievalConfig(mu_hot_max_T=2400., Tmax=2700.))
    for k, (a, T) in enumerate(truth):
        assert abs(o["mu"][k] - T) < 60
        assert abs(np.log10(o["alpha2"][k] / a)) < 0.2
