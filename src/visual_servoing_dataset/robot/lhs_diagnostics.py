"""Diagnósticos estatísticos e substituição equilibrada (Space-Filling) para a família P7 (LHS 6-DoF)."""

from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rich.table import Table
from scipy.spatial.distance import pdist
from scipy.stats import binomtest, ks_2samp, kstest, qmc, spearmanr, wasserstein_distance

from ..utils.logger import console, logger

AXIS_NAMES: Tuple[str, ...] = ("x", "y", "z", "rx", "ry", "rz")

DEFAULT_BALANCE_THRESHOLDS: Dict[str, float] = {
    "ks_alpha": 0.05,
    "max_abs_pearson": 0.30,
    "max_rms_pearson": 0.16,
    "min_stratum_coverage": 0.85,
    "max_mean_deviation": 0.08,
    "max_mc_cd2_percentile": 15.0,
    "min_pairwise_dist_ratio_vs_orig": 0.70,
    "ks_2samp_alpha": 0.05,
}


def _axis_label(idx: int) -> str:
    return AXIS_NAMES[idx] if idx < len(AXIS_NAMES) else f"axis_{idx}"


def _log_mean_exp(values: np.ndarray) -> float:
    max_val = float(np.max(values))
    return max_val + float(np.log(np.mean(np.exp(values - max_val))))


@lru_cache(maxsize=32)
def _monte_carlo_cd2_baseline(
    n_samples: int,
    dim: int,
    trials: int = 500,
    seed: int = 20260101,
) -> Tuple[float, float, float, Tuple[float, ...]]:
    """Calcula a distribuição de referência de CD2 sob amostragem aleatória uniforme (Monte Carlo)."""
    if n_samples < 2 or dim < 1 or trials < 1:
        return 0.0, 0.0, 0.0, tuple()

    rng = np.random.default_rng(seed)
    vals = np.empty(trials, dtype=np.float64)
    upper_lim = 1.0 - 1e-12
    for t in range(trials):
        sample = np.clip(rng.random((n_samples, dim)), 0.0, upper_lim)
        vals[t] = float(qmc.discrepancy(sample, method="CD"))

    vals_sorted = np.sort(vals)
    mean_v = float(np.mean(vals_sorted))
    std_v = float(np.std(vals_sorted, ddof=1)) if trials > 1 else 0.0
    p05_v = float(np.percentile(vals_sorted, 5.0))
    return mean_v, std_v, p05_v, tuple(float(v) for v in vals_sorted)


def _safe_correlation_matrices(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula matrizes de correlação de Pearson e Spearman de forma segura para qualquer N."""
    n, dim = u.shape
    eye = np.eye(dim, dtype=np.float64)
    if n < 2:
        return eye.copy(), eye.copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = np.corrcoef(u, rowvar=False)
    pearson = np.asarray(pearson, dtype=np.float64)
    if pearson.ndim == 0:
        pearson = eye.copy()
    pearson = np.nan_to_num(pearson, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(pearson, 1.0)

    if n < 3:
        return pearson, pearson.copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        sp_res = spearmanr(u, axis=0)
    sp_raw = getattr(sp_res, "statistic", sp_res[0] if isinstance(sp_res, tuple) else sp_res)
    if dim == 2 and np.ndim(sp_raw) == 0:
        val = float(0.0 if np.isnan(sp_raw) else sp_raw)
        spearman = np.array([[1.0, val], [val, 1.0]], dtype=np.float64)
    else:
        spearman = np.asarray(sp_raw, dtype=np.float64)
        if spearman.shape != (dim, dim):
            spearman = pearson.copy()
        spearman = np.nan_to_num(spearman, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(spearman, 1.0)

    return pearson, spearman


def evaluate_lhs_sample_set(
    u_samples: Sequence[Sequence[float]],
    mc_baseline_trials: int = 500,
    mc_seed: int = 20260101,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Avalia estatisticamente um conjunto normalizado U in [0, 1]^{N x d} da família P7."""
    th = dict(DEFAULT_BALANCE_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    u = np.asarray(u_samples, dtype=np.float64)
    if u.ndim == 1:
        u = u.reshape(1, -1) if u.size > 0 else np.zeros((0, 6), dtype=np.float64)
    n_samples = int(u.shape[0])
    dim = int(u.shape[1]) if u.ndim == 2 and u.shape[1] > 0 else 6

    if n_samples < 2:
        return {
            "n_samples": n_samples,
            "dim": dim,
            "valid": False,
            "thresholds_used": th,
            "discrepancy": None,
            "marginal_per_axis": {},
            "marginal_summary": None,
            "orthogonality": None,
            "space_filling": None,
            "quality_checks": {
                "sample_size_sufficient": False,
                "ks_uniformity_ok": False,
                "orthogonality_ok": False,
                "stratum_coverage_ok": False,
                "mean_centering_ok": False,
                "space_filling_cd2_ok": False,
            },
            "balanced": False,
        }

    u_clipped = np.clip(u, 0.0, 1.0 - 1e-12)
    cd2 = float(qmc.discrepancy(u_clipped, method="CD"))
    wd2 = float(qmc.discrepancy(u_clipped, method="WD"))
    mc_mean, mc_std, mc_p05, mc_sorted = _monte_carlo_cd2_baseline(
        n_samples=n_samples, dim=dim, trials=mc_baseline_trials, seed=mc_seed
    )
    if mc_sorted:
        mc_arr = np.asarray(mc_sorted, dtype=np.float64)
        cd2_percentile = float(np.mean(mc_arr <= cd2) * 100.0)
        cd2_zscore = float((cd2 - mc_mean) / mc_std) if mc_std > 1e-12 else 0.0
    else:
        cd2_percentile = 0.0
        cd2_zscore = 0.0

    # Uniformidade marginal, balanço de sinais e cobertura de estratos por eixo
    theo_std = 1.0 / math.sqrt(12.0)
    marginal_per_axis: Dict[str, Dict[str, Any]] = {}
    ks_pvals: List[float] = []
    ks_stats: List[float] = []
    cov_ratios: List[float] = []
    mean_devs: List[float] = []
    sign_pvals: List[float] = []

    for j in range(dim):
        col = u[:, j]
        axis_name = _axis_label(j)
        col_mean = float(np.mean(col))
        mean_dev = abs(col_mean - 0.5)
        col_std = float(np.std(col, ddof=1)) if n_samples > 1 else 0.0
        neg_cnt = int(np.sum(col < 0.5))
        pos_cnt = int(n_samples - neg_cnt)
        sign_pval = float(binomtest(neg_cnt, n=n_samples, p=0.5).pvalue)
        ks_res = kstest(col, "uniform")
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)
        strata_idx = np.clip(np.floor(col * n_samples).astype(int), 0, n_samples - 1)
        unique_strata = int(len(np.unique(strata_idx)))
        cov_ratio = float(unique_strata / n_samples)

        marginal_per_axis[axis_name] = {
            "mean": round(col_mean, 6),
            "mean_deviation_from_05": round(mean_dev, 6),
            "std": round(col_std, 6),
            "theoretical_uniform_std": round(theo_std, 6),
            "min_u": round(float(np.min(col)), 6),
            "max_u": round(float(np.max(col)), 6),
            "negative_count": neg_cnt,
            "positive_count": pos_cnt,
            "sign_balance_pvalue": round(sign_pval, 6),
            "ks_stat": round(ks_stat, 6),
            "ks_pvalue": round(ks_pval, 6),
            "unique_strata": unique_strata,
            "stratum_coverage_ratio": round(cov_ratio, 6),
        }
        ks_pvals.append(ks_pval)
        ks_stats.append(ks_stat)
        cov_ratios.append(cov_ratio)
        mean_devs.append(mean_dev)
        sign_pvals.append(sign_pval)

    min_ks_pval = float(min(ks_pvals))
    max_ks_stat = float(max(ks_stats))
    min_stratum_cov = float(min(cov_ratios))
    mean_stratum_cov = float(np.mean(cov_ratios))
    max_mean_dev = float(max(mean_devs))
    min_sign_pval = float(min(sign_pvals))

    # Ortogonalidade entre eixos
    pearson_mat, spearman_mat = _safe_correlation_matrices(u)
    triu_i, triu_j = np.triu_indices(dim, k=1)
    if len(triu_i) > 0:
        off_p = pearson_mat[triu_i, triu_j]
        off_s = spearman_mat[triu_i, triu_j]
        abs_off_p = np.abs(off_p)
        worst_idx = int(np.argmax(abs_off_p))
        max_abs_pearson = float(abs_off_p[worst_idx])
        rms_pearson = float(np.sqrt(np.mean(off_p ** 2)))
        worst_pair = {
            "axis_a": _axis_label(int(triu_i[worst_idx])),
            "axis_b": _axis_label(int(triu_j[worst_idx])),
            "pearson_r": round(float(off_p[worst_idx]), 6),
        }
        max_abs_spearman = float(np.max(np.abs(off_s)))
        rms_spearman = float(np.sqrt(np.mean(off_s ** 2)))
    else:
        max_abs_pearson = 0.0
        rms_pearson = 0.0
        worst_pair = None
        max_abs_spearman = 0.0
        rms_spearman = 0.0

    if n_samples > dim:
        col_stds = np.std(u, axis=0, ddof=1)
        safe_stds = np.where(col_stds > 1e-12, col_stds, 1.0)
        u_std = (u - np.mean(u, axis=0, keepdims=True)) / safe_stds
        cond_val = float(np.linalg.cond(u_std))
        cond_number = cond_val if math.isfinite(cond_val) else 1e6
    else:
        cond_number = 1.0

    # Espaçamento geométrico 6D (Maximin, phi_p e MaxPro)
    dists = pdist(u, metric="euclidean")
    min_pair_dist = float(np.min(dists))
    mean_pair_dist = float(np.mean(dists))
    safe_dists = np.maximum(dists, 1e-12)
    log_phi_terms = -10.0 * np.log(safe_dists)
    phi_p10 = float(np.exp((_log_mean_exp(log_phi_terms) + math.log(len(safe_dists))) / 10.0))

    pair_diff_logs = []
    for i in range(n_samples - 1):
        diffs = np.abs(u[i + 1 :] - u[i])
        diffs = np.maximum(diffs, 1e-6)
        pair_diff_logs.append(-2.0 * np.sum(np.log(diffs), axis=1))
    all_pair_logs = np.concatenate(pair_diff_logs)
    maxpro_criterion = float(np.exp(_log_mean_exp(all_pair_logs) / float(dim)))

    sample_size_ok = bool(n_samples >= 6)
    ks_ok = bool(min_ks_pval >= th["ks_alpha"])
    ortho_ok = bool(
        max_abs_pearson <= th["max_abs_pearson"] and rms_pearson <= th["max_rms_pearson"]
    )
    stratum_ok = bool(min_stratum_cov >= th["min_stratum_coverage"])
    centering_ok = bool(max_mean_dev <= th["max_mean_deviation"])
    cd2_ok = bool(cd2_percentile <= th["max_mc_cd2_percentile"])

    quality_checks = {
        "sample_size_sufficient": sample_size_ok,
        "ks_uniformity_ok": ks_ok,
        "orthogonality_ok": ortho_ok,
        "stratum_coverage_ok": stratum_ok,
        "mean_centering_ok": centering_ok,
        "space_filling_cd2_ok": cd2_ok,
    }
    balanced = bool(sample_size_ok and all(quality_checks.values()))

    return {
        "n_samples": n_samples,
        "dim": dim,
        "valid": True,
        "thresholds_used": th,
        "discrepancy": {
            "centered_l2_cd2": round(cd2, 8),
            "wrap_around_wd2": round(wd2, 8),
            "mc_baseline_trials": int(mc_baseline_trials),
            "mc_cd2_mean": round(mc_mean, 8),
            "mc_cd2_std": round(mc_std, 8),
            "mc_cd2_p05": round(mc_p05, 8),
            "cd2_percentile_vs_mc": round(cd2_percentile, 2),
            "cd2_zscore_vs_mc": round(cd2_zscore, 4),
        },
        "marginal_per_axis": marginal_per_axis,
        "marginal_summary": {
            "min_ks_pvalue": round(min_ks_pval, 6),
            "max_ks_stat": round(max_ks_stat, 6),
            "min_stratum_coverage": round(min_stratum_cov, 6),
            "mean_stratum_coverage": round(mean_stratum_cov, 6),
            "max_mean_deviation": round(max_mean_dev, 6),
            "min_sign_balance_pvalue": round(min_sign_pval, 6),
        },
        "orthogonality": {
            "pearson_matrix": [[round(float(v), 6) for v in row] for row in pearson_mat],
            "spearman_matrix": [[round(float(v), 6) for v in row] for row in spearman_mat],
            "max_abs_pearson": round(max_abs_pearson, 6),
            "rms_pearson": round(rms_pearson, 6),
            "worst_pearson_pair": worst_pair,
            "max_abs_spearman": round(max_abs_spearman, 6),
            "rms_spearman": round(rms_spearman, 6),
            "condition_number": round(cond_number, 6),
        },
        "space_filling": {
            "min_pairwise_dist": round(min_pair_dist, 6),
            "mean_pairwise_dist": round(mean_pair_dist, 6),
            "phi_p10": round(phi_p10, 6),
            "maxpro_criterion": round(maxpro_criterion, 6),
        },
        "quality_checks": quality_checks,
        "balanced": balanced,
    }


def compare_lhs_original_vs_final(
    u_original: Sequence[Sequence[float]],
    u_final: Sequence[Sequence[float]],
    mc_baseline_trials: int = 500,
    mc_seed: int = 20260101,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compara o conjunto LHS original contra o conjunto final após eventuais substituições."""
    th = dict(DEFAULT_BALANCE_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    orig_metrics = evaluate_lhs_sample_set(
        u_original, mc_baseline_trials=mc_baseline_trials, mc_seed=mc_seed, thresholds=th
    )
    final_metrics = evaluate_lhs_sample_set(
        u_final, mc_baseline_trials=mc_baseline_trials, mc_seed=mc_seed, thresholds=th
    )

    u_orig_arr = np.asarray(u_original, dtype=np.float64)
    u_fin_arr = np.asarray(u_final, dtype=np.float64)
    if u_orig_arr.ndim == 1 and u_orig_arr.size == 0:
        u_orig_arr = np.zeros((0, 6), dtype=np.float64)
    if u_fin_arr.ndim == 1 and u_fin_arr.size == 0:
        u_fin_arr = np.zeros((0, 6), dtype=np.float64)

    has_replacements = bool(
        u_orig_arr.shape != u_fin_arr.shape or not np.allclose(u_orig_arr, u_fin_arr, atol=1e-7)
    )

    comparison: Optional[Dict[str, Any]] = None
    if orig_metrics["valid"] and final_metrics["valid"]:
        dim = min(orig_metrics["dim"], final_metrics["dim"])
        per_axis_shift: Dict[str, Dict[str, float]] = {}
        ks2_pvals: List[float] = []
        wass_vals: List[float] = []

        for j in range(dim):
            axis_name = _axis_label(j)
            c_orig = u_orig_arr[:, j]
            c_fin = u_fin_arr[:, j]
            ks2 = ks_2samp(c_orig, c_fin)
            w1 = float(wasserstein_distance(c_orig, c_fin))
            ks2_p = float(ks2.pvalue)
            per_axis_shift[axis_name] = {
                "ks_2samp_stat": round(float(ks2.statistic), 6),
                "ks_2samp_pvalue": round(ks2_p, 6),
                "wasserstein_1d": round(w1, 6),
                "delta_mean": round(float(np.mean(c_fin) - np.mean(c_orig)), 6),
                "delta_std": round(float(np.std(c_fin, ddof=1) - np.std(c_orig, ddof=1)), 6),
            }
            ks2_pvals.append(ks2_p)
            wass_vals.append(w1)

        orig_cd2 = orig_metrics["discrepancy"]["centered_l2_cd2"]
        fin_cd2 = final_metrics["discrepancy"]["centered_l2_cd2"]
        delta_cd2 = fin_cd2 - orig_cd2
        delta_cd2_pct = (delta_cd2 / orig_cd2 * 100.0) if orig_cd2 > 1e-12 else 0.0

        delta_rho = (
            final_metrics["orthogonality"]["max_abs_pearson"]
            - orig_metrics["orthogonality"]["max_abs_pearson"]
        )

        orig_dmin = orig_metrics["space_filling"]["min_pairwise_dist"]
        fin_dmin = final_metrics["space_filling"]["min_pairwise_dist"]
        delta_dmin = fin_dmin - orig_dmin
        dmin_ratio = (fin_dmin / orig_dmin) if orig_dmin > 1e-12 else 1.0

        orig_mp = orig_metrics["space_filling"]["maxpro_criterion"]
        fin_mp = final_metrics["space_filling"]["maxpro_criterion"]
        delta_mp = fin_mp - orig_mp
        delta_mp_pct = (delta_mp / orig_mp * 100.0) if orig_mp > 1e-12 else 0.0

        min_ks2_p = float(min(ks2_pvals))
        max_w1 = float(max(wass_vals))
        no_shift_ok = bool(min_ks2_p >= th["ks_2samp_alpha"])
        dmin_ok = bool(dmin_ratio >= th["min_pairwise_dist_ratio_vs_orig"])

        comparison = {
            "per_axis_shift": per_axis_shift,
            "min_ks_2samp_pvalue": round(min_ks2_p, 6),
            "max_wasserstein_1d": round(max_w1, 6),
            "delta_cd2": round(delta_cd2, 8),
            "delta_cd2_pct": round(delta_cd2_pct, 3),
            "delta_max_abs_pearson": round(delta_rho, 6),
            "delta_min_pairwise_dist": round(delta_dmin, 6),
            "min_pairwise_dist_ratio": round(dmin_ratio, 6),
            "delta_maxpro": round(delta_mp, 6),
            "delta_maxpro_pct": round(delta_mp_pct, 3),
            "no_distribution_shift_ok": no_shift_ok,
            "min_dist_preserved_ok": dmin_ok,
        }

    overall_balanced = bool(final_metrics["balanced"])
    if has_replacements:
        if comparison is None:
            overall_balanced = False
        else:
            overall_balanced = bool(
                overall_balanced
                and comparison["no_distribution_shift_ok"]
                and comparison["min_dist_preserved_ok"]
            )

    return {
        "has_replacements": has_replacements,
        "original_metrics": orig_metrics,
        "final_metrics": final_metrics,
        "comparison": comparison,
        "overall_balanced": overall_balanced,
    }


def select_optimal_lhs_replacement(
    u_active: Sequence[Sequence[float]],
    u_rejected: Optional[Sequence[Sequence[float]]] = None,
    target_n: int = 30,
    seed: int = 1043,
    pool_size: int = 512,
    exclusion_radius: float = 0.15,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Seleciona uma amostra substituta em [0, 1]^d que preserva o equilíbrio global do conjunto
    (Min-CD2 + Maximin + baixa correlação + cobertura de estratos vazios), evitando regiões rejeitadas.
    """
    active = np.asarray(u_active, dtype=np.float64)
    if active.ndim == 1:
        active = active.reshape(1, -1)
    n_active, dim = active.shape
    n_bins = max(int(target_n), n_active + 1, 2)
    rng = np.random.default_rng(seed)

    # Identificar estratos vazios por eixo em [0, n_bins - 1]
    vacant_bins_per_axis: List[np.ndarray] = []
    for j in range(dim):
        occupied = set(np.clip(np.floor(active[:, j] * n_bins).astype(int), 0, n_bins - 1).tolist())
        vacant = np.array([b for b in range(n_bins) if b not in occupied], dtype=int)
        if len(vacant) == 0:
            vacant = np.arange(n_bins, dtype=int)
        vacant_bins_per_axis.append(vacant)

    # Gerar pool híbrido determinístico: Metade A guiada por estratos vazios + Metade B LHS uniforme
    n_a = max(1, pool_size // 2)
    n_b = max(1, pool_size - n_a)
    pool_a = np.empty((n_a, dim), dtype=np.float64)
    n_strict = n_a // 2
    for i in range(n_a):
        for j in range(dim):
            vb = vacant_bins_per_axis[j]
            b_chosen = int(vb[int(rng.integers(0, len(vb)))])
            if i >= n_strict:
                b_chosen = int(np.clip(b_chosen + int(rng.integers(-1, 2)), 0, n_bins - 1))
            u_val = (b_chosen + float(rng.random())) / float(n_bins)
            pool_a[i, j] = min(max(u_val, 0.0), 1.0 - 1e-9)

    lhs_pool_sampler = qmc.LatinHypercube(d=dim, seed=seed)
    pool_b = np.clip(lhs_pool_sampler.random(n=n_b), 0.0, 1.0 - 1e-9)
    pool = np.vstack([pool_a, pool_b])

    # Filtrar candidatos longe das poses rejeitadas
    rejected = (
        np.asarray(u_rejected, dtype=np.float64)
        if u_rejected is not None and len(u_rejected) > 0
        else np.zeros((0, dim), dtype=np.float64)
    )
    if rejected.ndim == 1 and rejected.size > 0:
        rejected = rejected.reshape(1, -1)

    if len(rejected) > 0:
        dists_rej_all = np.linalg.norm(pool[:, None, :] - rejected[None, :, :], axis=2)
        min_d_rej_all = np.min(dists_rej_all, axis=1)
        keep_mask = min_d_rej_all >= exclusion_radius
        if int(np.sum(keep_mask)) >= 8:
            filtered = pool[keep_mask]
            filtered_rej_dists = min_d_rej_all[keep_mask]
        else:
            # Fallback determinístico: manter os 25% mais distantes da região rejeitada
            top_q = max(8, len(pool) // 4)
            order_far = np.argsort(min_d_rej_all)[::-1][:top_q]
            filtered = pool[order_far]
            filtered_rej_dists = min_d_rej_all[order_far]
    else:
        filtered = pool
        filtered_rej_dists = np.full(len(filtered), float("inf"), dtype=np.float64)

    # Pré-filtragem vetorizada dos top-K finalistas
    dists_act_all = np.linalg.norm(filtered[:, None, :] - active[None, :, :], axis=2)
    min_d_act_all = np.min(dists_act_all, axis=1)

    cand_bins = np.clip(np.floor(filtered * n_bins).astype(int), 0, n_bins - 1)
    vacant_hits_vec = np.zeros(len(filtered), dtype=np.float64)
    for j in range(dim):
        vacant_hits_vec += np.isin(cand_bins[:, j], vacant_bins_per_axis[j]).astype(np.float64)
    vacant_ratio_vec = vacant_hits_vec / float(dim)

    k_top = min(64, len(filtered))
    top_dmin_idx = np.argsort(min_d_act_all)[::-1][: min(32, k_top)]
    top_comb_idx = np.argsort(vacant_ratio_vec + min_d_act_all)[::-1][: min(32, k_top)]
    finalist_indices = np.unique(np.concatenate([top_dmin_idx, top_comb_idx]))[:k_top]

    # Avaliação exata no conjunto unido U_active U {u_cand}
    eval_cd2: List[float] = []
    eval_rho: List[float] = []
    eval_dmin: List[float] = []
    eval_vac: List[float] = []
    eval_hits: List[int] = []

    upper_lim = 1.0 - 1e-12
    for idx in finalist_indices:
        u_cand = filtered[idx]
        u_union = np.vstack([active, u_cand.reshape(1, -1)])
        cd2_val = float(qmc.discrepancy(np.clip(u_union, 0.0, upper_lim), method="CD"))
        dmin_val = float(min_d_act_all[idx])
        if len(u_union) >= 3:
            p_mat, _ = _safe_correlation_matrices(u_union)
            triu_i, triu_j = np.triu_indices(dim, k=1)
            rho_val = float(np.max(np.abs(p_mat[triu_i, triu_j]))) if len(triu_i) > 0 else 0.0
        else:
            rho_val = 0.0
        hits_int = int(round(vacant_hits_vec[idx]))
        eval_cd2.append(cd2_val)
        eval_rho.append(rho_val)
        eval_dmin.append(dmin_val)
        eval_vac.append(float(hits_int / float(dim)))
        eval_hits.append(hits_int)

    def _norm01(arr_list: List[float]) -> np.ndarray:
        arr = np.asarray(arr_list, dtype=np.float64)
        span = float(np.max(arr) - np.min(arr))
        if span <= 1e-12:
            return np.zeros_like(arr)
        return (arr - float(np.min(arr))) / span

    cd2_n = _norm01(eval_cd2)
    rho_n = _norm01(eval_rho)
    dmin_n = _norm01(eval_dmin)
    vac_arr = np.asarray(eval_vac, dtype=np.float64)

    scores = 0.45 * cd2_n + 0.25 * rho_n + 0.20 * (1.0 - dmin_n) + 0.10 * (1.0 - vac_arr)
    best_pos = int(np.argmin(scores))
    best_pool_idx = int(finalist_indices[best_pos])
    best_u = filtered[best_pool_idx].copy()

    rej_dist_best = (
        round(float(filtered_rej_dists[best_pool_idx]), 6)
        if math.isfinite(float(filtered_rej_dists[best_pool_idx]))
        else None
    )

    selection_metrics: Dict[str, Any] = {
        "strategy": "space_filling_min_cd2_stratum_aware",
        "pool_size": int(len(pool)),
        "candidates_after_exclusion": int(len(filtered)),
        "exclusion_radius": float(exclusion_radius),
        "min_dist_to_active": round(eval_dmin[best_pos], 6),
        "min_dist_to_rejected": rej_dist_best,
        "vacant_strata_filled": int(eval_hits[best_pos]),
        "vacant_strata_total_dims": int(dim),
        "resulting_cd2": round(eval_cd2[best_pos], 8),
        "resulting_max_abs_pearson": round(eval_rho[best_pos], 6),
    }
    return best_u, selection_metrics


def log_lhs_balance_summary(scene_id: str, report: Dict[str, Any]) -> None:
    """Exibe no terminal uma tabela Rich resumindo o equilíbrio estatístico Original vs Final."""
    orig = report.get("original_metrics") or {}
    fin = report.get("final_metrics") or {}
    comp = report.get("comparison")

    if not fin.get("valid"):
        logger.info(
            f"[{scene_id}] Diagnóstico estatístico P7 ignorado (N={fin.get('n_samples', 0)} < 2)."
        )
        return

    table = Table(
        title=f"Diagnóstico Estatístico LHS (Família P7) — {scene_id}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Métrica", style="bold")
    table.add_column("Original", justify="right")
    table.add_column("Final", justify="right")
    table.add_column("Critério / Status", justify="left")

    orig_cd2 = orig.get("discrepancy", {}) if orig.get("valid") else {}
    fin_cd2 = fin["discrepancy"]
    orig_ms = orig.get("marginal_summary", {}) if orig.get("valid") else {}
    fin_ms = fin["marginal_summary"]
    orig_or = orig.get("orthogonality", {}) if orig.get("valid") else {}
    fin_or = fin["orthogonality"]
    orig_sf = orig.get("space_filling", {}) if orig.get("valid") else {}
    fin_sf = fin["space_filling"]
    th = fin["thresholds_used"]

    table.add_row(
        "Amostras (N)",
        str(orig.get("n_samples", "--")),
        str(fin.get("n_samples", "--")),
        "OK" if fin["quality_checks"]["sample_size_sufficient"] else "Insuficiente (<6)",
    )
    table.add_row(
        "CD2 (Centered L2)",
        f"{orig_cd2.get('centered_l2_cd2', '--')}",
        f"{fin_cd2['centered_l2_cd2']}",
        f"Percentil MC: {fin_cd2['cd2_percentile_vs_mc']}% (<= {th['max_mc_cd2_percentile']}%)",
    )
    table.add_row(
        "Min KS p-valor (1D)",
        f"{orig_ms.get('min_ks_pvalue', '--')}",
        f"{fin_ms['min_ks_pvalue']}",
        f">= {th['ks_alpha']} ({'OK' if fin['quality_checks']['ks_uniformity_ok'] else 'REPROVADO'})",
    )
    table.add_row(
        "Cobertura Estratos Min",
        f"{orig_ms.get('min_stratum_coverage', '--')}",
        f"{fin_ms['min_stratum_coverage']}",
        f">= {th['min_stratum_coverage']} ({'OK' if fin['quality_checks']['stratum_coverage_ok'] else 'REPROVADO'})",
    )
    table.add_row(
        "Max |Pearson|",
        f"{orig_or.get('max_abs_pearson', '--')}",
        f"{fin_or['max_abs_pearson']}",
        f"<= {th['max_abs_pearson']} (RMS={fin_or['rms_pearson']})",
    )
    table.add_row(
        "Dist. Mínima 6D (d_min)",
        f"{orig_sf.get('min_pairwise_dist', '--')}",
        f"{fin_sf['min_pairwise_dist']}",
        f"Ratio={comp['min_pairwise_dist_ratio']}" if comp else "--",
    )
    table.add_row(
        "MaxPro Criterion",
        f"{orig_sf.get('maxpro_criterion', '--')}",
        f"{fin_sf['maxpro_criterion']}",
        f"Delta={comp['delta_maxpro_pct']:+.2f}%" if comp else "--",
    )
    if comp and report.get("has_replacements"):
        table.add_row(
            "Shift Orig vs Final (KS 2-am.)",
            "--",
            f"min p={comp['min_ks_2samp_pvalue']}",
            f"max W1={comp['max_wasserstein_1d']} ({'OK' if comp['no_distribution_shift_ok'] else 'SHIFT'})",
        )

    status_badge = "[bold green]EQUILIBRADO[/bold green]" if report["overall_balanced"] else "[bold yellow]ATENÇÃO / FORA DE CRITÉRIO[/bold yellow]"
    table.add_row("Veredito Global", "--", "--", status_badge)
    console.print(table)


def format_lhs_balance_markdown(
    scene_reports: Dict[str, Dict[str, Any]],
    plan_path: Optional[Path] = None,
) -> str:
    """Formata relatório estatístico completo em Markdown para auditoria e documentação."""
    lines: List[str] = [
        "# Relatório de Equilíbrio Estatístico — Família P7 (Latin Hypercube Sampling 6-DoF)",
        "",
    ]
    if plan_path is not None:
        lines.append(f"**Plano de referência:** `{plan_path}`")
        lines.append("")

    for scene_id, report in scene_reports.items():
        orig = report.get("original_metrics") or {}
        fin = report.get("final_metrics") or {}
        comp = report.get("comparison")

        verdict = "APROVADO (Equilibrado)" if report.get("overall_balanced") else "REPROVADO / ATENÇÃO"
        lines.append(f"## Cena `{scene_id}` — Status: **{verdict}**")
        lines.append("")
        if not fin.get("valid"):
            lines.append(f"Conjunto insuficiente para análise estatística completa (`N={fin.get('n_samples', 0)}`).")
            lines.append("")
            continue

        orig_cd2 = orig.get("discrepancy", {}) if orig.get("valid") else {}
        fin_cd2 = fin["discrepancy"]
        orig_ms = orig.get("marginal_summary", {}) if orig.get("valid") else {}
        fin_ms = fin["marginal_summary"]
        orig_or = orig.get("orthogonality", {}) if orig.get("valid") else {}
        fin_or = fin["orthogonality"]
        orig_sf = orig.get("space_filling", {}) if orig.get("valid") else {}
        fin_sf = fin["space_filling"]

        lines.extend([
            "### 1. Resumo Global (Original vs. Conjunto Final)",
            "",
            "| Métrica | Original | Final | Variação / Critério |",
            "| :--- | :---: | :---: | :--- |",
            f"| Amostras (`N`) | {orig.get('n_samples', '--')} | {fin['n_samples']} | Substituições: {'Sim' if report.get('has_replacements') else 'Não'} |",
            f"| Centered L2-Discrepancy (`CD2`) | {orig_cd2.get('centered_l2_cd2', '--')} | {fin_cd2['centered_l2_cd2']} | Percentil MC: `{fin_cd2['cd2_percentile_vs_mc']}%` (z=`{fin_cd2['cd2_zscore_vs_mc']}`) |",
            f"| Wrap-Around Discrepancy (`WD2`) | {orig_cd2.get('wrap_around_wd2', '--')} | {fin_cd2['wrap_around_wd2']} | Baseline MC p05: `{fin_cd2['mc_cd2_p05']}` |",
            f"| Menor KS p-valor (Uniformidade 1D) | {orig_ms.get('min_ks_pvalue', '--')} | {fin_ms['min_ks_pvalue']} | {'OK' if fin['quality_checks']['ks_uniformity_ok'] else 'Abaixo de 0.05'} |",
            f"| Cobertura Mínima de Estratos 1D | {orig_ms.get('min_stratum_coverage', '--')} | {fin_ms['min_stratum_coverage']} | Média: `{fin_ms['mean_stratum_coverage']}` |",
            f"| Máxima Correlação `|Pearson|` | {orig_or.get('max_abs_pearson', '--')} | {fin_or['max_abs_pearson']} | RMS=`{fin_or['rms_pearson']}` / Cond=`{fin_or['condition_number']}` |",
            f"| Máxima Correlação `|Spearman|` | {orig_or.get('max_abs_spearman', '--')} | {fin_or['max_abs_spearman']} | RMS=`{fin_or['rms_spearman']}` |",
            f"| Distância Mínima 6D (`d_min`) | {orig_sf.get('min_pairwise_dist', '--')} | {fin_sf['min_pairwise_dist']} | " + (f"Razão Final/Orig: `{comp['min_pairwise_dist_ratio']}`" if comp else "--") + " |",
            f"| Critério Morris-Mitchell (`phi_p10`) | {orig_sf.get('phi_p10', '--')} | {fin_sf['phi_p10']} | Quanto menor, melhor espaçamento |",
            f"| Critério `MaxPro` (Projeções) | {orig_sf.get('maxpro_criterion', '--')} | {fin_sf['maxpro_criterion']} | " + (f"Delta: `{comp['delta_maxpro_pct']:+.2f}%`" if comp else "--") + " |",
            "",
        ])

        lines.extend([
            "### 2. Análise Marginal por Eixo ($u \\in [0, 1]$)",
            "",
            "| Eixo | Média | Desvio Padrão | Contagem (- / +) | KS p-valor | Estratos Únicos |"
            + (" KS 2-amostras p-valor | Wasserstein 1D |" if comp else ""),
            "| :---: | :---: | :---: | :---: | :---: | :---: |"
            + (" :---: | :---: |" if comp else ""),
        ])
        for axis_name, m in fin["marginal_per_axis"].items():
            row = (
                f"| `{axis_name}` | {m['mean']:.4f} | {m['std']:.4f} | "
                f"{m['negative_count']} / {m['positive_count']} | {m['ks_pvalue']:.4f} | "
                f"{m['unique_strata']}/{fin['n_samples']} ({m['stratum_coverage_ratio']*100:.1f}%) |"
            )
            if comp and axis_name in comp["per_axis_shift"]:
                sh = comp["per_axis_shift"][axis_name]
                row += f" {sh['ks_2samp_pvalue']:.4f} | {sh['wasserstein_1d']:.4f} |"
            lines.append(row)
        lines.append("")

        axes = list(fin["marginal_per_axis"].keys())
        lines.extend([
            "### 3. Matriz de Correlação de Pearson (Conjunto Final)",
            "",
            "| | " + " | ".join(f"`{a}`" for a in axes) + " |",
            "| :---: | " + " | ".join([":---:"] * len(axes)) + " |",
        ])
        for i, a in enumerate(axes):
            vals_str = " | ".join(f"{fin_or['pearson_matrix'][i][j]:+.3f}" for j in range(len(axes)))
            lines.append(f"| `{a}` | {vals_str} |")
        lines.append("")

    return "\n".join(lines)
