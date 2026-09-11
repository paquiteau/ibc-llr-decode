"""Subject-level and group-level statistics for comparing denoising strategies."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def nadeau_bengio_corrected_ttest(scores_a, scores_b, n_train, n_test):
    """Paired t-test with the Nadeau & Bengio (2003) correction for CV fold correlation.

    `n_train`/`n_test` are the actual number of trials in the train/test split
    of each fold (a scalar, or averaged if it varies slightly across folds).
    These must come from the real CV split sizes (see `decode.run_decoding`'s
    `fold_sizes` output) — never hardcode them, the correction is only valid
    for the CV scheme that actually produced `scores_a`/`scores_b`.
    """
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    j = len(diffs)
    if j < 2:
        raise ValueError("Need at least 2 CV folds for the corrected t-test.")
    n_train, n_test = float(np.mean(n_train)), float(np.mean(n_test))
    mean_diff, var_diff = diffs.mean(), diffs.var(ddof=1)
    if var_diff == 0:
        return (np.inf, 0.0) if mean_diff != 0 else (0.0, 1.0)
    correction = 1.0 / j + n_test / n_train
    t_stat = mean_diff / np.sqrt(correction * var_diff)
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=j - 1)
    return float(t_stat), float(p_value)


def _condition_dir(decode_root, strategy, sub_label, tag, k):
    return Path(decode_root) / strategy / f"sub-{sub_label}" / tag / f"test_anova_{k}"


def load_fold_scores(decode_root, strategy, sub_label, tag="decode", k=1500):
    """Read cvscores.npy + fold_sizes.json written by decode.save_results()."""
    condition_dir = _condition_dir(decode_root, strategy, sub_label, tag, k)
    scores = np.load(condition_dir / "cvscores.npy")
    fold_sizes = json.loads((condition_dir / "fold_sizes.json").read_text())
    n_train = [fs[0] for fs in fold_sizes]
    n_test = [fs[1] for fs in fold_sizes]
    return scores, n_train, n_test


def load_predictions(decode_root, strategy, sub_label, tag="decode", k=1500):
    """Read y.npy + cv_predict.npy written by decode.save_results()."""
    condition_dir = _condition_dir(decode_root, strategy, sub_label, tag, k)
    y_true = np.load(condition_dir / "y.npy", allow_pickle=True)
    y_pred = np.load(condition_dir / "cv_predict.npy", allow_pickle=True)
    return y_true, y_pred


def subject_level_pairwise_ttests(
    decode_root, strategies, subjects, tag="decode", k=1500
):
    """One row per (subject, strategy pair): means, Nadeau-Bengio t/p."""
    rows = []
    for sub_label in subjects:
        per_strategy = {
            s: load_fold_scores(decode_root, s, sub_label, tag, k) for s in strategies
        }
        for i, a in enumerate(strategies):
            for b in strategies[i + 1 :]:
                scores_a, n_train_a, n_test_a = per_strategy[a]
                scores_b, n_train_b, n_test_b = per_strategy[b]
                if len(scores_a) != len(scores_b):
                    raise ValueError(
                        f"sub-{sub_label}: {a} has {len(scores_a)} folds, {b} has "
                        f"{len(scores_b)} — a paired test requires the same number of folds."
                    )
                n_train = np.mean(n_train_a + n_train_b)
                n_test = np.mean(n_test_a + n_test_b)
                t_stat, p_value = nadeau_bengio_corrected_ttest(
                    scores_a, scores_b, n_train, n_test
                )
                rows.append(
                    dict(
                        subject=sub_label,
                        strategy_a=a,
                        strategy_b=b,
                        mean_a=float(scores_a.mean()),
                        mean_b=float(scores_b.mean()),
                        t_stat=t_stat,
                        p_value=p_value,
                    )
                )
    return pd.DataFrame(rows)


def build_long_dataframe(decode_root, strategies, subjects, tag="decode", k=1500):
    """One row per subject x fold x strategy — input to the group-level LMM."""
    rows = []
    for sub_label in subjects:
        for strategy in strategies:
            scores, _, _ = load_fold_scores(decode_root, strategy, sub_label, tag, k)
            for fold_idx, acc in enumerate(scores):
                rows.append(
                    dict(
                        subject=sub_label,
                        fold=fold_idx,
                        strategy=strategy,
                        accuracy=float(acc),
                    )
                )
    return pd.DataFrame(rows)


def fit_group_lmm(df_long, reference="fmriprep"):
    """accuracy ~ strategy (3-level fixed effect), subject as random effect (intercept)."""
    formula = f'accuracy ~ C(strategy, Treatment(reference="{reference}"))'
    return smf.mixedlm(formula, df_long, groups=df_long["subject"]).fit()


def pairwise_contrasts_from_lmm(lmm_result, strategies, reference):
    """All pairwise strategy contrasts (n choose 2, for n strategies) from the ONE fitted model.

    Uses numeric contrast vectors (not string formulas) so that strategy names
    containing "+" (e.g. "fmriprep+denoised") can't be misparsed as arithmetic
    operators.
    """
    others = [s for s in strategies if s != reference]
    # fe_params (not params) — params also includes random-effect variance terms
    # (e.g. "Group Var"), but t_test's r_matrix must have exactly k_fe columns.
    param_names = list(lmm_result.fe_params.index)

    def term_for(level):
        return f'C(strategy, Treatment(reference="{reference}"))[T.{level}]'

    def contrast_row(strategy_a, strategy_b, coeffs):
        r = np.zeros((1, len(param_names)))
        for level, sign in coeffs:
            r[0, param_names.index(term_for(level))] = sign
        result = lmm_result.t_test(r)
        return dict(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            estimate=float(np.asarray(result.effect).ravel()[0]),
            p_value=float(np.asarray(result.pvalue).ravel()[0]),
        )

    rows = [contrast_row(reference, other, [(other, 1.0)]) for other in others]
    rows.extend(
        contrast_row(a, b, [(a, 1.0), (b, -1.0)]) for a, b in combinations(others, 2)
    )
    return pd.DataFrame(rows)
