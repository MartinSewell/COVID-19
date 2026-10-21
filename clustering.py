# The Effectiveness of Lockdowns, Face Masks and Vaccination Programmes
# Martin Sewell. Reviewed 21 October 2026.
"""Reproducible K-means, with explicit missing-data handling and row alignment.

Inputs: clusteringcd.txt, clusteringem.txt, clustering_countries.txt.
Outputs: clusters.txt (19 COVID-death rows then 19 excess-mortality rows),
         clusters.inputs.fnv, clustering_audit.json.

The default uses only dates observed for every retained country. Countries
with no data receive -1, preserving the C++ country order without inventing
zero-valued histories. --missing-policy legacy-zero retains zero imputation
for partially missing histories as a sensitivity analysis. It still excludes
wholly missing countries and strips trailing delimiters correctly.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


def fingerprint(path):
    h = 14695981039346656037
    for byte in Path(path).read_bytes():
        h = ((h ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return h


def read_matrix(path):
    # Whitespace parsing does not manufacture a final NaN column from a trailing tab.
    x = np.loadtxt(path, ndmin=2)
    if x.shape[0] < 20 or x.shape[1] < 28 or np.isinf(x).any():
        raise ValueError(f'{path}: invalid dimensions or infinite values.')
    return x


def smooth28(data):
    # At the ends, average the available days; do not pad with fictitious zero days.
    weights = np.ones(28)
    return np.convolve(data, weights, mode='same') / np.convolve(np.ones(len(data)), weights, mode='same')


def cluster_matrix(x, policy, name):
    observed = np.isfinite(x)
    keep = observed.any(axis=1)
    if keep.sum() < 20:
        raise ValueError(f'{name}: fewer than 20 countries have any observations.')
    retained = x[keep].copy()
    if policy == 'common-dates':
        days = np.flatnonzero(np.isfinite(retained).all(axis=0))
        if len(days) < 28:
            raise ValueError(f'{name}: fewer than 28 dates shared by all observed countries.')
        # Smooth the calendar series first. Select only dates whose entire available
        # 28-day window is observed for every retained country; never bridge gaps.
        smoothed = np.apply_along_axis(smooth28, 1, retained)
        used = np.flatnonzero(np.isfinite(smoothed).all(axis=0))
        if len(used) < 28:
            raise ValueError(f'{name}: fewer than 28 complete smoothed dates remain.')
        values = smoothed[:, used]
    else:
        used = np.arange(x.shape[1])
        values = np.apply_along_axis(smooth28, 1, np.nan_to_num(retained, nan=0.0))
    if np.unique(values, axis=0).shape[0] < 20:
        raise ValueError(f'{name}: fewer than 20 distinct observed histories.')
    result = []
    with threadpool_limits(limits=1):
        for k in range(2, 21):
            labels = np.full(x.shape[0], -1, dtype=int)
            labels[keep] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(values)
            result.append(labels)
    info = dict(input_shape=list(x.shape), retained_countries=int(keep.sum()),
                excluded_country_rows=np.flatnonzero(~keep).tolist(),
                missing_policy=policy, feature_date_indices=used.tolist(),
                feature_count=len(used), units=('deaths per million people per day'
                    if name == 'COVID-19 deaths' else 'excess mortality P-score (%)'))
    print(f'{name}: {keep.sum()} countries retained; {(~keep).sum()} wholly missing countries excluded; {len(used)} smoothed date features.')
    return np.asarray(result), info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--missing-policy', choices=['common-dates', 'legacy-zero'], default='common-dates')
    args = parser.parse_args()
    paths = ['clusteringcd.txt', 'clusteringem.txt', 'clustering_countries.txt']
    before = [fingerprint(p) for p in paths]
    countries = Path(paths[2]).read_text(encoding='utf-8').split()
    matrices = [read_matrix(p) for p in paths[:2]]
    if matrices[0].shape != matrices[1].shape or matrices[0].shape[0] != len(countries):
        raise ValueError('The clustering matrices and country list have inconsistent dimensions.')
    results, audit = [], {'policy':args.missing_policy, 'numpy':np.__version__, 'scikit_learn':sklearn.__version__, 'countries':countries}
    for matrix, name in zip(matrices, ['COVID-19 deaths', 'Excess mortality']):
        labels, info = cluster_matrix(matrix, args.missing_policy, name)
        results.append(labels); audit[name] = info
    if before != [fingerprint(p) for p in paths]:
        raise RuntimeError('Clustering input files changed during the run; rerun clustering.py.')
    # Write fingerprints last, only after all fits and outputs have succeeded.
    np.savetxt('clusters.txt', np.vstack(results), fmt='%d', delimiter='\t')
    Path('clustering_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    Path('clusters.inputs.fnv').write_text(' '.join(map(str,before))+'\n', encoding='ascii')
    print('Finished! Written clusters.txt, clusters.inputs.fnv and clustering_audit.json.')


if __name__ == '__main__':
    main()
