"""
Compare execution time and RSS of different archetypal analysis algorithms.

Author: Guillermo García Cobo
"""

import numpy as np
import time

from AA_Original import AA_Original
from AA_PCHA import AA_PCHA
from AA_Fast import AA_Fast


def main(n_reps, verbose=False):
    results = {algorithm.__name__: [] for algorithm in (AA_Original, AA_PCHA, AA_Fast)}
    for n_samples in (100, 1000, 10000):
        for n_features in (5, 10, 25, 50):
            matrices = [np.random.rand(n_samples, n_features) for _ in range(n_reps)]
            for n_archetypes in range(1, 11):
                if verbose:
                    print(f"n_samples: {n_samples}, n_features: {n_features}, n_archetypes: {n_archetypes}")
                for algorithm in (AA_Original, AA_PCHA, AA_Fast):
                    seconds = 0
                    RSS = 0
                    for i in range(n_reps):
                        data = matrices[i]

                        t = time.time()
                        aa = algorithm(n_archetypes, max_iter=100, tol=1e-6).fit(data)
                        seconds += time.time() - t
                        RSS += aa.RSS

                    results[algorithm.__name__].append({'n_samples': n_samples, 'n_features': n_features,
                                                        'n_archetypes': n_archetypes, 'seconds': seconds / n_reps,
                                                        'RSS': RSS / n_reps})
                    if verbose:
                        print(f"{algorithm.__name__}: {seconds / n_reps} seconds, RSS: {RSS / n_reps}")

    with open('time_comparison.csv', 'w') as f:
        f.write('algorithm,n_samples,n_features,n_archetypes,seconds,RSS\n')
        for algorithm in results:
            for result in results[algorithm]:
                f.write(f'{algorithm},{result["n_samples"]},{result["n_features"]},{result["n_archetypes"]},'
                        f'{result["seconds"]},{result["RSS"]}\n')


if __name__ == '__main__':
    main(n_reps=20, verbose=True)
