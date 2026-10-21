# COVID-19

Code and data for the analysis of the relationship between lockdowns, face-covering
policies and vaccination and COVID-19 deaths and all-cause excess mortality, together
with SIR/Gompertz reconstructions for the United Kingdom, England and Sweden.

## Files

- `OurWorldinData.cpp` — reads the Our World in Data files and writes the per-country
  and country-pair series, the clustering inputs and the summary tables. Includes
  `DTW.hpp` and uses Boost.
- `DTW.hpp` — self-contained dynamic time warping header, compiled into
  `OurWorldinData.cpp`.
- `clustering.py` — k-means clustering of the COVID-19 death-rate and excess-mortality
  series.
- `SIR.py` — the SIR/Gompertz reconstruction, the descriptive statistics and all figures.
- `UKdeaths.txt`, `Swedendeaths.txt`, `Englanddeaths.txt` — daily COVID-19 death counts.
- `figures/` — the generated figures.

## Requirements

- A C++17 compiler with the Boost headers (`boost/math/interpolators/barycentric_rational.hpp`).
  `DTW.hpp` must be on the include path; define `OWID_SKIP_DTW` to build without the
  optional dynamic-time-warping stage.
- Python 3 with `numpy`, `scipy`, `matplotlib`, `scikit-learn` and `threadpoolctl`.
  `SIR.py` renders figure text with LaTeX when a `latex` installation is found, and
  otherwise falls back to Matplotlib's mathtext (or pass `--no-tex`).

## Data

Download the following from Our World in Data into the directory containing the
compiled program:

- `owid-covid-data.csv` (stringency index and COVID-19 death rate) —
  https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-data.csv
- `face-covering-policies-covid.csv` —
  https://ourworldindata.org/grapher/face-covering-policies-covid
- `daily-covid-vaccination-doses-per-capita.csv` —
  https://ourworldindata.org/grapher/daily-covid-vaccination-doses-per-capita
- `covid-vaccination-doses-per-capita.csv` —
  https://ourworldindata.org/grapher/covid-vaccination-doses-per-capita
- `excess-mortality-p-scores-average-baseline-by-age.csv` —
  https://ourworldindata.org/grapher/excess-mortality-p-scores-average-baseline-by-age

## Running the analysis

1. Download all files from this repository.
2. Put `UKdeaths.txt`, `Swedendeaths.txt` and `Englanddeaths.txt` in the same directory
   as `SIR.py`.
3. Compile `OurWorldinData.cpp` (with `DTW.hpp` on the include path).
4. Download the Our World in Data files listed above into the directory containing the
   compiled program.
5. Run the program. It writes the clustering inputs — `clusteringcd.txt`,
   `clusteringem.txt` and `clustering_countries.txt` — and then waits.
6. In the same directory, run `clustering.py`. It writes `clusters.txt`,
   `clusters.inputs.fnv` and `clustering_audit.json`.
7. Press Enter in the program's window. It reads `clusters.txt` (checking it against
   `clusters.inputs.fnv`) and writes its summary and its per-country and country-pair
   files.
8. Copy the program's output files to the directory containing `SIR.py`.
9. Run `SIR.py`.

Instead of pausing at step 7, the program can be run non-interactively: run it once with
`--prepare-only` to write the clustering inputs and exit, run `clustering.py`, then run
the program again with `--use-clusters`.
