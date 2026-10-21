# The Effectiveness of Lockdowns, Face Masks and Vaccination Programmes Vis-à-Vis Mitigating COVID-19
# Martin Sewell
# martin.sewell@cantab.net
# 23 October 2024
# Reviewed 21 October 2026: see AUDIT_REPORT.txt.
# Revised 17 October 2026: corrected beta at t=0; relabelled figures in terms of
# s = 100*S/N (the percentage of the population susceptible); documented the
# fixed-duration SIR reconstruction; made the output statistics explicitly
# descriptive and added unsmoothed variants; removed no-op prop_cycler lines;
# added a moving-block bootstrap uncertainty analysis for Statistics 1-3 and the Gompertz difference-in-differences.

import datetime as dt
import argparse
import io
from pathlib import Path
import shutil
import textwrap
import re
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import scipy.optimize as optimize
import statistics
from numpy import genfromtxt as _genfromtxt
import warnings
# The two warnings below are expected and benign: the cluster-plotting loop reads many
# legitimately empty (cluster x vaccination-tertile) files, and np.nanmean over all-NaN
# dates (e.g. 2020, before vaccination) returns NaN. Both cases are handled downstream.
warnings.filterwarnings("ignore", message="genfromtxt: Empty input file")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# Abbreviations:
# UK United Kingdom
# SE Sweden
# EN England
# HU Hungary
# RO Romania
# IFR infection fatality rate

# Constants
pop_UK = 67886011 # Worldometer (2020) https://www.worldometers.info/world-population/uk-population/
pop_SE = 10099265 # Worldometer (2020) https://www.worldometers.info/world-population/sweden-population/
pop_EN = 56550000 # ONS (2020) https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/bulletins/annualmidyearpopulationestimates/mid2020
IFR = 0.51/100    # Oke and Heneghan (2021) https://www.cebm.net/covid-19/global-covid-19-case-fatality-rates/
gamma = 1/9       # Van Beusekom (2020) https://www.cidrap.umn.edu/news-perspective/2020/11/covid-19-most-contagious-first-5-days-illness-study-finds

# Data sources:
# UK COVID-19 deaths: https://coronavirus.data.gov.uk/details/deaths
# Sweden COVID-19 deaths: https://fohm.maps.arcgis.com/sharing/rest/content/items/b5e7488e117749c19881cce45db13f7e/data Antal avlidna per dag
# England COVID-19 deaths: https://coronavirus.data.gov.uk/details/deaths?areaType=nation&areaName=England
# Vaccinations: https://ourworldindata.org/grapher/covid-vaccination-doses-per-capita
# Excess mortality: https://ourworldindata.org/grapher/excess-mortality-p-scores-average-baseline-by-age

# Enable LaTeX maths symbols in figures
parser = argparse.ArgumentParser(description='COVID-19 analyses and figures. Run in the directory containing the C++ output and deaths files.')
parser.add_argument('--bootstrap-replicates', type=int, default=10000)
parser.add_argument('--no-tex', action='store_true', help='Use Matplotlib maths rendering without a LaTeX installation.')
args = parser.parse_args()
if args.bootstrap_replicates < 2:
    parser.error('--bootstrap-replicates must be at least 2.')
plt.rcParams['text.usetex'] = not args.no_tex and shutil.which('latex') is not None
plt.rcParams['figure.figsize'] = (9, 5.6)
plt.rcParams['savefig.dpi'] = 160
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.15

# Distinguish series by line style as well as colour, so charts stay legible in greyscale print.
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=['#000000', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'], linestyle=['-', '--', '-.', ':', '-', '--'])

# Function to parse dates in input files
def date_parser(d_bytes):
    # NumPy passes bytes to genfromtxt converters in older versions and str in newer ones; handle both.
    s = d_bytes.decode('utf-8') if isinstance(d_bytes, bytes) else d_bytes
    return np.datetime64(dt.datetime.strptime(s, '%d/%m/%Y'))

def genfromtxt(path, **kwargs):
    """Read dates as before; preserve matrix shape without trailing-tab columns."""
    if kwargs.get('converters') is not None:
        return _genfromtxt(path, **kwargs)
    text = Path(path).read_text(encoding='utf-8')
    if not text.strip():
        return np.empty((len(dates), 0))
    kwargs['delimiter'] = None
    kwargs['ndmin'] = 2
    return _genfromtxt(io.StringIO(text), **kwargs)


def smooth(data, window_size):
    """Centred equal-weight average of available endpoint days.

    Missing observations remain missing: a window containing NaN stays NaN.
    For an even window, as in NumPy's 'same' convention, the window has one
    extra day to the left. No zero-valued days are invented outside the series.
    """
    data = np.asarray(data, dtype=float)
    w = int(window_size)
    if data.ndim != 1 or data.size == 0 or w < 1 or w != window_size:
        raise ValueError('Smoothing requires a non-empty vector and a positive integer window.')
    weights = np.ones(w)
    start = (w - 1) // 2
    total = np.convolve(data, weights, mode='full')[start:start+len(data)]
    count = np.convolve(np.ones(len(data)), weights, mode='full')[start:start+len(data)]
    return total / count


def sync_colour_cycle(ax_to, ax_from):
    """Use distinct public property cycles on twin axes (no private API)."""
    entries = list(plt.rcParams['axes.prop_cycle'])
    offset = len(entries) // 2
    shifted = entries[offset:] + entries[:offset]
    ax_to.set_prop_cycle(cycler(color=[v['color'] for v in shifted],
                               linestyle=[v['linestyle'] for v in shifted]))


def save_figure(fig, filename):
    """Apply explicit units, readable titles and dates before each export."""
    name = Path(filename).stem
    first = fig.axes[0]
    # A blank curve must not imply that observations exist for its legend entry.
    series_has_data = {}
    for ax in fig.axes:
        for line in ax.lines:
            label = ' '.join(line.get_label().split())
            values = np.ma.asarray(line.get_ydata(), dtype=float).compressed()
            series_has_data[label] = series_has_data.get(label, False) or bool(np.isfinite(values).any())
    for ax in fig.axes:
        if not ax.get_xlabel() and isinstance(ax.xaxis.get_major_formatter(), mdates.DateFormatter):
            ax.set_xlabel('Date')
        title = ax.get_title()
        if title and '$' not in title:
            ax.set_title(textwrap.fill(title, width=76))
        for legend in [ax.get_legend()]:
            if legend is not None:
                for text in legend.get_texts():
                    label = text.get_text()
                    if series_has_data.get(' '.join(label.split())) is False:
                        label += ' (no data)'
                    if '$' not in label:
                        label = textwrap.fill(label, width=70)
                    text.set_text(label)
    if Path(filename).name in ('fig5.pdf', 'fig18.png') and not getattr(fig, '_calendar_note', False):
        # The paper's fig5.pdf is fig18.png; fig5.png is a different SIR figure.
        fig.text(0.5, -0.02, 'Common calendar dates; additional centred 12-day average of beta.', ha='center', fontsize=8)
        fig._calendar_note = True
    if name.startswith('CroatiaHungary') and 'diff' in name:
        # Subtraction is Hungary minus Croatia throughout these figures.
        first.set_ylabel('Difference in daily doses per 100 people' + (' (all ages)' if '_0_' in name or '_15_' in name or '_65_' in name or '_75_' in name or '_85' in name else ''))
        right = fig.axes[1]
        if 'EM' in name:
            age = re.search(r'CroatiaHungary_(0_14|15_64|65_74|75_84|85p)', name)
            age_label = age.group(1).replace('_', '-').replace('85p', '85+') if age else 'all ages'
            right.set_ylabel('P-score difference (percentage points)\n' + age_label)
        else:
            right.set_ylabel('Difference in daily deaths per million people')
    if not name.startswith('fig') and not getattr(fig, '_smoothing_note', False):
        subject = 'Rates and P-scores' if first.get_ylabel().startswith('Scaled series') or name == 'SWEGBR' else 'Curves'
        note = subject + ': additional centred 28-day average; gaps remain missing.'
        if name.startswith('CroatiaHungaryCD'):
            note += '\nDeath rates have an additional 7-day average before the 28-day average.'
        labels = ' '.join(line.get_label() for ax in fig.axes for line in ax.lines)
        if 'cumulative' in labels:
            note += '\nGroups use cumulative totals; doses per 100 people, deaths per million, policy index-point days.'
        fig.text(0.5, -0.02, note, ha='center', fontsize=8)
        fig._smoothing_note = True
    fig.savefig(filename)
    # Each PNG is the final export for this figure; free memory in long loops.
    if str(filename).endswith('.png'):
        plt.close(fig)

# output file
f = open("output.txt","w", encoding='ascii')
f.write('Units: IFR is a fraction; gamma, beta and Gompertz r are per day; d(beta)/ds is per day per percentage point; s = 100*S/N.\n')
f.write('Smoothing: centred averages, using available days at endpoints, with missing values preserved.\n')
f.write('These are descriptive analyses conditional on the stated model and matching assumptions.\n\n')

# Extract dates and deaths from input files
# datesD refers to dates with respect to deaths
data_UK = genfromtxt('UKdeaths.txt', delimiter='\t', converters = {0: date_parser})
datesD_UK = [row[0] for row in data_UK]
deaths_UK = [row[1] for row in data_UK]
num_UK_dates = len(datesD_UK)

data_SE = genfromtxt('Swedendeaths.txt', delimiter='\t', converters = {0: date_parser})
datesD_SE = [row[0] for row in data_SE]
deaths_SE = [row[1] for row in data_SE]
num_SE_dates = len(datesD_SE)

data_EN = genfromtxt('Englanddeaths.txt', delimiter='\t', converters = {0: date_parser})
datesD_EN = [row[0] for row in data_EN]
deaths_EN = [row[1] for row in data_EN]
num_EN_dates = len(datesD_EN)

for country, calendar, counts, expected_start in [
        ('UK', datesD_UK, deaths_UK, '2020-03-01'),
        ('Sweden', datesD_SE, deaths_SE, '2020-03-10'),
        ('England', datesD_EN, deaths_EN, '2020-03-01')]:
    calendar = np.asarray(calendar, dtype='datetime64[D]')
    if calendar[0] != np.datetime64(expected_start) or len(calendar) < 197:
        raise ValueError(country + ': unexpected deaths date origin or insufficient observations.')
    if not np.all(np.diff(calendar) == np.timedelta64(1, 'D')):
        raise ValueError(country + ': deaths dates must be consecutive, without duplicates.')
    if not np.isfinite(counts).all() or np.any(np.asarray(counts) < 0):
        raise ValueError(country + ': deaths must be finite and non-negative.')


# Throughout, infections are inferred from deaths (because deaths data is more reliable than cases data).
# Time from infection to death: mean 26.8 days, standard deviation 12.4 days (source:  Wood (2021), https://doi.org/10.1111/biom.13462)

# The index for the final day of the epidemic (first wave), where smoothed deaths first reaches a minimum.
# We can use index numbers like this (and in general) for infections and deaths (the dates will be different, but the index the same).
last_UK_epi_index = 184 # 5 August 2020 for infections, 1 September 2020 for deaths
last_SE_epi_index = 179 # 9 August 2020 for infections, 5 September 2020 for deaths

# The first UK lockdown began when Boris Johnson addressed the nation in a press conference at 8.30pm on 23 March 2020 [index=49].
# Because we derive infections from deaths, we are interested in the time interval between minus two and plus two standard deviations either side of this date.
# Infections: 2020-02-28 [25] to 2020-04-17 [74] inclusive
# Deaths:     2020-03-26 [25] to 2020-05-14 [74] inclusive

# UK/England lockdown timeline [index]
# 24 March 2020 [50]: first full day of first UK lockdown
# 3 July 2020 [151]: last day of first UK lockdown
# 5 November 2020 [276]: first day of England's second lockdown
# 1 December 2020 [302]: last day of England's second lockdown
# 5 January 2021 [337]: first day of England's third lockdown
# 11 April 2021 [433]: last day of step 1 of England's third lockdown
# We analyse the first lockdown, but plot all three
first_day_UK_lockdown_index = 50
last_day_UK_lockdown_index = 151

# Deaths data is smoothed to remove weekly cycles
deaths_UK_smoothed = smooth(deaths_UK, 7)
deaths_SE_smoothed = smooth(deaths_SE, 7)
deaths_EN_smoothed = smooth(deaths_EN, 7)

# datesI refers to dates with respect to infections
datesI_UK = datesD_UK[:]
for i in range(num_UK_dates):
    datesI_UK[i] = np.datetime64(datesD_UK[i]) - np.timedelta64(27, 'D')

datesI_SE = datesD_SE[:]
for i in range(num_SE_dates):
    datesI_SE[i] = np.datetime64(datesD_SE[i]) - np.timedelta64(27, 'D')

datesI_EN = datesD_EN[:]
for i in range(num_EN_dates):
    datesI_EN[i] = np.datetime64(datesD_EN[i]) - np.timedelta64(27, 'D')

# UK COVID-19 deaths
fig1 = plt.figure(facecolor='w')
ax1 = fig1.add_subplot(111, axisbelow=True)
ax1.plot(datesD_UK, deaths_UK, alpha=0.5, lw=2, label='Daily')
ax1.plot(datesD_UK, deaths_UK_smoothed, alpha=0.5, lw=2, label='7-day moving average')
ax1.set_ylabel('Daily deaths')
ax1.yaxis.set_tick_params(length=0)
ax1.xaxis.set_tick_params(length=0)
ax1.set_xlabel('Death date')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7*8))
ax1.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig1.autofmt_xdate()
ax1.grid(visible=True)
ax1.set_title('UK COVID-19 deaths')
legend = ax1.legend()
legend.get_frame().set_alpha(1)

#    ax1.spines[spine].set_visible(False)
save_figure(fig1, "fig1.png")

# Sweden COVID-19 deaths
fig2 = plt.figure(facecolor='w')
ax2 = fig2.add_subplot(111, axisbelow=True)
ax2.plot(datesD_SE, deaths_SE, alpha=0.5, lw=2, label='Daily')
ax2.plot(datesD_SE, deaths_SE_smoothed, alpha=0.5, lw=2, label='7-day moving average')
ax2.set_ylabel('Daily deaths')
ax2.yaxis.set_tick_params(length=0)
ax2.xaxis.set_tick_params(length=0)
ax2.set_xlabel('Death date')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7*8))
ax2.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig2.autofmt_xdate()
ax2.grid(visible=True)
ax2.set_title('Sweden COVID-19 deaths')
legend = ax2.legend()
legend.get_frame().set_alpha(1)

#    ax2.spines[spine].set_visible(False)
save_figure(fig2, "fig2.png")

# COVID-19 death rate
deathrate_UK = [0.0] * num_UK_dates
deathrate_SE = [0.0] * num_SE_dates
deathrate_EN = [0.0] * num_EN_dates
for t, y in enumerate(deathrate_UK):
    deathrate_UK[t] = 100000*deaths_UK_smoothed[t]/pop_UK
for t, y in enumerate(deathrate_SE):
    deathrate_SE[t] = 100000*deaths_SE_smoothed[t]/pop_SE
for t, y in enumerate(deathrate_EN):
    deathrate_EN[t] = 100000*deaths_EN_smoothed[t]/pop_EN

first_death_UK = next(i for i, v in enumerate(deaths_UK) if v > 0)
first_death_SE = next(i for i, v in enumerate(deaths_SE) if v > 0)
deathrate1_UK = deathrate_UK[first_death_UK:]
deathrate1_SE = deathrate_SE[first_death_SE:]

# COVID-19 death rate in the UK and Sweden
fig3 = plt.figure(facecolor='w')
ax3 = fig3.add_subplot(111, axisbelow=True)
ax3.plot(deathrate1_UK, alpha=0.5, lw=2, label='UK')
ax3.plot(deathrate1_SE, alpha=0.5, lw=2, label='Sweden')
ax3.set_xlabel('Days since first COVID-19 death')
ax3.set_ylabel('Deaths per 100,000 population per day')
ax3.yaxis.set_tick_params(length=0)
ax3.xaxis.set_tick_params(length=0)
ax3.grid(visible=True)
legend = ax3.legend()
legend.get_frame().set_alpha(1)

#    ax3.spines[spine].set_visible(False)
save_figure(fig3, "fig1.pdf") # first figure in paper
ax3.set_title('COVID-19 death rate in the UK and Sweden')
save_figure(fig3, "fig3.png")

# Use deaths as a proxy for infections
infections_UK = [0.0] * num_UK_dates
infections_SE = [0.0] * num_SE_dates
infections_EN = [0.0] * num_EN_dates
for t, y in enumerate(infections_UK):
    infections_UK[t] = deaths_UK_smoothed[t]/IFR
for t, y in enumerate(infections_SE):
    infections_SE[t] = deaths_SE_smoothed[t]/IFR
for t, y in enumerate(infections_EN):
    infections_EN[t] = deaths_EN_smoothed[t]/IFR


#############
# SIR Model #
#############
# The compartments are reconstructed directly from the inferred daily infections c_t
# rather than by integrating the SIR ODEs:
#   I_t = sum of c_k over the previous 9 days (a fixed infectious period of 1/gamma = 9 days),
#   R_t = cumulative infections lagged 9 days, and S_t = S_{t-1} - c_t.
# This is a fixed-duration variant of the SIR model: removal occurs exactly 9 days after
# infection, not at the constant per-capita rate gamma*I of the continuous model.
# The 35% assumed to have pre-existing immunity form a separate inactive compartment
# P = 0.35*N, so S + I + R = 0.65*N and N = S + I + R + P.

S_UK = [0.0] * num_UK_dates
I_UK = [0.0] * num_UK_dates
R_UK = [0.0] * num_UK_dates

S_SE = [0.0] * num_SE_dates
I_SE = [0.0] * num_SE_dates
R_SE = [0.0] * num_SE_dates

S_EN = [0.0] * num_EN_dates
I_EN = [0.0] * num_EN_dates
R_EN = [0.0] * num_EN_dates

for t, y in enumerate(S_UK):
    if t == 0:
        S_UK[t] = pop_UK - infections_UK[t] - 0.35*pop_UK # Assume that 35% of the population have pre-existing immunity (Doshi 2020).
    else:
        S_UK[t] = S_UK[t-1] - infections_UK[t]

for t, y in enumerate(S_SE):
    if t == 0:
        S_SE[t] = pop_SE - infections_SE[t] - 0.35*pop_SE
    else:
        S_SE[t] = S_SE[t-1] - infections_SE[t]

for t, y in enumerate(S_EN):
    if t == 0:
        S_EN[t] = pop_EN - infections_EN[t] - 0.35*pop_EN
    else:
        S_EN[t] = S_EN[t-1] - infections_EN[t]

for t, y in enumerate(I_UK):
    if t == 0:
        I_UK[t] = infections_UK[t]
    elif t < 1/gamma:
        I_UK[t] = sum(infections_UK[0:t+1])
    else:
        I_UK[t] = sum(infections_UK[t-int(1/gamma - 1):t+1])

for t, y in enumerate(I_SE):
    if t == 0:
        I_SE[t] = infections_SE[t]
    elif t < 1/gamma:
        I_SE[t] = sum(infections_SE[0:t+1])
    else:
        I_SE[t] = sum(infections_SE[t-int(1/gamma - 1):t+1])

for t, y in enumerate(I_EN):
    if t == 0:
        I_EN[t] = infections_EN[t]
    elif t < 1/gamma:
        I_EN[t] = sum(infections_EN[0:t+1])
    else:
        I_EN[t] = sum(infections_EN[t-int(1/gamma - 1):t+1])

for t, y in enumerate(R_UK):
    if t == 0:
        R_UK[t] = 0
    else:
        R_UK[t] = S_UK[0]+I_UK[0]+R_UK[0] - I_UK[t] - S_UK[t]

for t, y in enumerate(R_SE):
    if t == 0:
        R_SE[t] = 0
    else:
        R_SE[t] = S_SE[0]+I_SE[0]+R_SE[0] - I_SE[t] - S_SE[t]

for t, y in enumerate(R_EN):
    if t == 0:
        R_EN[t] = 0
    else:
        R_EN[t] = S_EN[0]+I_EN[0]+R_EN[0] - I_EN[t] - S_EN[t]

# UK COVID-19 SIR model
fig4 = plt.figure(facecolor='w')
ax4 = fig4.add_subplot(111, axisbelow=True)
ax4.plot(datesI_UK, S_UK, alpha=0.5, lw=2, label='S')
ax4.plot(datesI_UK, I_UK, alpha=0.5, lw=2, label='I')
ax4.plot(datesI_UK, R_UK, alpha=0.5, lw=2, label='R')
ax4.set_ylabel('Number of people')
ax4.yaxis.set_tick_params(length=0)
ax4.xaxis.set_tick_params(length=0)
ax4.set_xlabel('Inferred infection date')
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax4.xaxis.set_major_locator(mdates.DayLocator(interval=17*4))
ax4.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig4.autofmt_xdate()
ax4.grid(visible=True)
ax4.set_title('UK COVID-19 SIR model')
legend = ax4.legend()
legend.get_frame().set_alpha(1)

#    ax4.spines[spine].set_visible(False)
save_figure(fig4, "fig4.png")

# Sweden COVID-19 SIR model
fig5 = plt.figure(facecolor='w')
ax5 = fig5.add_subplot(111, axisbelow=True)
ax5.plot(datesI_SE, S_SE, alpha=0.5, lw=2, label='S')
ax5.plot(datesI_SE, I_SE, alpha=0.5, lw=2, label='I')
ax5.plot(datesI_SE, R_SE, alpha=0.5, lw=2, label='R')
ax5.set_ylabel('Number of people')
ax5.yaxis.set_tick_params(length=0)
ax5.xaxis.set_tick_params(length=0)
ax5.set_xlabel('Inferred infection date')
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax5.xaxis.set_major_locator(mdates.DayLocator(interval=7*8))
ax5.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig5.autofmt_xdate()
ax5.grid(visible=True)
ax5.set_title('Sweden COVID-19 SIR model')
legend = ax5.legend()
legend.get_frame().set_alpha(1)

#    ax5.spines[spine].set_visible(False)
save_figure(fig5, "fig5.png")


##############################
# Gompertz Function Analysis #
##############################

# Fit Gompertz function to UK data

# Only use epidemic (first wave) data, so truncate data
epi_infections_UK = infections_UK[0 : last_UK_epi_index+1] # slice includes the stated final index
epi_datesI_UK = datesI_UK[0 : last_UK_epi_index+1] # slice includes the stated final index

def funUK(params):
    A, r, t0 = params
    sum = 0
    for t, y in enumerate(epi_infections_UK):
        sum = sum + (epi_infections_UK[t] - A*r*np.exp(-np.exp(-r*(t-t0))-r*(t-t0)))**2
    return sum

initial_guess_UK = [7821991, 0.06256, 42]

resultUK = optimize.minimize(funUK, initial_guess_UK, method = 'Nelder-Mead')
if resultUK.success:
    A_UK = resultUK.x[0]
    r_UK = resultUK.x[1]
    t0_UK = resultUK.x[2]
else:
    raise ValueError(resultUK.message)

GompUK = [0.0] * (last_UK_epi_index+1)
for t, y in enumerate(GompUK):
    GompUK[t] = A_UK*r_UK*np.exp(-np.exp(-r_UK*(t-t0_UK))-r_UK*(t-t0_UK))

# Identify the peak of the UK epidemic
for t, y in enumerate(GompUK):
    if t == 0:
        maxGompUKindex = 0
        maxGompUK = y
    else:
        if y > maxGompUK:
            maxGompUKindex = t
            maxGompUK = y

fig6 = plt.figure(facecolor='w')
ax6 = fig6.add_subplot(111, axisbelow=True)
ax6.plot(epi_datesI_UK, epi_infections_UK, alpha=0.5, lw=2, label='Data')
ax6.plot(epi_datesI_UK, GompUK, alpha=0.5, lw=2, label='Gompertz derivative')
ax6.set_ylabel('Inferred new infections per day')
ax6.yaxis.set_tick_params(length=0)
ax6.xaxis.set_tick_params(length=0)
ax6.set_xlabel('Inferred infection date')
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax6.xaxis.set_major_locator(mdates.DayLocator(interval=7*4))
ax6.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig6.autofmt_xdate()
ax6.grid(visible=True)
ax6.set_title('UK infections and fitted first derivative of the Gompertz function')
legend = ax6.legend()
legend.get_frame().set_alpha(1)

#    ax6.spines[spine].set_visible(False)
save_figure(fig6, "fig6.png")


# Fit Gompertz function to Sweden data

# Only use epidemic (first wave) data, so truncate data
epi_infections_SE = infections_SE[0 : last_SE_epi_index+1] # slice includes the stated final index
epi_datesI_SE = datesI_SE[0 : last_SE_epi_index+1] # slice includes the stated final index

def funSE(params):
    A, r, t0 = params
    sum = 0
    for t, y in enumerate(epi_infections_SE):
        sum = sum + (epi_infections_SE[t] - A*r*np.exp(-np.exp(-r*(t-t0))-r*(t-t0)))**2
    return sum

initial_guess_SE = [1122580, 0.04284, 42]

resultSE = optimize.minimize(funSE, initial_guess_SE, method = 'Nelder-Mead')
if resultSE.success:
    A_SE = resultSE.x[0]
    r_SE = resultSE.x[1]
    t0_SE = resultSE.x[2]
else:
    raise ValueError(resultSE.message)

GompSE = [0.0] * (last_SE_epi_index+1)
for t, y in enumerate(GompSE):
    GompSE[t] = A_SE*r_SE*np.exp(-np.exp(-r_SE*(t-t0_SE))-r_SE*(t-t0_SE))

# Identify the peak of the Sweden epidemic
for t, y in enumerate(GompSE):
    if t==0:
        maxGompSEindex = 0
        maxGompSE = y
    else:
        if y > maxGompSE:
            maxGompSEindex = t
            maxGompSE = y

# Sweden infections and fitted first derivative of the Gompertz function
fig7 = plt.figure(facecolor='w')
ax7 = fig7.add_subplot(111, axisbelow=True)
ax7.plot(epi_datesI_SE, epi_infections_SE, alpha=0.5, lw=2, label='Data')
ax7.plot(epi_datesI_SE, GompSE, alpha=0.5, lw=2, label='Gompertz derivative')
ax7.set_ylabel('Inferred new infections per day')
ax7.yaxis.set_tick_params(length=0)
ax7.xaxis.set_tick_params(length=0)
ax7.set_xlabel('Inferred infection date')
ax7.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax7.xaxis.set_major_locator(mdates.DayLocator(interval=7*4))
ax7.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig7.autofmt_xdate()
ax7.grid(visible=True)
legend = ax7.legend()
legend.get_frame().set_alpha(1)
save_figure(fig7, "fig2.pdf") # second figure in paper
ax7.set_title('Sweden infections and fitted first derivative of the Gompertz function')
save_figure(fig7, "fig7.png")

# Gompertz function - UK before lockdown

# We only want epidemic (first wave) data, so truncate data
epi_infections_UKbefore = infections_UK[0 : (first_day_UK_lockdown_index-1)+1] # slice includes the stated final index
epi_datesI_UKbefore = datesI_UK[0 : (first_day_UK_lockdown_index-1)+1] # slice includes the stated final index

def funUKbefore(params):
    A, r, t0 = params
    sum = 0
    for t, y in enumerate(epi_infections_UKbefore):
        sum = sum + (epi_infections_UKbefore[t] - A*r*np.exp(-np.exp(-r*(t-t0))-r*(t-t0)))**2
    return sum

initial_guess_UKbefore = [6509003, 0.07864, 39]

resultUKbefore = optimize.minimize(funUKbefore, initial_guess_UKbefore, method = 'Nelder-Mead')
if resultUKbefore.success:
    A_UKbefore = resultUKbefore.x[0]
    r_UKbefore = resultUKbefore.x[1]
    t0_UKbefore = resultUKbefore.x[2]
else:
    raise ValueError(resultUKbefore.message)

GompUKbefore = [0.0] * ((first_day_UK_lockdown_index-1)+1)
for t, y in enumerate(GompUKbefore):
    GompUKbefore[t] = A_UKbefore*r_UKbefore*np.exp(-np.exp(-r_UKbefore*(t-t0_UKbefore))-r_UKbefore*(t-t0_UKbefore))

# UK infections before lockdown and fitted first derivative of the Gompertz function
fig8 = plt.figure(facecolor='w')
ax8 = fig8.add_subplot(111, axisbelow=True)
ax8.plot(epi_datesI_UKbefore, epi_infections_UKbefore, alpha=0.5, lw=2, label='Data')
ax8.plot(epi_datesI_UKbefore, GompUKbefore, alpha=0.5, lw=2, label='Gompertz derivative')
ax8.set_ylabel('Inferred new infections per day')
ax8.yaxis.set_tick_params(length=0)
ax8.xaxis.set_tick_params(length=0)
ax8.set_xlabel('Inferred infection date')
ax8.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax8.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax8.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig8.autofmt_xdate()
ax8.grid(visible=True)
ax8.set_title('UK infections before lockdown and fitted first derivative of the Gompertz function')
legend = ax8.legend()
legend.get_frame().set_alpha(1)

#    ax8.spines[spine].set_visible(False)
save_figure(fig8, "fig8.png")


# Gompertz function - UK during lockdown

# only use epidemic data, so truncate data
epi_infections_UKlockdown = infections_UK[first_day_UK_lockdown_index : last_day_UK_lockdown_index+1] # slice includes the stated final index
epi_datesI_UKlockdown = datesI_UK[first_day_UK_lockdown_index : last_day_UK_lockdown_index+1]

def funUKlockdown(params):
    r  = params
    sum = 0
    for t, y in enumerate(epi_infections_UKlockdown):
        t1 = t + first_day_UK_lockdown_index
        sum = sum + (epi_infections_UKlockdown[t] - A_UKbefore*r*np.exp(-np.exp(-r*(t1-t0_UKbefore))-r*(t1-t0_UKbefore)))**2
    return sum

initial_guess_UKlockdown = [0.05529]

resultUKlockdown = optimize.minimize(funUKlockdown, initial_guess_UKlockdown, method = 'Nelder-Mead')
if resultUKlockdown.success:
    r_UKlockdown = resultUKlockdown.x[0]
else:
    raise ValueError(resultUKlockdown.message)

GompUKlockdown = [0.0] * (last_day_UK_lockdown_index+1 - first_day_UK_lockdown_index)
for t, y in enumerate(GompUKlockdown):
    t1 = t + first_day_UK_lockdown_index
    GompUKlockdown[t] = A_UKbefore*r_UKlockdown*np.exp(-np.exp(-r_UKlockdown*(t1-t0_UKbefore))-r_UKlockdown*(t1-t0_UKbefore))

# UK infections during lockdown and fitted first derivative of the Gompertz function
fig9 = plt.figure(facecolor='w')
ax9 = fig9.add_subplot(111, axisbelow=True)
ax9.plot(epi_datesI_UKlockdown, epi_infections_UKlockdown, alpha=0.5, lw=2, label='Data')
ax9.plot(epi_datesI_UKlockdown, GompUKlockdown, alpha=0.5, lw=2, label='Fitted Gompertz derivative')
ax9.set_ylabel('Inferred new infections per day')
ax9.yaxis.set_tick_params(length=0)
ax9.xaxis.set_tick_params(length=0)
ax9.set_xlabel('Inferred infection date')
ax9.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax9.xaxis.set_major_locator(mdates.DayLocator(interval=7*2))
ax9.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig9.autofmt_xdate()
ax9.grid(visible=True)
ax9.set_title('UK infections during lockdown and fitted first derivative of the Gompertz function')
legend = ax9.legend()
legend.get_frame().set_alpha(1)

#    ax9.spines[spine].set_visible(False)
save_figure(fig9, "fig9.png")


# Gompertz function - Sweden before UK lockdown

# Sweden didn't have a lockdown, but we wish to identify the part of Sweden's epidemic curve that corresponds to the UK's epidemic curve during the UK lockdown by matching the peaks of the curves.
first_SE_lockdown_index = first_day_UK_lockdown_index + (maxGompSEindex - maxGompUKindex)
last_SE_lockdown_index = last_day_UK_lockdown_index + (maxGompSEindex - maxGompUKindex)

# Only use epidemic data, so truncate data
epi_infections_SEbefore = infections_SE[0 : (first_SE_lockdown_index-1)+1] # slice includes the stated final index
epi_datesI_SEbefore = datesI_SE[0 : (first_SE_lockdown_index-1)+1] # slice includes the stated final index

def funSEbefore(params):
    A, r, t0 = params
    sum = 0
    for t, y in enumerate(epi_infections_SEbefore):
        sum = sum + (epi_infections_SEbefore[t] - A*r*np.exp(-np.exp(-r*(t-t0))-r*(t-t0)))**2
    return sum

initial_guess_SEbefore = [765735, 0.06728, 35]

resultSEbefore = optimize.minimize(funSEbefore, initial_guess_SEbefore, method = 'Nelder-Mead')
if resultSEbefore.success:
    A_SEbefore = resultSEbefore.x[0]
    r_SEbefore = resultSEbefore.x[1]
    t0_SEbefore = resultSEbefore.x[2]
else:
    raise ValueError(resultSEbefore.message)

GompSEbefore = [0.0] * ((first_SE_lockdown_index-1)+1)
for t, y in enumerate(GompSEbefore):
    GompSEbefore[t] = A_SEbefore*r_SEbefore*np.exp(-np.exp(-r_SEbefore*(t-t0_SEbefore))-r_SEbefore*(t-t0_SEbefore))

# Sweden infections before UK lockdown and fitted first derivative of the Gompertz function
fig10 = plt.figure(facecolor='w')
ax10 = fig10.add_subplot(111, axisbelow=True)
ax10.plot(epi_datesI_SEbefore, epi_infections_SEbefore, alpha=0.5, lw=2, label='Data')
ax10.plot(epi_datesI_SEbefore, GompSEbefore, alpha=0.5, lw=2, label='Gompertz derivative')
ax10.set_ylabel('Inferred new infections per day')
ax10.yaxis.set_tick_params(length=0)
ax10.xaxis.set_tick_params(length=0)
ax10.set_xlabel('Inferred infection date')
ax10.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax10.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax10.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig10.autofmt_xdate()
ax10.grid(visible=True)
ax10.set_title('Sweden infections before UK lockdown and fitted first derivative of the Gompertz function')
legend = ax10.legend()
legend.get_frame().set_alpha(1)

#    ax10.spines[spine].set_visible(False)
save_figure(fig10, "fig10.png")


# Gompertz function - Sweden during UK lockdown

# only use lockdown data, so truncate data
epi_infections_SElockdown = infections_SE[first_SE_lockdown_index : last_SE_lockdown_index + 1] # slice includes the stated final index
epi_datesI_SElockdown = datesI_SE[first_SE_lockdown_index : last_SE_lockdown_index + 1] # slice includes the stated final index

def funSElockdown(params):
    r = params
    sum = 0
    for t, y in enumerate(epi_infections_SElockdown):
        t1 = t + first_SE_lockdown_index
        sum = sum + (epi_infections_SElockdown[t] - A_SEbefore*r*np.exp(-np.exp(-r*(t1-t0_SEbefore))-r*(t1-t0_SEbefore)))**2
    return sum

initial_guess_SElockdown = [0.03888]

resultSElockdown = optimize.minimize(funSElockdown, initial_guess_SElockdown, method = 'Nelder-Mead')
if resultSElockdown.success:
    r_SElockdown = resultSElockdown.x[0]
else:
    raise ValueError(resultSElockdown.message)

GompSElockdown = [0.0] * (last_SE_lockdown_index+1 - first_SE_lockdown_index)
for t, y in enumerate(GompSElockdown):
    t1 = t + first_SE_lockdown_index
    GompSElockdown[t] = A_SEbefore*r_SElockdown*np.exp(-np.exp(-r_SElockdown*(t1-t0_SEbefore))-r_SElockdown*(t1-t0_SEbefore))

# Sweden infections during UK lockdown and fitted first derivative of the Gompertz function
fig11 = plt.figure(facecolor='w')
ax11 = fig11.add_subplot(111, axisbelow=True)
ax11.plot(epi_datesI_SElockdown, epi_infections_SElockdown, alpha=0.5, lw=2, label='Infections')
ax11.plot(epi_datesI_SElockdown, GompSElockdown, alpha=0.5, lw=2, label='Gompertz derivative')
ax11.set_ylabel('Inferred new infections per day')
ax11.yaxis.set_tick_params(length=0)
ax11.xaxis.set_tick_params(length=0)
ax11.set_xlabel('Inferred infection date')
ax11.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax11.xaxis.set_major_locator(mdates.DayLocator(interval=7*2))
ax11.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig11.autofmt_xdate()
ax11.grid(visible=True)
ax11.set_title('Sweden infections during UK lockdown and fitted first derivative of the Gompertz function')
legend = ax11.legend()
legend.get_frame().set_alpha(1)

#    ax11.spines[spine].set_visible(False)
save_figure(fig11, "fig11.png")


#################
# Beta Analysis #
#################

# Calculate s = 100*S/N, the percentage of the population who remain susceptible (S1 in this code)
# 100 * S divided by the total population
S1_UK = [0.0] * num_UK_dates
S1_SE = [0.0] * num_SE_dates
S1_EN = [0.0] * num_EN_dates
for t, y in enumerate(S1_UK):
    S1_UK[t] = 100*S_UK[t]/pop_UK
for t, y in enumerate(S1_SE):
    S1_SE[t] = 100*S_SE[t]/pop_SE
for t, y in enumerate(S1_EN):
    S1_EN[t] = 100*S_EN[t]/pop_EN

beta_UK = [0.0] * num_UK_dates
beta_SE = [0.0] * num_SE_dates
beta_EN = [0.0] * num_EN_dates

def inferred_beta(incidence, susceptible, infectious, population):
    incidence = np.asarray(incidence, dtype=float)
    denominator = np.asarray(susceptible) * np.asarray(infectious)
    out = np.full(incidence.shape, np.nan)
    valid = (np.asarray(susceptible) > 0) & (np.asarray(infectious) > 0)
    np.divide(population * incidence, denominator, out=out, where=valid)
    return out

beta_UK = inferred_beta(infections_UK, S_UK, I_UK, pop_UK)
beta_SE = inferred_beta(infections_SE, S_SE, I_SE, pop_SE)
beta_EN = inferred_beta(infections_EN, S_EN, I_EN, pop_EN)

# beta
fig12 = plt.figure(facecolor='w')
ax12 = fig12.add_subplot(111, axisbelow=True)
ax12.plot(datesI_UK, beta_UK, alpha=0.5, lw=2, label='UK')
ax12.plot(datesI_SE, beta_SE, alpha=0.5, lw=2, label='Sweden')
ax12.set_ylabel(r'$\beta$ (day$^{-1}$)')
ax12.yaxis.set_tick_params(length=0)
ax12.xaxis.set_tick_params(length=0)
ax12.set_xlabel('Inferred infection date')
ax12.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax12.xaxis.set_major_locator(mdates.DayLocator(interval=7*8))
ax12.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig12.autofmt_xdate()
ax12.grid(visible=True)
ax12.set_ylim(0,0.6) # ax12.set_ylim(0,0.4)
ax12.set_title(r'$\beta$ from the SIR models for the UK and Sweden')
legend = ax12.legend()
legend.get_frame().set_alpha(1)

#    ax12.spines[spine].set_visible(False)
save_figure(fig12, "fig12.png")

# S/N
fig13 = plt.figure(facecolor='w')
ax13 = fig13.add_subplot(111, axisbelow=True)
ax13.plot(datesI_UK, S1_UK, alpha=0.5, lw=2, label='UK')
ax13.plot(datesI_SE, S1_SE, alpha=0.5, lw=2, label='Sweden')
ax13.set_ylabel(r'$s$ (percentage of population susceptible)')
ax13.yaxis.set_tick_params(length=0)
ax13.xaxis.set_tick_params(length=0)
ax13.set_xlabel('Inferred infection date')
ax13.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax13.xaxis.set_major_locator(mdates.DayLocator(interval=7*8))
ax13.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig13.autofmt_xdate()
ax13.grid(visible=True)
ax13.set_title(r'$s$, the percentage of the population susceptible, in the SIR models for the UK and Sweden')
legend = ax13.legend()
legend.get_frame().set_alpha(1)

#    ax13.spines[spine].set_visible(False)
save_figure(fig13, "fig13.png")

# Identify the three UK/England lockdowns for figure 14.
# '1' masks the value, so '0' signifies lockdown.
lockdown_days = np.zeros(num_EN_dates, dtype=bool)
for start, stop in [('2020-03-24','2020-07-03'), ('2020-11-05','2020-12-01'), ('2021-01-05','2021-04-11')]:
    lockdown_days |= (np.asarray(datesI_EN) >= np.datetime64(start)) & (np.asarray(datesI_EN) <= np.datetime64(stop))
S1_EN_lockdown = ma.masked_array(S1_EN, mask=~lockdown_days)
S1_EN_unrestricted = ma.masked_array(S1_EN, mask=lockdown_days)

# Beta vs percentage of population susceptible in the COVID-19 SIR model
fig14 = plt.figure(facecolor='w')
ax14 = fig14.add_subplot(111, axisbelow=True)
ax14.plot(S1_EN_unrestricted, beta_EN, 'g', alpha=0.5, lw=2, label='England without lockdown')
ax14.plot(S1_EN_lockdown, beta_EN, 'r', alpha=0.5, lw=2, label='England under lockdown')
ax14.plot(S1_SE, beta_SE, 'b', alpha=0.5, lw=2, label='Sweden')
ax14.set_xlabel(r'Percentage of population susceptible, $s$')
ax14.set_ylabel(r'$\beta$ (day$^{-1}$)')
ax14.yaxis.set_tick_params(length=0)
ax14.xaxis.set_tick_params(length=0)
ax14.grid(visible=True)
ax14.set_xlim(80, 20) #ax14.set_xlim(100, 60)
ax14.set_ylim(0,0.6) #ax14.set_ylim(0,0.3)
legend = ax14.legend()
legend.get_frame().set_alpha(1)
save_figure(fig14, "fig3.pdf") # third figure in paper
ax14.set_title(r'$\beta$ vs $s$ in the COVID-19 SIR model, deaths data to 30 June 2021')
save_figure(fig14, "fig14.png")

dS1dt_UK = [0.0] * num_UK_dates
dS1dt_SE = [0.0] * num_SE_dates
dbetadt_UK = [0.0] * num_UK_dates
dbetadt_SE = [0.0] * num_SE_dates
dbetadS1_UK = [0.0] * num_UK_dates
dbetadS1_SE = [0.0] * num_SE_dates

for t, y in enumerate(dbetadt_UK):
    if t == 0:
        dbetadt_UK[t] = beta_UK[t+1]-beta_UK[t]
    else:
        dbetadt_UK[t] = beta_UK[t]-beta_UK[t-1]

for t, y in enumerate(dbetadt_SE):
    if t == 0:
        dbetadt_SE[t] = beta_SE[t+1]-beta_SE[t]
    else:
        dbetadt_SE[t] = beta_SE[t]-beta_SE[t-1]

for t, y in enumerate(dS1dt_UK):
    if t == 0:
        dS1dt_UK[t] = S1_UK[t+1]-S1_UK[t]
    else:
        dS1dt_UK[t] = S1_UK[t]-S1_UK[t-1]

for t, y in enumerate(dS1dt_SE):
    if t == 0:
        dS1dt_SE[t] = S1_SE[t+1]-S1_SE[t]
    else:
        dS1dt_SE[t] = S1_SE[t]-S1_SE[t-1]

for t, y in enumerate(dbetadS1_UK):
    if dS1dt_UK[t] != 0:
        dbetadS1_UK[t] = dbetadt_UK[t]/dS1dt_UK[t]
    else:
        dbetadS1_UK[t] = np.nan

for t, y in enumerate(dbetadS1_SE):
    if dS1dt_SE[t] != 0:
        dbetadS1_SE[t] = dbetadt_SE[t]/dS1dt_SE[t]
    else:
        dbetadS1_SE[t] = np.nan

# Only use epidemic (first wave) data, so truncate data
epi_S1_UK = S1_UK[0 : last_UK_epi_index+1]
epi_S1_SE = S1_SE[0 : last_SE_epi_index+1]
epi_dbetadS1_UK = dbetadS1_UK[0 : last_UK_epi_index+1]

# For each UK day, we cycle through Sweden days and select the day where S1_UK and S1_SE match most closely.
# We then use the dbetadS1_SE for that Sweden day, and use it as a control for dbetadS1_UK
control = [0.0] * len(epi_datesI_UK)
for UKday, S1UK in enumerate(epi_S1_UK): # outer loop: for each day in the UK during the UK epidemic (first wave)
    for SEday, S1SE in enumerate(epi_S1_SE): # inner loop: for each day in Sweden during the Sweden epidemic (first wave)
        if SEday == 0:
            SmallestDiffSoFar = (epi_S1_UK[UKday] - epi_S1_SE[SEday])**2
            closestday = SEday
        else:
            if (epi_S1_UK[UKday] - epi_S1_SE[SEday])**2 < SmallestDiffSoFar: # find the Sweden day with the closest S
                SmallestDiffSoFar = (epi_S1_UK[UKday] - epi_S1_SE[SEday])**2
                closestday = SEday
    control[UKday] = dbetadS1_SE[closestday] # reconstruct Sweden dbetadS1_SE

# dbeta/S1
fig15 = plt.figure(facecolor='w')
ax15 = fig15.add_subplot(111, axisbelow=True)
ax15.plot(epi_datesI_UK, epi_dbetadS1_UK, alpha=0.5, lw=2, label='UK')
ax15.plot(datesI_SE, dbetadS1_SE, alpha=0.5, lw=2, label='Sweden')
ax15.plot(epi_datesI_UK, control, alpha=0.5, lw=2, label='Control')
ax15.set_ylabel(r'$\mathrm{d}\beta/\mathrm{d}s$ (day$^{-1}$ per percentage point)')
ax15.yaxis.set_tick_params(length=0)
ax15.xaxis.set_tick_params(length=0)
ax15.set_xlabel('Inferred infection date')
ax15.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax15.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax15.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig15.autofmt_xdate()
ax15.grid(visible=True)
ax15.set_xlim([dt.date(2020, 2, 28), dt.date(2020, 4, 17)])
ax15.set_ylim(-0.25,0.25) # ax15.set_ylim(-0.2,0.2)
legend = ax15.legend()
legend.get_frame().set_alpha(1)

#    ax15.spines[spine].set_visible(False)
ax15.set_title(r'$\frac{\mathrm{d}\beta}{\mathrm{d}s}$ as the UK transitioned into its first lockdown')
save_figure(fig15, "fig15.png")

# Smooth data to reduce noise that may unduly influence end values measured, choose a window size of one standard deviation (12 days).
epi_dbetadS1_UK_smoothed = smooth(epi_dbetadS1_UK, 12)
control_smoothed = smooth(control, 12)
dbetadS1_SE_smoothed = smooth(dbetadS1_SE, 12)

# Smoothed dbeta/d(S/N) as the UK transitioned into its first lockdown
fig16 = plt.figure(facecolor='w')
ax16 = fig16.add_subplot(111, axisbelow=True)
ax16.plot(epi_datesI_UK, epi_dbetadS1_UK_smoothed, alpha=0.5, lw=2, label='UK')
ax16.plot(datesI_SE, dbetadS1_SE_smoothed, alpha=0.5, lw=2, label='Sweden')
ax16.plot(epi_datesI_UK, control_smoothed, alpha=0.5, lw=2, label='Control')
ax16.set_ylabel(r'$\mathrm{d}\beta/\mathrm{d}s$ (day$^{-1}$ per percentage point)')
ax16.yaxis.set_tick_params(length=0)
ax16.xaxis.set_tick_params(length=0)
ax16.set_xlabel('Inferred infection date')
ax16.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax16.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax16.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig16.autofmt_xdate()
ax16.grid(visible=True)
ax16.set_xlim([dt.date(2020, 2, 28), dt.date(2020, 4, 17)])
ax16.set_ylim(-0.25,0.25) # ax16.set_ylim(-0.2,0.2)
legend = ax16.legend()
legend.get_frame().set_alpha(1)
save_figure(fig16, "fig4.pdf") # fourth figure in paper
ax16.set_title(r'Smoothed $\frac{\mathrm{d}\beta}{\mathrm{d}s}$ as the UK transitioned into its first lockdown')
save_figure(fig16, "fig16.png")


##############
# Face Masks #
##############

# In England face coverings became compulsory in enclosed public places at 00:00 on 24 July 2020.
# Time from infection to death: mean 26.8 days, standard deviation 12.4 days (2SD = 24.8)
# Any effect of masks should show up gradually between 29 June 2020 and 17 August 2020.

# New daily infections inferred from deaths for England and Sweden
fig17 = plt.figure(facecolor='w')
ax17 = fig17.add_subplot(111, axisbelow=True)
ax17.plot(datesI_EN, infections_EN, alpha=0.5, lw=2, label='England')
ax17.plot(datesI_SE, infections_SE, alpha=0.5, lw=2, label='Sweden')
ax17.set_ylabel('New daily infections inferred from deaths')
ax17.yaxis.set_tick_params(length=0)
ax17.xaxis.set_tick_params(length=0)
ax17.set_xlabel('Inferred infection date')
ax17.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax17.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax17.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig17.autofmt_xdate()
ax17.grid(visible=True)
ax17.set_xlim([dt.date(2020, 6, 29), dt.date(2020, 8, 17)])
ax17.set_ylim(0,3500) # ax17.set_ylim(0,4000)
ax17.set_title('England and Sweden around the English face-covering mandate')
legend = ax17.legend()
legend.get_frame().set_alpha(1)

#    ax17.spines[spine].set_visible(False)
save_figure(fig17, "fig17.png")

# Smooth data to reduce noise that may unduly influence end values measured, choose a window size of one standard deviation (12 days).
beta_EN_smoothed = [0.0] * len(beta_EN)
beta_EN_smoothed = smooth(beta_EN, 12)
beta_SE_smoothed = [0.0] * len(beta_SE)
beta_SE_smoothed = smooth(beta_SE, 12)

# Smoothed beta in the SIR model inferred from deaths for England and Sweden
fig18 = plt.figure(facecolor='w')
ax18 = fig18.add_subplot(111, axisbelow=True)
ax18.plot(datesI_EN, beta_EN_smoothed, alpha=0.5, lw=2, label='England')
ax18.plot(datesI_SE, beta_SE_smoothed, alpha=0.5, lw=2, label='Sweden')
ax18.set_ylabel(r'Inferred $\beta$ (day$^{-1}$)')
ax18.yaxis.set_tick_params(length=0)
ax18.xaxis.set_tick_params(length=0)
ax18.set_xlabel('Inferred infection date')
ax18.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax18.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax18.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig18.autofmt_xdate()
ax18.grid(visible=True)
ax18.set_xlim([dt.date(2020, 6, 29), dt.date(2020, 8, 17)])
ax18.set_ylim(0,0.3) #ax18.set_ylim(0,0.2)
legend = ax18.legend()
legend.get_frame().set_alpha(1)
save_figure(fig18, "fig5.pdf") # fifth figure in paper
ax18.set_title('England and Sweden around the English face-covering mandate')
save_figure(fig18, "fig18.png")


################
# Vaccinations #
################

# The excess mortality data was converted from weekly to daily using barycentric rational interpolation from the Boost C++ Libraries.

dates = genfromtxt('dates.txt', converters = {0: date_parser})

# LOW LOCKDOWN VS VAX
lockdownlowvaxlow = genfromtxt('lockdownlowvaxlow.txt', delimiter='\t')
lockdownlowvaxmed = genfromtxt('lockdownlowvaxmed.txt', delimiter='\t')
lockdownlowvaxhigh = genfromtxt('lockdownlowvaxhigh.txt', delimiter='\t')
em1 = np.empty(len(dates))
em1[:] = np.nan
em2 = np.empty(len(dates))
em2[:] = np.nan
em3 = np.empty(len(dates))
em3[:] = np.nan
for t, y in enumerate(em1):
    em1[t] = np.nanmean(lockdownlowvaxlow[t], axis=0)
for t, y in enumerate(em2):
    em2[t] = np.nanmean(lockdownlowvaxmed[t], axis=0)
for t, y in enumerate(em3):
    em3[t] = np.nanmean(lockdownlowvaxhigh[t], axis=0)
figLV = plt.figure(facecolor='w')
axLV = figLV.add_subplot(111, axisbelow=True)
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high cumulative doses')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium cumulative doses')
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low cumulative doses')
axLV.set_ylabel('Excess mortality P-score ($\\%$)')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
save_figure(figLV, "lockdownstringencylow.pdf")
axLV.set_title('Excess mortality for countries with low cumulative stringency')
save_figure(figLV, "lockdownstringencylow.png")

# MED LOCKDOWN VS VAX
lockdownmedvaxlow = genfromtxt('lockdownmedvaxlow.txt', delimiter='\t')
lockdownmedvaxmed = genfromtxt('lockdownmedvaxmed.txt', delimiter='\t')
lockdownmedvaxhigh = genfromtxt('lockdownmedvaxhigh.txt', delimiter='\t')
em1 = np.empty(len(dates))
em1[:] = np.nan
em2 = np.empty(len(dates))
em2[:] = np.nan
em3 = np.empty(len(dates))
em3[:] = np.nan
for t, y in enumerate(em1):
    em1[t] = np.nanmean(lockdownmedvaxlow[t], axis=0)
for t, y in enumerate(em2):
    em2[t] = np.nanmean(lockdownmedvaxmed[t], axis=0)
for t, y in enumerate(em3):
    em3[t] = np.nanmean(lockdownmedvaxhigh[t], axis=0)
figLV = plt.figure(facecolor='w')
axLV = figLV.add_subplot(111, axisbelow=True)
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high cumulative doses')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium cumulative doses')
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low cumulative doses')
axLV.set_ylabel('Excess mortality P-score ($\\%$)')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
save_figure(figLV, "lockdownstringencymed.pdf")
axLV.set_title('Excess mortality for countries with medium cumulative stringency')
save_figure(figLV, "lockdownstringencymed.png")


# HIGH LOCKDOWN VS VAX
lockdownhighvaxlow = genfromtxt('lockdownhighvaxlow.txt', delimiter='\t')
lockdownhighvaxmed = genfromtxt('lockdownhighvaxmed.txt', delimiter='\t')
lockdownhighvaxhigh = genfromtxt('lockdownhighvaxhigh.txt', delimiter='\t')
em1 = np.empty(len(dates))
em1[:] = np.nan
em2 = np.empty(len(dates))
em2[:] = np.nan
em3 = np.empty(len(dates))
em3[:] = np.nan
for t, y in enumerate(em1):
    em1[t] = np.nanmean(lockdownhighvaxlow[t], axis=0)
for t, y in enumerate(em2):
    em2[t] = np.nanmean(lockdownhighvaxmed[t], axis=0)
for t, y in enumerate(em3):
    em3[t] = np.nanmean(lockdownhighvaxhigh[t], axis=0)
figLV = plt.figure(facecolor='w')
axLV = figLV.add_subplot(111, axisbelow=True)
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high cumulative doses')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium cumulative doses')
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low cumulative doses')
axLV.set_ylabel('Excess mortality P-score ($\\%$)')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
save_figure(figLV, "lockdownstringencyhigh.pdf")
axLV.set_title('Excess mortality for countries with high cumulative stringency')
save_figure(figLV, "lockdownstringencyhigh.png")



# LOW VAX VS LOCKDOWN
lockdownhighvaxlow = genfromtxt('lockdownhighvaxlow.txt', delimiter='\t')
lockdownmedvaxlow = genfromtxt('lockdownmedvaxlow.txt', delimiter='\t')
lockdownlowvaxlow = genfromtxt('lockdownlowvaxlow.txt', delimiter='\t')
emll = np.empty(len(dates))
emll[:] = np.nan
emml = np.empty(len(dates))
emml[:] = np.nan
emhl = np.empty(len(dates))
emhl[:] = np.nan
for t, y in enumerate(emll):
    emll[t] = np.nanmean(lockdownlowvaxlow[t], axis=0)
for t, y in enumerate(emml):
    emml[t] = np.nanmean(lockdownmedvaxlow[t], axis=0)
for t, y in enumerate(emhl):
    emhl[t] = np.nanmean(lockdownhighvaxlow[t], axis=0)
figLV = plt.figure(facecolor='w')
axLV = figLV.add_subplot(111, axisbelow=True)
axLV.plot(dates, smooth(emhl, 28), alpha=0.5, lw=2, label='Countries with high cumulative stringency')
axLV.plot(dates, smooth(emml, 28), alpha=0.5, lw=2, label='Countries with medium cumulative stringency')
axLV.plot(dates, smooth(emll, 28), alpha=0.5, lw=2, label='Countries with low cumulative stringency')
axLV.set_ylabel('Excess mortality P-score ($\\%$)')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
save_figure(figLV, "lockdownlowvax.pdf")
axLV.set_title('Excess mortality for countries with low cumulative doses')
save_figure(figLV, "lockdownlowvax.png")

# MED VAX VS LOCKDOWN
lockdownlowvaxmed = genfromtxt('lockdownlowvaxmed.txt', delimiter='\t')
lockdownmedvaxmed = genfromtxt('lockdownmedvaxmed.txt', delimiter='\t')
lockdownhighvaxmed = genfromtxt('lockdownhighvaxmed.txt', delimiter='\t')
em1 = np.empty(len(dates))
em1[:] = np.nan
em2 = np.empty(len(dates))
em2[:] = np.nan
em3 = np.empty(len(dates))
em3[:] = np.nan
for t, y in enumerate(em1):
    em1[t] = np.nanmean(lockdownlowvaxmed[t], axis=0)
for t, y in enumerate(em2):
    em2[t] = np.nanmean(lockdownmedvaxmed[t], axis=0)
for t, y in enumerate(em3):
    em3[t] = np.nanmean(lockdownhighvaxmed[t], axis=0)
figLV = plt.figure(facecolor='w')
axLV = figLV.add_subplot(111, axisbelow=True)
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low cumulative stringency')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium cumulative stringency')
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high cumulative stringency')
axLV.set_ylabel('Excess mortality P-score ($\\%$)')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
save_figure(figLV, "lockdownmedvax.pdf")
axLV.set_title('Excess mortality for countries with medium cumulative doses')
save_figure(figLV, "lockdownmedvax.png")


# HIGH VAX VS LOCKDOWN
lockdownlowvaxhigh = genfromtxt('lockdownlowvaxhigh.txt', delimiter='\t')
lockdownmedvaxhigh = genfromtxt('lockdownmedvaxhigh.txt', delimiter='\t')
lockdownhighvaxhigh = genfromtxt('lockdownhighvaxhigh.txt', delimiter='\t')
em1 = np.empty(len(dates))
em1[:] = np.nan
em2 = np.empty(len(dates))
em2[:] = np.nan
em3 = np.empty(len(dates))
em3[:] = np.nan
for t, y in enumerate(em1):
    em1[t] = np.nanmean(lockdownlowvaxhigh[t], axis=0)
for t, y in enumerate(em2):
    em2[t] = np.nanmean(lockdownmedvaxhigh[t], axis=0)
for t, y in enumerate(em3):
    em3[t] = np.nanmean(lockdownhighvaxhigh[t], axis=0)
figLV = plt.figure(facecolor='w')
axLV = figLV.add_subplot(111, axisbelow=True)
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low cumulative stringency')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium cumulative stringency')
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high cumulative stringency')
axLV.set_ylabel('Excess mortality P-score ($\\%$)')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
save_figure(figLV, "lockdownhighvax.pdf")
axLV.set_title('Excess mortality for countries with high cumulative doses')
save_figure(figLV, "lockdownhighvax.png")



# LOW VAX VS MASKS
maskslowvaxlow = genfromtxt('maskslowvaxlow.txt', delimiter='\t')
masksmedvaxlow = genfromtxt('masksmedvaxlow.txt', delimiter='\t')
maskshighvaxlow = genfromtxt('maskshighvaxlow.txt', delimiter='\t')
em1 = np.empty(len(dates))
em1[:] = np.nan
em2 = np.empty(len(dates))
em2[:] = np.nan
em3 = np.empty(len(dates))
em3[:] = np.nan
for t, y in enumerate(em1):
    em1[t] = np.nanmean(maskslowvaxlow[t], axis=0)
for t, y in enumerate(em2):
    em2[t] = np.nanmean(masksmedvaxlow[t], axis=0)
for t, y in enumerate(em3):
    em3[t] = np.nanmean(maskshighvaxlow[t], axis=0)
figLV = plt.figure(facecolor='w')
axLV = figLV.add_subplot(111, axisbelow=True)
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high cumulative mask policy index')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium cumulative mask policy index')
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low cumulative mask policy index')
axLV.set_ylabel('Excess mortality P-score ($\\%$)')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
save_figure(figLV, "maskslowvax.pdf")
axLV.set_title('Excess mortality for countries with low cumulative doses')
save_figure(figLV, "maskslowvax.png")

# MED VAX VS masks
maskslowvaxmed = genfromtxt('maskslowvaxmed.txt', delimiter='\t')
masksmedvaxmed = genfromtxt('masksmedvaxmed.txt', delimiter='\t')
maskshighvaxmed = genfromtxt('maskshighvaxmed.txt', delimiter='\t')
em1 = np.empty(len(dates))
em1[:] = np.nan
em2 = np.empty(len(dates))
em2[:] = np.nan
em3 = np.empty(len(dates))
em3[:] = np.nan
for t, y in enumerate(em1):
    em1[t] = np.nanmean(maskslowvaxmed[t], axis=0)
for t, y in enumerate(em2):
    em2[t] = np.nanmean(masksmedvaxmed[t], axis=0)
for t, y in enumerate(em3):
    em3[t] = np.nanmean(maskshighvaxmed[t], axis=0)
figLV = plt.figure(facecolor='w')
axLV = figLV.add_subplot(111, axisbelow=True)
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high cumulative mask policy index')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium cumulative mask policy index')
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low cumulative mask policy index')
axLV.set_ylabel('Excess mortality P-score ($\\%$)')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
save_figure(figLV, "masksmedvax.pdf")
axLV.set_title('Excess mortality for countries with medium cumulative doses')
save_figure(figLV, "masksmedvax.png")

# HIGH VAX VS masks
maskslowvaxhigh = genfromtxt('maskslowvaxhigh.txt', delimiter='\t')
masksmedvaxhigh = genfromtxt('masksmedvaxhigh.txt', delimiter='\t')
maskshighvaxhigh = genfromtxt('maskshighvaxhigh.txt', delimiter='\t')
em1 = np.empty(len(dates))
em1[:] = np.nan
em2 = np.empty(len(dates))
em2[:] = np.nan
em3 = np.empty(len(dates))
em3[:] = np.nan
for t, y in enumerate(em1):
    em1[t] = np.nanmean(maskslowvaxhigh[t], axis=0)
for t, y in enumerate(em2):
    em2[t] = np.nanmean(masksmedvaxhigh[t], axis=0)
for t, y in enumerate(em3):
    em3[t] = np.nanmean(maskshighvaxhigh[t], axis=0)
figLV = plt.figure(facecolor='w')
axLV = figLV.add_subplot(111, axisbelow=True)
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high cumulative mask policy index')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium cumulative mask policy index')
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low cumulative mask policy index')
axLV.set_ylabel('Excess mortality P-score ($\\%$)')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
save_figure(figLV, "maskshighvax.pdf")
axLV.set_title('Excess mortality for countries with high cumulative doses')
save_figure(figLV, "maskshighvax.png")

# LOW COVIDDEATHS VS VAX
coviddeathslowvaxhigh = genfromtxt('coviddeathslowvaxhigh.txt', delimiter='\t')
coviddeathslowvaxmed = genfromtxt('coviddeathslowvaxmed.txt', delimiter='\t')
coviddeathslowvaxlow = genfromtxt('coviddeathslowvaxlow.txt', delimiter='\t')
em1 = np.empty(len(dates))
em1[:] = np.nan
em2 = np.empty(len(dates))
em2[:] = np.nan
em3 = np.empty(len(dates))
em3[:] = np.nan
for t, y in enumerate(em1):
    em1[t] = np.nanmean(coviddeathslowvaxhigh[t], axis=0)
for t, y in enumerate(em2):
    em2[t] = np.nanmean(coviddeathslowvaxmed[t], axis=0)
for t, y in enumerate(em3):
    em3[t] = np.nanmean(coviddeathslowvaxlow[t], axis=0)
figLV = plt.figure(facecolor='w')
axLV = figLV.add_subplot(111, axisbelow=True)
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with high cumulative doses')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium cumulative doses')
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with low cumulative doses')
axLV.set_ylabel('Excess mortality P-score ($\\%$)')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
save_figure(figLV, "coviddeathslowvax.pdf")
axLV.set_title('Excess mortality for countries with low cumulative COVID-19 deaths per million')
save_figure(figLV, "coviddeathslowvax.png")

# MED COVIDDEATHS VS VAX
coviddeathsmedvaxhigh = genfromtxt('coviddeathsmedvaxhigh.txt', delimiter='\t')
coviddeathsmedvaxmed = genfromtxt('coviddeathsmedvaxmed.txt', delimiter='\t')
coviddeathsmedvaxlow = genfromtxt('coviddeathsmedvaxlow.txt', delimiter='\t')
emh = np.empty(len(dates))
emh[:] = np.nan
emm = np.empty(len(dates))
emm[:] = np.nan
eml = np.empty(len(dates))
eml[:] = np.nan
for t, y in enumerate(em1):
    emh[t] = np.nanmean(coviddeathsmedvaxhigh[t], axis=0)
for t, y in enumerate(em2):
    emm[t] = np.nanmean(coviddeathsmedvaxmed[t], axis=0)
for t, y in enumerate(em3):
    eml[t] = np.nanmean(coviddeathsmedvaxlow[t], axis=0)
figLV = plt.figure(facecolor='w')
axLV = figLV.add_subplot(111, axisbelow=True)
axLV.plot(dates, smooth(emh, 28), alpha=0.5, lw=2, label='Countries with high cumulative doses')
axLV.plot(dates, smooth(emm, 28), alpha=0.5, lw=2, label='Countries with medium cumulative doses')
axLV.plot(dates, smooth(eml, 28), alpha=0.5, lw=2, label='Countries with low cumulative doses')
axLV.set_ylabel('Excess mortality P-score ($\\%$)')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
save_figure(figLV, "coviddeathsmedvax.pdf")
axLV.set_title('Excess mortality for countries with medium cumulative COVID-19 deaths per million')
save_figure(figLV, "coviddeathsmedvax.png")

# HIGH COVIDDEATHS VS VAX
coviddeathshighvaxhigh = genfromtxt('coviddeathshighvaxhigh.txt', delimiter='\t')
coviddeathshighvaxmed = genfromtxt('coviddeathshighvaxmed.txt', delimiter='\t')
coviddeathshighvaxlow = genfromtxt('coviddeathshighvaxlow.txt', delimiter='\t')
emh = np.empty(len(dates))
emh[:] = np.nan
emm = np.empty(len(dates))
emm[:] = np.nan
eml = np.empty(len(dates))
eml[:] = np.nan
for t, y in enumerate(em1):
    emh[t] = np.nanmean(coviddeathshighvaxhigh[t], axis=0)
for t, y in enumerate(em2):
    emm[t] = np.nanmean(coviddeathshighvaxmed[t], axis=0)
for t, y in enumerate(em3):
    eml[t] = np.nanmean(coviddeathshighvaxlow[t], axis=0)
figLV = plt.figure(facecolor='w')
axLV = figLV.add_subplot(111, axisbelow=True)
axLV.plot(dates, smooth(emh, 28), alpha=0.5, lw=2, label='Countries with high cumulative doses')
axLV.plot(dates, smooth(emm, 28), alpha=0.5, lw=2, label='Countries with medium cumulative doses')
axLV.plot(dates, smooth(eml, 28), alpha=0.5, lw=2, label='Countries with low cumulative doses')
axLV.set_ylabel('Excess mortality P-score ($\\%$)')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
save_figure(figLV, "coviddeathshighvax.pdf")
axLV.set_title('Excess mortality for countries with high cumulative COVID-19 deaths per million')
save_figure(figLV, "coviddeathshighvax.png")


# COVID DEATHS VS LOCKDOWN VS VAX
coviddeathshighlockdownlowvaxlow = genfromtxt('coviddeathshighlockdownlowvaxlow.txt', delimiter='\t')
coviddeathslowlockdownhighvaxlow = genfromtxt('coviddeathslowlockdownhighvaxlow.txt', delimiter='\t')
coviddeathslowlockdownlowvaxhigh = genfromtxt('coviddeathslowlockdownlowvaxhigh.txt', delimiter='\t')
coviddeathslowlockdownlowvaxlow = genfromtxt('coviddeathslowlockdownlowvaxlow.txt', delimiter='\t')
emhll = np.empty(len(dates))
emhll[:] = np.nan
emlhl = np.empty(len(dates))
emlhl[:] = np.nan
emllh = np.empty(len(dates))
emllh[:] = np.nan
emlll = np.empty(len(dates))
emlll[:] = np.nan
for t, y in enumerate(emlll):
    emlll[t] = np.nanmean(coviddeathslowlockdownlowvaxlow[t], axis=0)
for t, y in enumerate(emhll):
    emhll[t] = np.nanmean(coviddeathshighlockdownlowvaxlow[t], axis=0)
for t, y in enumerate(emlhl):
    emlhl[t] = np.nanmean(coviddeathslowlockdownhighvaxlow[t], axis=0)
for t, y in enumerate(emllh):
    emllh[t] = np.nanmean(coviddeathslowlockdownlowvaxhigh[t], axis=0)
figCDLV = plt.figure(facecolor='w')
axCDLV = figCDLV.add_subplot(111, axisbelow=True)
axCDLV.plot(dates, smooth(emlll, 28), alpha=0.5, lw=2, label='Cumulative deaths low, cumulative stringency low, cumulative doses low')
axCDLV.plot(dates, smooth(emhll, 28), alpha=0.5, lw=2, label='Cumulative deaths high, cumulative stringency low, cumulative doses low')
axCDLV.plot(dates, smooth(emlhl, 28), alpha=0.5, lw=2, label='Cumulative deaths low, cumulative stringency high, cumulative doses low')
axCDLV.plot(dates, smooth(emllh, 28), alpha=0.5, lw=2, label='Cumulative deaths low, cumulative stringency low, cumulative doses high')
axCDLV.set_ylabel('Excess mortality P-score ($\\%$)')
axCDLV.xaxis.set_tick_params(length=0)
axCDLV.yaxis.set_tick_params(length=0)
axCDLV.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
axCDLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axCDLV.fmt_xdata = mdates.DateFormatter('%d %b %Y')
figCDLV.autofmt_xdate()
axCDLV.grid(visible=True)
legend = axCDLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
#    axCDLV.spines[spine].set_visible(False)
save_figure(figCDLV, "coviddeathslockdownvax.pdf")
axCDLV.set_title('Excess mortality by cumulative deaths, stringency and vaccine doses')
save_figure(figCDLV, "coviddeathslockdownvax.png")


# COVID DEATHS
coviddeathshighcd = genfromtxt('coviddeathshighcd.txt', delimiter='\t')
coviddeathshighem = genfromtxt('coviddeathshighem.txt', delimiter='\t')
coviddeathsmedcd = genfromtxt('coviddeathsmedcd.txt', delimiter='\t')
coviddeathsmedem = genfromtxt('coviddeathsmedem.txt', delimiter='\t')
coviddeathslowem = genfromtxt('coviddeathslowem.txt', delimiter='\t')
coviddeathslowcd = genfromtxt('coviddeathslowcd.txt', delimiter='\t')
cdh = np.empty(len(dates))
cdh[:] = np.nan
cdl = np.empty(len(dates))
cdl[:] = np.nan
emh = np.empty(len(dates))
emh[:] = np.nan
eml = np.empty(len(dates))
eml[:] = np.nan
for t, y in enumerate(cdh):
    cdh[t] = np.nanmean(coviddeathshighcd[t], axis=0)
for t, y in enumerate(cdl):
    cdl[t] = np.nanmean(coviddeathslowcd[t], axis=0)
for t, y in enumerate(emh):
    emh[t] = np.nanmean(coviddeathshighem[t], axis=0)
for t, y in enumerate(eml):
    eml[t] = np.nanmean(coviddeathslowem[t], axis=0)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
p1 = ax19.plot(dates, smooth(cdh, 28), alpha=0.5, lw=2, label='COVID-19 death rate for countries with high cumulative COVID-19 deaths per million')
p2 = ax19.plot(dates, smooth(cdl, 28), alpha=0.5, lw=2, label='COVID-19 death rate for countries with low cumulative COVID-19 deaths per million')
p3 = ax20.plot(dates, smooth(emh, 28), alpha=0.5, lw=2, label='Excess mortality for countries with high cumulative COVID-19 deaths per million')
p4 = ax20.plot(dates, smooth(eml, 28), alpha=0.5, lw=2, label='Excess mortality for countries with low cumulative COVID-19 deaths per million')
ax19.set_ylabel('Deaths per million people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax20.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "coviddeaths.pdf")
ax19.set_title('COVID-19 death rate and excess mortality')
save_figure(fig19, "coviddeaths.png")


# LOCKDOWN
lockdownhighem = genfromtxt('lockdownhighem.txt', delimiter='\t')
lockdownhighld = genfromtxt('lockdownhighld.txt', delimiter='\t')
lockdownmedem = genfromtxt('lockdownmedem.txt', delimiter='\t')
lockdownmedld = genfromtxt('lockdownmedld.txt', delimiter='\t')
lockdownlowem = genfromtxt('lockdownlowem.txt', delimiter='\t')
lockdownlowld = genfromtxt('lockdownlowld.txt', delimiter='\t')
ldh = np.empty(len(dates))
ldh[:] = np.nan
ldl = np.empty(len(dates))
ldl[:] = np.nan
emh = np.empty(len(dates))
emh[:] = np.nan
eml = np.empty(len(dates))
eml[:] = np.nan
for t, y in enumerate(ldh):
    ldh[t] = np.nanmean(lockdownhighld[t], axis=0)
for t, y in enumerate(ldl):
    ldl[t] = np.nanmean(lockdownlowld[t], axis=0)
for t, y in enumerate(emh):
    emh[t] = np.nanmean(lockdownhighem[t], axis=0)
for t, y in enumerate(eml):
    eml[t] = np.nanmean(lockdownlowem[t], axis=0)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
p1 = ax19.plot(dates, smooth(ldh, 28), alpha=0.5, lw=2, label='Stringency index: high cumulative stringency')
p2 = ax19.plot(dates, smooth(ldl, 28), alpha=0.5, lw=2, label='Stringency index: low cumulative stringency')
p3 = ax20.plot(dates, smooth(emh, 28), alpha=0.5, lw=2, label='Excess mortality: high cumulative stringency')
p4 = ax20.plot(dates, smooth(eml, 28), alpha=0.5, lw=2, label='Excess mortality: low cumulative stringency')
ax19.set_ylabel('Stringency index (0-100)')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax20.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "lockdown.pdf") # sixth figure in paper
ax19.set_title('Lockdown stringency and excess mortality')
save_figure(fig19, "lockdown.png")



# MASKS
maskshighem = genfromtxt('maskshighem.txt', delimiter='\t')
maskshighma = genfromtxt('maskshighma.txt', delimiter='\t')
masksmedem = genfromtxt('masksmedem.txt', delimiter='\t')
masksmedma = genfromtxt('masksmedma.txt', delimiter='\t')
maskslowem = genfromtxt('maskslowem.txt', delimiter='\t')
maskslowma = genfromtxt('maskslowma.txt', delimiter='\t')
mh = np.empty(len(dates))
mh[:] = np.nan
ml = np.empty(len(dates))
ml[:] = np.nan
emh = np.empty(len(dates))
emh[:] = np.nan
eml = np.empty(len(dates))
eml[:] = np.nan
for t, y in enumerate(mh):
    mh[t] = np.nanmean(maskshighma[t], axis=0)
for t, y in enumerate(ml):
    ml[t] = np.nanmean(maskslowma[t], axis=0)
for t, y in enumerate(emh):
    emh[t] = np.nanmean(maskshighem[t], axis=0)
for t, y in enumerate(eml):
    eml[t] = np.nanmean(maskslowem[t], axis=0)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
p1 = ax19.plot(dates, smooth(mh, 28), alpha=0.5, lw=2, label='Mask index: high cumulative mask policy index')
p2 = ax19.plot(dates, smooth(ml, 28), alpha=0.5, lw=2, label='Mask index: low cumulative mask policy index')
p3 = ax20.plot(dates, smooth(emh, 28), alpha=0.5, lw=2, label='Excess mortality: high cumulative mask policy index')
p4 = ax20.plot(dates, smooth(eml, 28), alpha=0.5, lw=2, label='Excess mortality: low cumulative mask policy index')
ax19.set_ylabel('Face-covering policy index (0-4)')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax20.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "masks.pdf")
ax19.set_title('Face masks')
save_figure(fig19, "masks.png")


# Vax and COVID-19 death rate
vaxhighcd = genfromtxt('vaxhighcd.txt', delimiter='\t')
vaxhighvx = genfromtxt('vaxhighvx.txt', delimiter='\t')
vaxmedcd = genfromtxt('vaxmedcd.txt', delimiter='\t')
vaxmedvx = genfromtxt('vaxmedvx.txt', delimiter='\t')
vaxlowcd = genfromtxt('vaxlowcd.txt', delimiter='\t')
vaxlowvx = genfromtxt('vaxlowvx.txt', delimiter='\t')
vh = np.empty(len(dates))
vh[:] = np.nan
vm = np.empty(len(dates))
vm[:] = np.nan
vl = np.empty(len(dates))
vl[:] = np.nan
cdh = np.empty(len(dates))
cdh[:] = np.nan
cdm = np.empty(len(dates))
cdm[:] = np.nan
cdl = np.empty(len(dates))
cdl[:] = np.nan
for t, y in enumerate(vh):
    vh[t] = np.nanmean(vaxhighvx[t], axis=0)
for t, y in enumerate(vm):
    vm[t] = np.nanmean(vaxmedvx[t], axis=0)
for t, y in enumerate(vl):
    vl[t] = np.nanmean(vaxlowvx[t], axis=0)
for t, y in enumerate(cdh):
    cdh[t] = np.nanmean(vaxhighcd[t], axis=0)
for t, y in enumerate(cdm):
    cdm[t] = np.nanmean(vaxmedcd[t], axis=0)    
for t, y in enumerate(cdl):
    cdl[t] = np.nanmean(vaxlowcd[t], axis=0)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
p1 = ax19.plot(dates, smooth(vh, 28), alpha=0.5, lw=2, label='Daily vaccine doses: high cumulative doses')
p2 = ax19.plot(dates, smooth(vm, 28), alpha=0.5, lw=2, label='Daily vaccine doses: medium cumulative doses')
p3 = ax19.plot(dates, smooth(vl, 28), alpha=0.5, lw=2, label='Daily vaccine doses: low cumulative doses')
p4 = ax20.plot(dates, smooth(cdh, 28), alpha=0.5, lw=2, label='COVID-19 death rate: high cumulative doses')
p5 = ax20.plot(dates, smooth(cdm, 28), alpha=0.5, lw=2, label='COVID-19 death rate: medium cumulative doses')
p6 = ax20.plot(dates, smooth(cdl, 28), alpha=0.5, lw=2, label='COVID-19 death rate: low cumulative doses')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax20.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "vaccinationsCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths')
save_figure(fig19, "vaccinationsCD.png")


for c in range(19):
    d = str(c)
    for a in range(c + 2):
        b = str(a)
        # Vax and COVID-19 deaths clusters
        vaxhighcd = genfromtxt(d+'_clustcd'+b+'_vaxhigh.txt', delimiter='\t')
        vaxhighvx = genfromtxt(d+'_clustcd'+b+'_vaxhigh_vx.txt', delimiter='\t')
        vaxmedcd = genfromtxt(d+'_clustcd'+b+'_vaxmed.txt', delimiter='\t')
        vaxmedvx = genfromtxt(d+'_clustcd'+b+'_vaxmed_vx.txt', delimiter='\t')
        vaxlowcd = genfromtxt(d+'_clustcd'+b+'_vaxlow.txt', delimiter='\t')
        vaxlowvx = genfromtxt(d+'_clustcd'+b+'_vaxlow_vx.txt', delimiter='\t')
        fig19 = plt.figure(facecolor='w')
        ax19 = fig19.add_subplot(111, axisbelow=True)
        ax20 = ax19.twinx()
        sync_colour_cycle(ax20, ax19)
        if vaxhighvx.size > 0:
            cd1 = np.empty(len(dates))
            cd1[:] = np.nan
            for t, y in enumerate(cd1):
                cd1[t] = np.nanmean(vaxhighvx[t], axis=0)
            p1 = ax19.plot(dates, smooth(cd1, 28), alpha=0.5, lw=2, label='Daily vaccine doses: high cumulative doses') # Vaccination rate in highly vaccinated countries
        if vaxmedvx.size > 0:
            cd2 = np.empty(len(dates))
            cd2[:] = np.nan
            for t, y in enumerate(cd2):
                cd2[t] = np.nanmean(vaxmedvx[t], axis=0)
            p2 = ax19.plot(dates, smooth(cd2, 28), alpha=0.5, lw=2, label='Daily vaccine doses: medium cumulative doses') # Vaccination rate in medium vaccinated countries
        if vaxlowvx.size > 0:
            cd3 = np.empty(len(dates))
            cd3[:] = np.nan
            for t, y in enumerate(cd3):
                cd3[t] = np.nanmean(vaxlowvx[t], axis=0)
            p3 = ax19.plot(dates, smooth(cd3, 28), alpha=0.5, lw=2, label='Daily vaccine doses: low cumulative doses') # Vaccination rate in low vaccinated countries
        if vaxhighcd.size > 0:
            cd4 = np.empty(len(dates))
            cd4[:] = np.nan
            for t, y in enumerate(cd4):
                cd4[t] = np.nanmean(vaxhighcd[t], axis=0)
            p4 = ax20.plot(dates, smooth(cd4, 28), alpha=0.5, lw=2, label='COVID-19 death rate: high cumulative doses') # COVID-19 deaths for highly vaccinated countries
        if vaxmedcd.size > 0:
            cd5 = np.empty(len(dates))
            cd5[:] = np.nan
            for t, y in enumerate(cd5):
                cd5[t] = np.nanmean(vaxmedcd[t], axis=0)
            p5 = ax20.plot(dates, smooth(cd5, 28), alpha=0.5, lw=2, label='COVID-19 death rate: medium cumulative doses') # COVID-19 deaths for medium vaccinated countries
        if vaxlowcd.size > 0:
            cd6 = np.empty(len(dates))
            cd6[:] = np.nan
            for t, y in enumerate(cd6):
                cd6[t] = np.nanmean(vaxlowcd[t], axis=0)
            p6 = ax20.plot(dates, smooth(cd6, 28), alpha=0.5, lw=2, label='COVID-19 death rate: low cumulative doses') # COVID-19 deaths for low vaccinated countries
        if (vaxhighvx.size > 0 or vaxmedvx.size > 0 or vaxlowvx.size > 0 or vaxhighcd.size > 0 or vaxmedcd.size > 0 or vaxlowcd.size > 0):
            ax19.set_ylabel('Doses per 100 people per day')
            ax20.set_ylabel('Deaths per million people per day')
            ax19.yaxis.set_tick_params(length=0)
            ax20.xaxis.set_tick_params(length=0)
            ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
            ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
            fig19.autofmt_xdate()
            ax19.grid(visible=True)
            legend.get_frame().set_alpha(1)
            lines, labels = ax19.get_legend_handles_labels()
            lines2, labels2 = ax20.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
            save_figure(fig19, d+"_vaccinationsCD"+b+".pdf")
            titlecd = 'Vaccinations and COVID-19 deaths: cluster '+str(a+1)+' of '+str(c+2)
            ax19.set_title(titlecd)
            save_figure(fig19, d+"_vaccinationsCD"+b+".png")
            plt.close('all')


        vaxhighem = genfromtxt(d+'_clustem'+b+'_vaxhigh.txt', delimiter='\t')
        vaxhighvx = genfromtxt(d+'_clustem'+b+'_vaxhigh_vx.txt', delimiter='\t')
        vaxmedem = genfromtxt(d+'_clustem'+b+'_vaxmed.txt', delimiter='\t')
        vaxmedvx = genfromtxt(d+'_clustem'+b+'_vaxmed_vx.txt', delimiter='\t')
        vaxlowem = genfromtxt(d+'_clustem'+b+'_vaxlow.txt', delimiter='\t')
        vaxlowvx = genfromtxt(d+'_clustem'+b+'_vaxlow_vx.txt', delimiter='\t')
        fig19 = plt.figure(facecolor='w')
        ax19 = fig19.add_subplot(111, axisbelow=True)
        ax20 = ax19.twinx()
        sync_colour_cycle(ax20, ax19)
        if vaxhighvx.size > 0:
            em1 = np.empty(len(dates))
            em1[:] = np.nan
            for t, y in enumerate(em1):
                em1[t] = np.nanmean(vaxhighvx[t], axis=0)
            p1 = ax19.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Daily vaccine doses: high cumulative doses') # Vaccination rate in highly vaccinated countries
        if vaxmedvx.size > 0:
            em2 = np.empty(len(dates))
            em2[:] = np.nan
            for t, y in enumerate(em2):
                em2[t] = np.nanmean(vaxmedvx[t], axis=0)
            p2 = ax19.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Daily vaccine doses: medium cumulative doses') # Vaccination rate in medium vaccinated countries
        if vaxlowvx.size > 0:
            em3 = np.empty(len(dates))
            em3[:] = np.nan
            for t, y in enumerate(em3):
                em3[t] = np.nanmean(vaxlowvx[t], axis=0)
            p3 = ax19.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Daily vaccine doses: low cumulative doses') # Vaccination rate in low vaccinated countries
        if vaxhighem.size > 0:
            em4 = np.empty(len(dates))
            em4[:] = np.nan
            for t, y in enumerate(em4):
                em4[t] = np.nanmean(vaxhighem[t], axis=0)
            p4 = ax20.plot(dates, smooth(em4, 28), alpha=0.5, lw=2, label='Excess mortality: high cumulative doses') # Excess mortality for highly vaccinated countries
        if vaxmedem.size > 0:
            em5 = np.empty(len(dates))
            em5[:] = np.nan
            for t, y in enumerate(em5):
                em5[t] = np.nanmean(vaxmedem[t], axis=0)
            p5 = ax20.plot(dates, smooth(em5, 28), alpha=0.5, lw=2, label='Excess mortality: medium cumulative doses') # Excess mortality for medium vaccinated countries
        if vaxlowem.size > 0:
            em6 = np.empty(len(dates))
            em6[:] = np.nan
            for t, y in enumerate(em6):
                em6[t] = np.nanmean(vaxlowem[t], axis=0)
            p6 = ax20.plot(dates, smooth(em6, 28), alpha=0.5, lw=2, label='Excess mortality: low cumulative doses') # Excess mortality for low vaccinated countries
        if (vaxhighvx.size > 0 or vaxmedvx.size > 0 or vaxlowvx.size > 0 or vaxhighem.size > 0 or vaxmedem.size > 0 or vaxlowem.size > 0):
            ax19.set_ylabel('Doses per 100 people per day')
            ax20.set_ylabel('Excess mortality P-score ($\\%$)')
            ax19.yaxis.set_tick_params(length=0)
            ax20.xaxis.set_tick_params(length=0)
            ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
            ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
            fig19.autofmt_xdate()
            ax19.grid(visible=True)
            legend.get_frame().set_alpha(1)
            lines, labels = ax19.get_legend_handles_labels()
            lines2, labels2 = ax20.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
            save_figure(fig19, d+"_vaccinationsEM"+b+".pdf")
            titleem = 'Vaccinations and excess mortality: cluster '+str(a+1)+' of '+str(c+2)
            ax19.set_title(titleem)
            save_figure(fig19, d+"_vaccinationsEM"+b+".png")
            plt.close('all')


# Vax and excess mortality
vaxhighem = genfromtxt('vaxhighem.txt', delimiter='\t')
vaxhighvx = genfromtxt('vaxhighvx.txt', delimiter='\t')
vaxmedem = genfromtxt('vaxmedem.txt', delimiter='\t')
vaxmedvx = genfromtxt('vaxmedvx.txt', delimiter='\t')
vaxlowem = genfromtxt('vaxlowem.txt', delimiter='\t')
vaxlowvx = genfromtxt('vaxlowvx.txt', delimiter='\t')
vh = np.empty(len(dates))
vh[:] = np.nan
vm = np.empty(len(dates))
vm[:] = np.nan
vl = np.empty(len(dates))
vl[:] = np.nan
emh = np.empty(len(dates))
emh[:] = np.nan
emm = np.empty(len(dates))
emm[:] = np.nan
eml = np.empty(len(dates))
eml[:] = np.nan
for t, y in enumerate(vh):
    vh[t] = np.nanmean(vaxhighvx[t], axis=0)
for t, y in enumerate(vm):
    vm[t] = np.nanmean(vaxmedvx[t], axis=0)
for t, y in enumerate(vl):
    vl[t] = np.nanmean(vaxlowvx[t], axis=0)
for t, y in enumerate(emh):
    emh[t] = np.nanmean(vaxhighem[t], axis=0)
for t, y in enumerate(emm):
    emm[t] = np.nanmean(vaxmedem[t], axis=0)    
for t, y in enumerate(eml):
    eml[t] = np.nanmean(vaxlowem[t], axis=0)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
p1 = ax19.plot(dates, smooth(vh, 28), alpha=0.5, lw=2, label='Daily vaccine doses: high cumulative doses')
p2 = ax19.plot(dates, smooth(vm, 28), alpha=0.5, lw=2, label='Daily vaccine doses: medium cumulative doses')
p3 = ax19.plot(dates, smooth(vl, 28), alpha=0.5, lw=2, label='Daily vaccine doses: low cumulative doses')
p4 = ax20.plot(dates, smooth(emh, 28), alpha=0.5, lw=2, label='Excess mortality: high cumulative doses')
p5 = ax20.plot(dates, smooth(emm, 28), alpha=0.5, lw=2, label='Excess mortality: medium cumulative doses')
p6 = ax20.plot(dates, smooth(eml, 28), alpha=0.5, lw=2, label='Excess mortality: low cumulative doses')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax20.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "vaccinationsEM.pdf")
ax19.set_title('Vaccinations and excess mortality')
save_figure(fig19, "vaccinationsEM.png")
plt.close('all')


with open('countrylist.txt', 'r') as file:
    countries = file.read().splitlines()

for i, c in enumerate(countries):
    cf = c.replace(" ", "_")
    CNTRYdata = genfromtxt(cf + '.txt', delimiter='\t', converters = {0: date_parser})
    dates = [row[0] for row in CNTRYdata]
    CNTRYlockdown = [row[1] for row in CNTRYdata]
    CNTRYmasks = [row[2] for row in CNTRYdata]
    CNTRYvax = [row[3] for row in CNTRYdata]
    CNTRYcoviddeaths = [row[4] for row in CNTRYdata]
    CNTRYexcessmortality = [row[5] for row in CNTRYdata]
    CNTRYlockdown0 = [ 0 if np.isnan(x) else x for x in CNTRYlockdown ]
    CNTRYmasks0 = [ 0 if np.isnan(x) else x for x in CNTRYmasks ]
    CNTRYvax0 = [ 0 if np.isnan(x) else x for x in CNTRYvax ]
    CNTRYcoviddeaths0 = [ 0 if np.isnan(x) else x for x in CNTRYcoviddeaths ]
    CNTRYexcessmortality0 = [ 0 if np.isnan(x) else x for x in CNTRYexcessmortality ]
    
    if (not all(val == 0 for val in CNTRYlockdown0)) or (not all(val == 0 for val in CNTRYmasks0)) or (not all(val == 0 for val in CNTRYvax0)) or (not all(val == 0 for val in CNTRYcoviddeaths0)) or (not all(val == 0 for val in CNTRYexcessmortality0)):
        figc = plt.figure(facecolor='w')
        axc = figc.add_subplot(111, axisbelow=True)
        CNTRY1 = CNTRY2 = CNTRY3 = CNTRY4 = CNTRY5 = []
        if sum(CNTRYlockdown0) > 0:
            axc.plot(dates, CNTRYlockdown, lw=2, label='Stringency index (0-100)', color='grey')
        if sum(CNTRYmasks0) > 0:
            axc.plot(dates, np.asarray(CNTRYmasks)*10, lw=2, label='Face-covering index (0-4), multiplied by 10', color='c', linestyle='--')
        if sum(CNTRYvax0) > 0:
            axc.plot(dates, smooth(np.asarray(CNTRYvax)*100, 28), lw=2, label='Vaccine doses per 10,000 people per day', color='b', linestyle='-.')
        if sum(CNTRYcoviddeaths0) > 0:
            axc.plot(dates, smooth(np.asarray(CNTRYcoviddeaths)*10, 28), lw=2, label='Deaths per million people per day, multiplied by 10', color='k', linestyle=':')
        axc.set_ylabel('Scaled series (units and multipliers in legend)')

        if not all(val == 0 for val in CNTRYexcessmortality0):
            axem = axc.twinx()
            CNTRY5 = axem.plot(dates, smooth(CNTRYexcessmortality, 28), alpha=0.5, lw=2, label='Excess mortality', color='r')
            sync_colour_cycle(axem, axc)
            axem.set_ylabel('Excess mortality P-score ($\\%$)')
            lines, labels = axc.get_legend_handles_labels()
            lines2, labels2 = axem.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
        else:
            lines, labels = axc.get_legend_handles_labels()
            plt.legend(lines, labels, fancybox=True, framealpha=0.8)
        axc.yaxis.set_tick_params(length=0)
        axc.xaxis.set_tick_params(length=0)
        axc.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        if (c == 'Falkland Islands') or (c == 'Saint Helena'):
            axc.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        else:
            axc.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axc.fmt_xdata = mdates.DateFormatter('%d %b %Y')
        figc.autofmt_xdate()
        axc.grid(visible=True)
        legend.get_frame().set_alpha(1)
        save_figure(figc, cf + ".pdf")
        axc.set_title(c)
        save_figure(figc, cf + ".png")
        plt.close('all')

# Sweden United Kingdom lockdowns
SWEGBRdata = genfromtxt('SWEGBR.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in SWEGBRdata]
lockdown_SWE = [row[1] for row in SWEGBRdata]
lockdown_GBR = [row[2] for row in SWEGBRdata]
deaths_SWE = [row[3] for row in SWEGBRdata]
deaths_GBR = [row[4] for row in SWEGBRdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
SWEl = ax19.plot(dates, lockdown_SWE, alpha=0.5, lw=2, label='Sweden: stringency')
GBRl = ax19.plot(dates, lockdown_GBR, alpha=0.5, lw=2, label='UK: stringency')
SWEem = ax20.plot(dates, smooth(deaths_SWE, 28), alpha=0.5, lw=2, label='Sweden: excess mortality')
GBRem = ax20.plot(dates, smooth(deaths_GBR, 28), alpha=0.5, lw=2, label='UK: excess mortality')
ax19.set_ylabel('Stringency index (0-100)')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "SWEGBR.pdf")
ax19.set_title('Lockdowns and excess mortality for Sweden and the United Kingdom')
save_figure(fig19, "SWEGBR.png")




######################
# PAIRS OF COUNTRIES #
######################

DZAEGYdata = genfromtxt('AlgeriaEgyptCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in DZAEGYdata]
vax_DZA = [row[1] for row in DZAEGYdata]
vax_EGY = [row[2] for row in DZAEGYdata]
deaths_DZA = [row[3] for row in DZAEGYdata]
deaths_EGY = [row[4] for row in DZAEGYdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
DZAv = ax19.plot(dates, smooth(vax_DZA, 28), alpha=0.5, lw=2, label='Algeria vaccination rate')
EGYv = ax19.plot(dates, smooth(vax_EGY, 28), alpha=0.5, lw=2, label='Egypt vaccination rate')
DZAcd = ax20.plot(dates, smooth(deaths_DZA, 28), alpha=0.5, lw=2, label='Algeria COVID-19 death rate')
EGYcd = ax20.plot(dates, smooth(deaths_EGY, 28), alpha=0.5, lw=2, label='Egypt COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "AlgeriaEgyptCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Algeria and Egypt')
save_figure(fig19, "AlgeriaEgyptCD.png")
plt.close('all')

DZAEGYdata = genfromtxt('AlgeriaEgyptEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in DZAEGYdata]
vax_DZA = [row[1] for row in DZAEGYdata]
vax_EGY = [row[2] for row in DZAEGYdata]
deaths_DZA = [row[3] for row in DZAEGYdata]
deaths_EGY = [row[4] for row in DZAEGYdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
DZAv = ax19.plot(dates, smooth(vax_DZA, 28), alpha=0.5, lw=2, label='Algeria vaccination rate')
EGYv = ax19.plot(dates, smooth(vax_EGY, 28), alpha=0.5, lw=2, label='Egypt vaccination rate')
DZAem = ax20.plot(dates, smooth(deaths_DZA, 28), alpha=0.5, lw=2, label='Algeria excess mortality')
EGYem = ax20.plot(dates, smooth(deaths_EGY, 28), alpha=0.5, lw=2, label='Egypt excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "AlgeriaEgyptEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Algeria and Egypt')
save_figure(fig19, "AlgeriaEgyptEM.png")
plt.close('all')

ATGCUBdata = genfromtxt('AntiguaandBarbudaCubaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in ATGCUBdata]
vax_ATG = [row[1] for row in ATGCUBdata]
vax_CUB = [row[2] for row in ATGCUBdata]
deaths_ATG = [row[3] for row in ATGCUBdata]
deaths_CUB = [row[4] for row in ATGCUBdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
ATGv = ax19.plot(dates, smooth(vax_ATG, 28), alpha=0.5, lw=2, label='Antigua and Barbuda vaccination rate')
CUBv = ax19.plot(dates, smooth(vax_CUB, 28), alpha=0.5, lw=2, label='Cuba vaccination rate')
ATGem = ax20.plot(dates, smooth(deaths_ATG, 28), alpha=0.5, lw=2, label='Antigua and Barbuda excess mortality')
CUBem = ax20.plot(dates, smooth(deaths_CUB, 28), alpha=0.5, lw=2, label='Cuba excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "AntiguaandBarbudaCubaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Antigua and Barbuda and Cuba')
save_figure(fig19, "AntiguaandBarbudaCubaEM.png")
plt.close('all')

ARGPRYdata = genfromtxt('ArgentinaParaguayEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in ARGPRYdata]
vax_ARG = [row[1] for row in ARGPRYdata]
vax_PRY = [row[2] for row in ARGPRYdata]
deaths_ARG = [row[3] for row in ARGPRYdata]
deaths_PRY = [row[4] for row in ARGPRYdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
ARGv = ax19.plot(dates, smooth(vax_ARG, 28), alpha=0.5, lw=2, label='Argentina vaccination rate')
PRYv = ax19.plot(dates, smooth(vax_PRY, 28), alpha=0.5, lw=2, label='Paraguay vaccination rate')
ARGem = ax20.plot(dates, smooth(deaths_ARG, 28), alpha=0.5, lw=2, label='Argentina excess mortality')
PRYem = ax20.plot(dates, smooth(deaths_PRY, 28), alpha=0.5, lw=2, label='Paraguay excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "ArgentinaParaguayEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Argentina and Paraguay')
save_figure(fig19, "ArgentinaParaguayEM.png")
plt.close('all')

ARMAZEdata = genfromtxt('ArmeniaAzerbaijanCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in ARMAZEdata]
vax_ARM = [row[1] for row in ARMAZEdata]
vax_AZE = [row[2] for row in ARMAZEdata]
deaths_ARM = [row[3] for row in ARMAZEdata]
deaths_AZE = [row[4] for row in ARMAZEdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
ARMv = ax19.plot(dates, smooth(vax_ARM, 28), alpha=0.5, lw=2, label='Armenia vaccination rate')
AZEv = ax19.plot(dates, smooth(vax_AZE, 28) , alpha=0.5, lw=2, label='Azerbaijan vaccination rate')
ARMcd = ax20.plot(dates, smooth(deaths_ARM, 28), alpha=0.5, lw=2, label='Armenia COVID-19 death rate')
AZEcd = ax20.plot(dates, smooth(deaths_AZE, 28), alpha=0.5, lw=2, label='Azerbaijan COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "ArmeniaAzerbaijanCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Armenia and Azerbaijan')
save_figure(fig19, "ArmeniaAzerbaijanCD.png")
plt.close('all')

ARMAZEdata = genfromtxt('ArmeniaAzerbaijanEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in ARMAZEdata]
vax_ARM = [row[1] for row in ARMAZEdata]
vax_AZE = [row[2] for row in ARMAZEdata]
deaths_ARM = [row[3] for row in ARMAZEdata]
deaths_AZE = [row[4] for row in ARMAZEdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
ARMv = ax19.plot(dates, smooth(vax_ARM, 28), alpha=0.5, lw=2, label='Armenia vaccination rate')
AZEv = ax19.plot(dates, smooth(vax_AZE, 28) , alpha=0.5, lw=2, label='Azerbaijan vaccination rate')
ARMem = ax20.plot(dates, smooth(deaths_ARM, 28), alpha=0.5, lw=2, label='Armenia excess mortality')
AZEem = ax20.plot(dates, smooth(deaths_AZE, 28), alpha=0.5, lw=2, label='Azerbaijan excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "ArmeniaAzerbaijanEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Armenia and Azerbaijan')
save_figure(fig19, "ArmeniaAzerbaijanEM.png")
plt.close('all')

AUSNZLdata = genfromtxt('AustraliaNewZealandCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in AUSNZLdata]
vax_AUS = [row[1] for row in AUSNZLdata]
vax_NZL = [row[2] for row in AUSNZLdata]
deaths_AUS = [row[3] for row in AUSNZLdata]
deaths_NZL = [row[4] for row in AUSNZLdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
AUSv = ax19.plot(dates, smooth(vax_AUS, 28), alpha=0.5, lw=2, label='Australia vaccination rate')
NZLv = ax19.plot(dates, smooth(vax_NZL, 28), alpha=0.5, lw=2, label='New Zealand vaccination rate')
AUScd = ax20.plot(dates, smooth(deaths_AUS, 28), alpha=0.5, lw=2, label='Australia COVID-19 death rate')
NZLcd = ax20.plot(dates, smooth(deaths_NZL, 28), alpha=0.5, lw=2, label='New Zealand COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "AustraliaNewZealandCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Australia and New Zealand')
save_figure(fig19, "AustraliaNewZealandCD.png")
plt.close('all')

AUSNZLdata = genfromtxt('AustraliaNewZealandEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in AUSNZLdata]
vax_AUS = [row[1] for row in AUSNZLdata]
vax_NZL = [row[2] for row in AUSNZLdata]
deaths_AUS = [row[3] for row in AUSNZLdata]
deaths_NZL = [row[4] for row in AUSNZLdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
AUSv = ax19.plot(dates, smooth(vax_AUS, 28), alpha=0.5, lw=2, label='Australia vaccination rate')
NZLv = ax19.plot(dates, smooth(vax_NZL, 28), alpha=0.5, lw=2, label='New Zealand vaccination rate')
AUSem = ax20.plot(dates, smooth(deaths_AUS, 28), alpha=0.5, lw=2, label='Australia excess mortality')
NZLem = ax20.plot(dates, smooth(deaths_NZL, 28), alpha=0.5, lw=2, label='New Zealand excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "AustraliaNewZealandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Australia and New Zealand')
save_figure(fig19, "AustraliaNewZealandEM.png")
plt.close('all')

BLRDNKdata = genfromtxt('BelarusDenmarkCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in BLRDNKdata]
vax_BLR = [row[1] for row in BLRDNKdata]
vax_DNK = [row[2] for row in BLRDNKdata]
deaths_BLR = [row[3] for row in BLRDNKdata]
deaths_DNK = [row[4] for row in BLRDNKdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
BLRv = ax19.plot(dates, smooth(vax_BLR, 28), alpha=0.5, lw=2, label='Belarus vaccination rate')
DNKv = ax19.plot(dates, smooth(vax_DNK, 28), alpha=0.5, lw=2, label='Denmark vaccination rate')
BLRcd = ax20.plot(dates, smooth(deaths_BLR, 28), alpha=0.5, lw=2, label='Belarus COVID-19 death rate')
DNKcd = ax20.plot(dates, smooth(deaths_DNK, 28), alpha=0.5, lw=2, label='Denmark COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "BelarusDenmarkCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Belarus and Denmark')
save_figure(fig19, "BelarusDenmarkCD.png")
plt.close('all')

BLRDNKdata = genfromtxt('BelarusDenmarkEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in BLRDNKdata]
vax_BLR = [row[1] for row in BLRDNKdata]
vax_DNK = [row[2] for row in BLRDNKdata]
deaths_BLR = [row[3] for row in BLRDNKdata]
deaths_DNK = [row[4] for row in BLRDNKdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
BLRv = ax19.plot(dates, smooth(vax_BLR, 28), alpha=0.5, lw=2, label='Belarus vaccination rate')
DNKv = ax19.plot(dates, smooth(vax_DNK, 28), alpha=0.5, lw=2, label='Denmark vaccination rate')
BLRem = ax20.plot(dates, smooth(deaths_BLR, 28), alpha=0.5, lw=2, label='Belarus excess mortality')
DNKem = ax20.plot(dates, smooth(deaths_DNK, 28), alpha=0.5, lw=2, label='Denmark excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "BelarusDenmarkEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Belarus and Denmark')
save_figure(fig19, "BelarusDenmarkEM.png")
plt.close('all')

BTNSGPdata = genfromtxt('BhutanSingaporeCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in BTNSGPdata]
vax_BTN = [row[1] for row in BTNSGPdata]
vax_SGP = [row[2] for row in BTNSGPdata]
deaths_BTN = [row[3] for row in BTNSGPdata]
deaths_SGP = [row[4] for row in BTNSGPdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
BTNv = ax19.plot(dates, smooth(vax_BTN, 28), alpha=0.5, lw=2, label='Bhutan vaccination rate')
SGPv = ax19.plot(dates, smooth(vax_SGP, 28), alpha=0.5, lw=2, label='Singapore vaccination rate')
BTNcd = ax20.plot(dates, smooth(deaths_BTN, 28), alpha=0.5, lw=2, label='Bhutan COVID-19 death rate')
SGPcd = ax20.plot(dates, smooth(deaths_SGP, 28), alpha=0.5, lw=2, label='Singapore COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "BhutanSingaporeCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Bhutan and Singapore')
save_figure(fig19, "BhutanSingaporeCD.png")
plt.close('all')

BTNVNMdata = genfromtxt('BhutanVietnamCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in BTNVNMdata]
vax_BTN = [row[1] for row in BTNVNMdata]
vax_VNM = [row[2] for row in BTNVNMdata]
deaths_BTN = [row[3] for row in BTNVNMdata]
deaths_VNM = [row[4] for row in BTNVNMdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
BTNv = ax19.plot(dates, smooth(vax_BTN, 28), alpha=0.5, lw=2, label='Bhutan vaccination rate')
VNMv = ax19.plot(dates, smooth(vax_VNM, 28), alpha=0.5, lw=2, label='Vietnam vaccination rate')
BTNcd = ax20.plot(dates, smooth(deaths_BTN, 28), alpha=0.5, lw=2, label='Bhutan COVID-19 death rate')
VNMcd = ax20.plot(dates, smooth(deaths_VNM, 28), alpha=0.5, lw=2, label='Vietnam COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "BhutanVietnamCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Bhutan and Vietnam')
save_figure(fig19, "BhutanVietnamCD.png")
plt.close('all')

BTNSGPdata = genfromtxt('BhutanSingaporeEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in BTNSGPdata]
vax_BTN = [row[1] for row in BTNSGPdata]
vax_SGP = [row[2] for row in BTNSGPdata]
deaths_BTN = [row[3] for row in BTNSGPdata]
deaths_SGP = [row[4] for row in BTNSGPdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
BTNv = ax19.plot(dates, smooth(vax_BTN, 28), alpha=0.5, lw=2, label='Bhutan vaccination rate')
SGPv = ax19.plot(dates, smooth(vax_SGP, 28), alpha=0.5, lw=2, label='Singapore vaccination rate')
BTNem = ax20.plot(dates, smooth(deaths_BTN, 28), alpha=0.5, lw=2, label='Bhutan excess mortality')
SGPem = ax20.plot(dates, smooth(deaths_SGP, 28), alpha=0.5, lw=2, label='Singapore excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "BhutanSingaporeEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Bhutan and Singapore')
save_figure(fig19, "BhutanSingaporeEM.png")
plt.close('all')

BTNTHAdata = genfromtxt('BhutanThailandCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in BTNTHAdata]
vax_BTN = [row[1] for row in BTNTHAdata]
vax_THA = [row[2] for row in BTNTHAdata]
deaths_BTN = [row[3] for row in BTNTHAdata]
deaths_THA = [row[4] for row in BTNTHAdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
BTNv = ax19.plot(dates, smooth(vax_BTN, 28), alpha=0.5, lw=2, label='Bhutan vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
BTNcd = ax20.plot(dates, smooth(deaths_BTN, 28), alpha=0.5, lw=2, label='Bhutan COVID-19 death rate')
THAcd = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "BhutanThailandCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Bhutan and Thailand')
save_figure(fig19, "BhutanThailandCD.png")
plt.close('all')

BTNTHAdata = genfromtxt('BhutanThailandEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in BTNTHAdata]
vax_BTN = [row[1] for row in BTNTHAdata]
vax_THA = [row[2] for row in BTNTHAdata]
deaths_BTN = [row[3] for row in BTNTHAdata]
deaths_THA = [row[4] for row in BTNTHAdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
BTNv = ax19.plot(dates, smooth(vax_BTN, 28), alpha=0.5, lw=2, label='Bhutan vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
BTNem = ax20.plot(dates, smooth(deaths_BTN, 28), alpha=0.5, lw=2, label='Bhutan excess mortality')
THAem = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "BhutanThailandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Bhutan and Thailand')
save_figure(fig19, "BhutanThailandEM.png")
plt.close('all')

BIHROUdata = genfromtxt('BosniaandHerzegovinaRomaniaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in BIHROUdata]
vax_BIH = [row[1] for row in BIHROUdata]
vax_ROU = [row[2] for row in BIHROUdata]
deaths_BIH = [row[3] for row in BIHROUdata]
deaths_ROU = [row[4] for row in BIHROUdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
BIHv = ax19.plot(dates, smooth(vax_BIH, 28), alpha=0.5, lw=2, label='Bosnia and Herzegovina vaccination rate')
ROUv = ax19.plot(dates, smooth(vax_ROU, 28), alpha=0.5, lw=2, label='Romania vaccination rate')
BIHem = ax20.plot(dates, smooth(deaths_BIH, 28), alpha=0.5, lw=2, label='Bosnia and Herzegovina excess mortality')
ROUem = ax20.plot(dates, smooth(deaths_ROU, 28), alpha=0.5, lw=2, label='Romania excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "BosniaandHerzegovinaRomaniaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Bosnia and Herzegovina and Romania')
save_figure(fig19, "BosniaandHerzegovinaRomaniaEM.png")
plt.close('all')

BGRSRBdata = genfromtxt('BulgariaSerbiaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in BGRSRBdata]
vax_BGR = [row[1] for row in BGRSRBdata]
vax_SRB = [row[2] for row in BGRSRBdata]
deaths_BGR = [row[3] for row in BGRSRBdata]
deaths_SRB = [row[4] for row in BGRSRBdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
BGRv = ax19.plot(dates, smooth(vax_BGR, 28), alpha=0.5, lw=2, label='Bulgaria vaccination rate')
SRBv = ax19.plot(dates, smooth(vax_SRB, 28), alpha=0.5, lw=2, label='Serbia vaccination rate')
BGRem = ax20.plot(dates, smooth(deaths_BGR, 28), alpha=0.5, lw=2, label='Bulgaria excess mortality')
SRBem = ax20.plot(dates, smooth(deaths_SRB, 28), alpha=0.5, lw=2, label='Serbia excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "BulgariaSerbiaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Bulgaria and Serbia')
save_figure(fig19, "BulgariaSerbiaEM.png")
plt.close('all')

BDIERIdata = genfromtxt('BurundiEritreaCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in BDIERIdata]
vax_BDI = [row[1] for row in BDIERIdata]
vax_ERI = [row[2] for row in BDIERIdata]
deaths_BDI = [row[3] for row in BDIERIdata]
deaths_ERI = [row[4] for row in BDIERIdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
BDIv = ax19.plot(dates, smooth(vax_BDI, 28), alpha=0.5, lw=2, label='Burundi vaccination rate')
ERIv = ax19.plot(dates, smooth(vax_ERI, 28) , alpha=0.5, lw=2, label='Eritrea vaccination rate')
BDIcd = ax20.plot(dates, smooth(deaths_BDI, 28), alpha=0.5, lw=2, label='Burundi COVID-19 death rate')
ERIcd = ax20.plot(dates, smooth(deaths_ERI, 28), alpha=0.5, lw=2, label='Eritrea COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "BurundiEritreaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Burundi and Eritrea')
save_figure(fig19, "BurundiEritreaCD.png")
plt.close('all')

BDISYCdata = genfromtxt('BurundiSeychellesCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in BDISYCdata]
vax_BDI = [row[1] for row in BDISYCdata]
vax_SYC = [row[2] for row in BDISYCdata]
deaths_BDI = [row[3] for row in BDISYCdata]
deaths_SYC = [row[4] for row in BDISYCdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
BDIv = ax19.plot(dates, smooth(vax_BDI, 28), alpha=0.5, lw=2, label='Burundi vaccination rate')
SYCv = ax19.plot(dates, smooth(vax_SYC, 28), alpha=0.5, lw=2, label='Seychelles vaccination rate')
BDIcd = ax20.plot(dates, smooth(deaths_BDI, 28), alpha=0.5, lw=2, label='Burundi COVID-19 death rate')
SYCcd = ax20.plot(dates, smooth(deaths_SYC, 28), alpha=0.5, lw=2, label='Seychelles COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "BurundiSeychellesCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Burundi and Seychelles')
save_figure(fig19, "BurundiSeychellesCD.png")
plt.close('all')

KHMVNMdata = genfromtxt('CambodiaVietnamCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in KHMVNMdata]
vax_KHM = [row[1] for row in KHMVNMdata]
vax_VNM = [row[2] for row in KHMVNMdata]
deaths_KHM = [row[3] for row in KHMVNMdata]
deaths_VNM = [row[4] for row in KHMVNMdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
KHMv = ax19.plot(dates, smooth(vax_KHM, 28), alpha=0.5, lw=2, label='Cambodia vaccination rate')
VNMv = ax19.plot(dates, smooth(vax_VNM, 28), alpha=0.5, lw=2, label='Vietnam vaccination rate')
KHMcd = ax20.plot(dates, smooth(deaths_KHM, 28), alpha=0.5, lw=2, label='Cambodia COVID-19 death rate')
VNMcd = ax20.plot(dates, smooth(deaths_VNM, 28), alpha=0.5, lw=2, label='Vietnam COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CambodiaVietnamCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Cambodia and Vietnam')
save_figure(fig19, "CambodiaVietnamCD.png")
plt.close('all')

CANCUBdata = genfromtxt('CanadaCubaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in CANCUBdata]
vax_CAN = [row[1] for row in CANCUBdata]
vax_CUB = [row[2] for row in CANCUBdata]
deaths_CAN = [row[3] for row in CANCUBdata]
deaths_CUB = [row[4] for row in CANCUBdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
CANv = ax19.plot(dates, smooth(vax_CAN, 28), alpha=0.5, lw=2, label='Canada vaccination rate')
CUBv = ax19.plot(dates, smooth(vax_CUB, 28), alpha=0.5, lw=2, label='Cuba vaccination rate')
CANem = ax20.plot(dates, smooth(deaths_CAN, 28), alpha=0.5, lw=2, label='Canada excess mortality')
CUBem = ax20.plot(dates, smooth(deaths_CUB, 28), alpha=0.5, lw=2, label='Cuba excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CanadaCubaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Canada and Cuba')
save_figure(fig19, "CanadaCubaEM.png")
plt.close('all')

CANVCTdata = genfromtxt('CanadaSaintVincentandtheGrenadinesEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in CANVCTdata]
vax_CAN = [row[1] for row in CANVCTdata]
vax_VCT = [row[2] for row in CANVCTdata]
deaths_CAN = [row[3] for row in CANVCTdata]
deaths_VCT = [row[4] for row in CANVCTdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
CANv = ax19.plot(dates, smooth(vax_CAN, 28), alpha=0.5, lw=2, label='Canada vaccination rate')
VCTv = ax19.plot(dates, smooth(vax_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines vaccination rate')
CANem = ax20.plot(dates, smooth(deaths_CAN, 28), alpha=0.5, lw=2, label='Canada excess mortality')
VCTem = ax20.plot(dates, smooth(deaths_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CanadaSaintVincentandtheGrenadinesEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Canada and Saint Vincent and the Grenadines')
save_figure(fig19, "CanadaSaintVincentandtheGrenadinesEM.png")
plt.close('all')

CPVTUNdata = genfromtxt('CapeVerdeTunisiaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in CPVTUNdata]
vax_CPV = [row[1] for row in CPVTUNdata]
vax_TUN = [row[2] for row in CPVTUNdata]
deaths_CPV = [row[3] for row in CPVTUNdata]
deaths_TUN = [row[4] for row in CPVTUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
CPVv = ax19.plot(dates, smooth(vax_CPV, 28), alpha=0.5, lw=2, label='Cape Verde vaccination rate')
TUNv = ax19.plot(dates, smooth(vax_TUN, 28), alpha=0.5, lw=2, label='Tunisia vaccination rate')
CPVem = ax20.plot(dates, smooth(deaths_CPV, 28), alpha=0.5, lw=2, label='Cape Verde excess mortality')
TUNem = ax20.plot(dates, smooth(deaths_TUN, 28), alpha=0.5, lw=2, label='Tunisia excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CapeVerdeTunisiaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Cape Verde and Tunisia')
save_figure(fig19, "CapeVerdeTunisiaEM.png")
plt.close('all')

HRVHUNdata = genfromtxt('CroatiaHungaryCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
deaths_HRVs = smooth(deaths_HRV, 7)
deaths_HUNs = smooth(deaths_HUN, 7)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVcd = ax20.plot(dates, smooth(deaths_HRVs, 28), alpha=0.5, lw=2, label='Croatia COVID-19 death rate')
HUNcd = ax20.plot(dates, smooth(deaths_HUNs, 28), alpha=0.5, lw=2, label='Hungary COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungaryCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Croatia and Hungary')
save_figure(fig19, "CroatiaHungaryCD.png")
plt.close('all')

HRVHUNdata = genfromtxt('CroatiaHungaryCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
deaths_HRVs = smooth(deaths_HRV, 7)
deaths_HUNs = smooth(deaths_HUN, 7)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVcd = ax20.plot(dates, smooth(deaths_HRVs, 28), alpha=0.5, lw=2, label='Croatia COVID-19 death rate')
HUNcd = ax20.plot(dates, smooth(deaths_HUNs, 28), alpha=0.5, lw=2, label='Hungary COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)]) # zoom in on this period
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungaryCDz.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Croatia and Hungary')
save_figure(fig19, "CroatiaHungaryCDz.png")
plt.close('all')



HRVHUNdata = genfromtxt('CroatiaHungaryCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
deaths_HRVs = smooth(deaths_HRV, 7)
deaths_HUNs = smooth(deaths_HUN, 7)
vax_diff = np.subtract(vax_HUN, vax_HRV) # np.subtract(x1, x2) = x1 - x2
deaths_diff = np.subtract(deaths_HUNs, deaths_HRVs)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNcd = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary COVID-19 death rate minus Croatia COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungaryCDdiffz.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Croatia and Hungary (differences)')
save_figure(fig19, "CroatiaHungaryCDdiffz.png")
plt.close('all')

# all ages
HRVHUNdata = genfromtxt('CroatiaHungaryEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel('Excess mortality P-score ($\\%$) (all ages)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungaryEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Croatia and Hungary')
save_figure(fig19, "CroatiaHungaryEM.png")
plt.close('all')




# all ages
HRVHUNdata = genfromtxt('CroatiaHungaryEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel('Excess mortality P-score ($\\%$) (all ages)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)]) # zoom in on this period
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungaryEMz.pdf")
ax19.set_title('Vaccinations and excess mortality for Croatia and Hungary')
save_figure(fig19, "CroatiaHungaryEMz.png")
plt.close('all')

# _0_14
HRVHUNdata = genfromtxt('CroatiaHungary_0_14EM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel(r'Excess mortality P-score ($\%$) (ages 0-14)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_0_14EM.pdf")
ax19.set_title(r'All-age vaccine doses and excess mortality at ages 0-14 for Croatia and Hungary')
save_figure(fig19, "CroatiaHungary_0_14EM.png")
plt.close('all')

# _0_14 z
HRVHUNdata = genfromtxt('CroatiaHungary_0_14EM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel(r'Excess mortality P-score ($\%$) (ages 0-14)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_0_14EMz.pdf")
ax19.set_title(r'All-age vaccine doses and excess mortality at ages 0-14 for Croatia and Hungary')
save_figure(fig19, "CroatiaHungary_0_14EMz.png")
plt.close('all')


# _0_14 z diff
HRVHUNdata = genfromtxt('CroatiaHungary_0_14EM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
vax_diff = np.subtract(vax_HUN, vax_HRV) # np.subtract(x1, x2) = x1 - x2
deaths_diff = np.subtract(deaths_HUN, deaths_HRV)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNcd = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel(r'Excess mortality P-score ($\%$) (ages 0-14)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_0_14EMdiffz.pdf")
ax19.set_title(r'All-age vaccine doses and excess mortality at ages 0-14 for Croatia and Hungary (differences)')
save_figure(fig19, "CroatiaHungary_0_14EMdiffz.png")
plt.close('all')





# _15_64
HRVHUNdata = genfromtxt('CroatiaHungary_15_64EM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel(r'Excess mortality P-score ($\%$) (ages 15-64)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_15_64EM.pdf")
ax19.set_title(r'All-age vaccine doses and excess mortality at ages 15-64 for Croatia and Hungary')
save_figure(fig19, "CroatiaHungary_15_64EM.png")
plt.close('all')





# _15_64 z
HRVHUNdata = genfromtxt('CroatiaHungary_15_64EM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel(r'Excess mortality P-score ($\%$) (ages 15-64)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_15_64EMz.pdf")
ax19.set_title(r'All-age vaccine doses and excess mortality at ages 15-64 for Croatia and Hungary')
save_figure(fig19, "CroatiaHungary_15_64EMz.png")
plt.close('all')


# _15_64 z diff
HRVHUNdata = genfromtxt('CroatiaHungary_15_64EM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
vax_diff = np.subtract(vax_HUN, vax_HRV) # np.subtract(x1, x2) = x1 - x2
deaths_diff = np.subtract(deaths_HUN, deaths_HRV)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNcd = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel(r'Excess mortality P-score ($\%$) (ages 15-64)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_15_64EMdiffz.pdf")
ax19.set_title(r'All-age vaccine doses and excess mortality at ages 15-64 for Croatia and Hungary (differences)')
save_figure(fig19, "CroatiaHungary_15_64EMdiffz.png")
plt.close('all')




# _65_74
HRVHUNdata = genfromtxt('CroatiaHungary_65_74EM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel(r'Excess mortality P-score ($\%$) (ages 65-74)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_65_74EM.pdf")
ax19.set_title(r'All-age vaccine doses and excess mortality at ages 65-74 for Croatia and Hungary')
save_figure(fig19, "CroatiaHungary_65_74EM.png")
plt.close('all')

# _65_74 z
HRVHUNdata = genfromtxt('CroatiaHungary_65_74EM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel(r'Excess mortality P-score ($\%$) (ages 65-74)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_65_74EMz.pdf")
ax19.set_title(r'All-age vaccine doses and excess mortality at ages 65-74 for Croatia and Hungary')
save_figure(fig19, "CroatiaHungary_65_74EMz.png")
plt.close('all')




# _65_74 z diff
HRVHUNdata = genfromtxt('CroatiaHungary_65_74EM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
vax_diff = np.subtract(vax_HUN, vax_HRV) # np.subtract(x1, x2) = x1 - x2
deaths_diff = np.subtract(deaths_HUN, deaths_HRV)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNcd = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel(r'Excess mortality P-score ($\%$) (ages 65-74)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_65_74EMdiffz.pdf")
ax19.set_title(r'All-age vaccine doses and excess mortality at ages 65-74 for Croatia and Hungary (differences)')
save_figure(fig19, "CroatiaHungary_65_74EMdiffz.png")
plt.close('all')



# _75_84
HRVHUNdata = genfromtxt('CroatiaHungary_75_84EM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel(r'Excess mortality P-score ($\%$) (ages 75-84)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_75_84EM.pdf")
ax19.set_title(r'All-age vaccine doses and excess mortality at ages 75-84 for Croatia and Hungary')
save_figure(fig19, "CroatiaHungary_75_84EM.png")
plt.close('all')

# _75_84 z
HRVHUNdata = genfromtxt('CroatiaHungary_75_84EM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel(r'Excess mortality P-score ($\%$) (ages 75-84)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_75_84EMz.pdf")
ax19.set_title(r'All-age vaccine doses and excess mortality at ages 75-84 for Croatia and Hungary')
save_figure(fig19, "CroatiaHungary_75_84EMz.png")
plt.close('all')

# _75_84 z diff
HRVHUNdata = genfromtxt('CroatiaHungary_75_84EM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
vax_diff = np.subtract(vax_HUN, vax_HRV) # np.subtract(x1, x2) = x1 - x2
deaths_diff = np.subtract(deaths_HUN, deaths_HRV)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNcd = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel(r'Excess mortality P-score ($\%$) (ages 75-84)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_75_84EMdiffz.pdf")
ax19.set_title(r'All-age vaccine doses and excess mortality at ages 75-84 for Croatia and Hungary (differences)')
save_figure(fig19, "CroatiaHungary_75_84EMdiffz.png")
plt.close('all')

# 85p
HRVHUNdata = genfromtxt('CroatiaHungary_85pEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel('Excess mortality P-score ($\\%$) (ages 85+)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_85pEM.pdf")
ax19.set_title('All-age vaccine doses and excess mortality at ages 85+ for Croatia and Hungary')
save_figure(fig19, "CroatiaHungary_85pEM.png")
plt.close('all')

# 85p z
HRVHUNdata = genfromtxt('CroatiaHungary_85pEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel('Excess mortality P-score ($\\%$) (ages 85+)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_85pEMz.pdf")
ax19.set_title('All-age vaccine doses and excess mortality at ages 85+ for Croatia and Hungary')
save_figure(fig19, "CroatiaHungary_85pEMz.png")
plt.close('all')

# 85p z diff
HRVHUNdata = genfromtxt('CroatiaHungary_85pEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
vax_diff = np.subtract(vax_HUN, vax_HRV) # np.subtract(x1, x2) = x1 - x2
deaths_diff = np.subtract(deaths_HUN, deaths_HRV)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNcd = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Doses per 100 people per day (all ages)')
ax20.set_ylabel('Excess mortality P-score ($\\%$) (ages 85+)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungary_85pEMdiffz.pdf")
ax19.set_title('All-age vaccine doses and excess mortality at ages 85+ for Croatia and Hungary (differences)')
save_figure(fig19, "CroatiaHungary_85pEMdiffz.png")
plt.close('all')





HRVHUNdata = genfromtxt('CroatiaHungaryEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
vax_diff = np.subtract(vax_HUN, vax_HRV)
deaths_diff = np.subtract(deaths_HUN, deaths_HRV)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNem = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungaryEMdiff.pdf")
ax19.set_title('Vaccinations and excess mortality for Croatia and Hungary (differences)')
save_figure(fig19, "CroatiaHungaryEMdiff.png")
plt.close('all')

HRVHUNdata = genfromtxt('CroatiaHungaryEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HRVHUNdata]
vax_HRV = [row[1] for row in HRVHUNdata]
vax_HUN = [row[2] for row in HRVHUNdata]
deaths_HRV = [row[3] for row in HRVHUNdata]
deaths_HUN = [row[4] for row in HRVHUNdata]
vax_diff = np.subtract(vax_HUN, vax_HRV)
deaths_diff = np.subtract(deaths_HUN, deaths_HRV)
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNem = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CroatiaHungaryEMdiffz.pdf")
ax19.set_title('Vaccinations and excess mortality for Croatia and Hungary (differences)')
save_figure(fig19, "CroatiaHungaryEMdiffz.png")
plt.close('all')







CUBJAMdata = genfromtxt('CubaJamaicaCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in CUBJAMdata]
vax_CUB = [row[1] for row in CUBJAMdata]
vax_JAM = [row[2] for row in CUBJAMdata]
deaths_CUB = [row[3] for row in CUBJAMdata]
deaths_JAM = [row[4] for row in CUBJAMdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
CUBv = ax19.plot(dates, smooth(vax_CUB, 28), alpha=0.5, lw=2, label='Cuba vaccination rate')
JAMv = ax19.plot(dates, smooth(vax_JAM, 28), alpha=0.5, lw=2, label='Jamaica vaccination rate')
CUBcd = ax20.plot(dates, smooth(deaths_CUB, 28), alpha=0.5, lw=2, label='Cuba COVID-19 death rate')
JAMcd = ax20.plot(dates, smooth(deaths_JAM, 28), alpha=0.5, lw=2, label='Jamaica COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CubaJamaicaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Cuba and Jamaica')
save_figure(fig19, "CubaJamaicaCD.png")
plt.close('all')

CUBJAMdata = genfromtxt('CubaJamaicaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in CUBJAMdata]
vax_CUB = [row[1] for row in CUBJAMdata]
vax_JAM = [row[2] for row in CUBJAMdata]
deaths_CUB = [row[3] for row in CUBJAMdata]
deaths_JAM = [row[4] for row in CUBJAMdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
CUBv = ax19.plot(dates, smooth(vax_CUB, 28), alpha=0.5, lw=2, label='Cuba vaccination rate')
JAMv = ax19.plot(dates, smooth(vax_JAM, 28), alpha=0.5, lw=2, label='Jamaica vaccination rate')
CUBem = ax20.plot(dates, smooth(deaths_CUB, 28), alpha=0.5, lw=2, label='Cuba excess mortality')
JAMem = ax20.plot(dates, smooth(deaths_JAM, 28), alpha=0.5, lw=2, label='Jamaica excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CubaJamaicaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Cuba and Jamaica')
save_figure(fig19, "CubaJamaicaEM.png")
plt.close('all')

CUBVCTdata = genfromtxt('CubaSaintVincentandtheGrenadinesEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in CUBVCTdata]
vax_CUB = [row[1] for row in CUBVCTdata]
vax_VCT = [row[2] for row in CUBVCTdata]
deaths_CUB = [row[3] for row in CUBVCTdata]
deaths_VCT = [row[4] for row in CUBVCTdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
CUBv = ax19.plot(dates, smooth(vax_CUB, 28), alpha=0.5, lw=2, label='Cuba vaccination rate')
VCTv = ax19.plot(dates, smooth(vax_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines vaccination rate')
CUBem = ax20.plot(dates, smooth(deaths_CUB, 28), alpha=0.5, lw=2, label='Cuba excess mortality')
VCTem = ax20.plot(dates, smooth(deaths_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "CubaSaintVincentandtheGrenadinesEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Cuba and Saint Vincent and the Grenadines')
save_figure(fig19, "CubaSaintVincentandtheGrenadinesEM.png")
plt.close('all')

DNKFINdata = genfromtxt('DenmarkFinlandEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in DNKFINdata]
vax_DNK = [row[1] for row in DNKFINdata]
vax_FIN = [row[2] for row in DNKFINdata]
deaths_DNK = [row[3] for row in DNKFINdata]
deaths_FIN = [row[4] for row in DNKFINdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
DNKv = ax19.plot(dates, smooth(vax_DNK, 28), alpha=0.5, lw=2, label='Denmark vaccination rate')
FINv = ax19.plot(dates, smooth(vax_FIN, 28), alpha=0.5, lw=2, label='Finland vaccination rate')
DNKem = ax20.plot(dates, smooth(deaths_DNK, 28), alpha=0.5, lw=2, label='Denmark excess mortality')
FINem = ax20.plot(dates, smooth(deaths_FIN, 28), alpha=0.5, lw=2, label='Finland excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "DenmarkFinlandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Denmark and Finland')
save_figure(fig19, "DenmarkFinlandEM.png")
plt.close('all')

DNKDEUdata = genfromtxt('DenmarkGermanyEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in DNKDEUdata]
vax_DNK = [row[1] for row in DNKDEUdata]
vax_DEU = [row[2] for row in DNKDEUdata]
deaths_DNK = [row[3] for row in DNKDEUdata]
deaths_DEU = [row[4] for row in DNKDEUdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
DNKv = ax19.plot(dates, smooth(vax_DNK, 28), alpha=0.5, lw=2, label='Denmark vaccination rate')
DEUv = ax19.plot(dates, smooth(vax_DEU, 28), alpha=0.5, lw=2, label='Germany vaccination rate')
DNKem = ax20.plot(dates, smooth(deaths_DNK, 28), alpha=0.5, lw=2, label='Denmark excess mortality')
DEUem = ax20.plot(dates, smooth(deaths_DEU, 28), alpha=0.5, lw=2, label='Germany excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "DenmarkGermanyEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Denmark and Germany')
save_figure(fig19, "DenmarkGermanyEM.png")
plt.close('all')

DNKNORdata = genfromtxt('DenmarkNorwayCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in DNKNORdata]
vax_DNK = [row[1] for row in DNKNORdata]
vax_NOR = [row[2] for row in DNKNORdata]
deaths_DNK = [row[3] for row in DNKNORdata]
deaths_NOR = [row[4] for row in DNKNORdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
DNKv = ax19.plot(dates, smooth(vax_DNK, 28), alpha=0.5, lw=2, label='Denmark vaccination rate')
NORv = ax19.plot(dates, smooth(vax_NOR, 28), alpha=0.5, lw=2, label='Norway vaccination rate')
DNKcd = ax20.plot(dates, smooth(deaths_DNK, 28), alpha=0.5, lw=2, label='Denmark COVID-19 death rate')
NORcd = ax20.plot(dates, smooth(deaths_NOR, 28), alpha=0.5, lw=2, label='Norway COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "DenmarkNorwayCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Denmark and Norway')
save_figure(fig19, "DenmarkNorwayCD.png")
plt.close('all')

DNKNORdata = genfromtxt('DenmarkNorwayEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in DNKNORdata]
vax_DNK = [row[1] for row in DNKNORdata]
vax_NOR = [row[2] for row in DNKNORdata]
deaths_DNK = [row[3] for row in DNKNORdata]
deaths_NOR = [row[4] for row in DNKNORdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
DNKv = ax19.plot(dates, smooth(vax_DNK, 28), alpha=0.5, lw=2, label='Denmark vaccination rate')
NORv = ax19.plot(dates, smooth(vax_NOR, 28), alpha=0.5, lw=2, label='Norway vaccination rate')
DNKem = ax20.plot(dates, smooth(deaths_DNK, 28), alpha=0.5, lw=2, label='Denmark excess mortality')
NORem = ax20.plot(dates, smooth(deaths_NOR, 28), alpha=0.5, lw=2, label='Norway excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "DenmarkNorwayEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Denmark and Norway')
save_figure(fig19, "DenmarkNorwayEM.png")
plt.close('all')

DOMHTIdata = genfromtxt('DominicanRepublicHaitiCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in DOMHTIdata]
vax_DOM = [row[1] for row in DOMHTIdata]
vax_HTI = [row[2] for row in DOMHTIdata]
deaths_DOM = [row[3] for row in DOMHTIdata]
deaths_HTI = [row[4] for row in DOMHTIdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
DOMv = ax19.plot(dates, smooth(vax_DOM, 28), alpha=0.5, lw=2, label='Dominican Republic vaccination rate')
HTIv = ax19.plot(dates, smooth(vax_HTI, 28), alpha=0.5, lw=2, label='Haiti vaccination rate')
DOMcd = ax20.plot(dates, smooth(deaths_DOM, 28), alpha=0.5, lw=2, label='Dominican Republic COVID-19 death rate')
HTIcd = ax20.plot(dates, smooth(deaths_HTI, 28), alpha=0.5, lw=2, label='Haiti COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "DominicanRepublicHaitiCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Dominican Republic and Haiti')
save_figure(fig19, "DominicanRepublicHaitiCD.png")
plt.close('all')

FRONORdata = genfromtxt('FaroeIslandsNorwayCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in FRONORdata]
vax_FRO = [row[1] for row in FRONORdata]
vax_NOR = [row[2] for row in FRONORdata]
deaths_FRO = [row[3] for row in FRONORdata]
deaths_NOR = [row[4] for row in FRONORdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
FROv = ax19.plot(dates, smooth(vax_FRO, 28), alpha=0.5, lw=2, label='Faroe Islands vaccination rate')
NORv = ax19.plot(dates, smooth(vax_NOR, 28), alpha=0.5, lw=2, label='Norway vaccination rate')
FROcd = ax20.plot(dates, smooth(deaths_FRO, 28), alpha=0.5, lw=2, label='Faroe Islands COVID-19 death rate')
NORcd = ax20.plot(dates, smooth(deaths_NOR, 28), alpha=0.5, lw=2, label='Norway COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "FaroeIslandsNorwayCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Faroe Islands and Norway')
save_figure(fig19, "FaroeIslandsNorwayCD.png")
plt.close('all')

FRONORdata = genfromtxt('FaroeIslandsNorwayEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in FRONORdata]
vax_FRO = [row[1] for row in FRONORdata]
vax_NOR = [row[2] for row in FRONORdata]
deaths_FRO = [row[3] for row in FRONORdata]
deaths_NOR = [row[4] for row in FRONORdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
FROv = ax19.plot(dates, smooth(vax_FRO, 28), alpha=0.5, lw=2, label='Faroe Islands vaccination rate')
NORv = ax19.plot(dates, smooth(vax_NOR, 28), alpha=0.5, lw=2, label='Norway vaccination rate')
FROem = ax20.plot(dates, smooth(deaths_FRO, 28), alpha=0.5, lw=2, label='Faroe Islands excess mortality')
NORem = ax20.plot(dates, smooth(deaths_NOR, 28), alpha=0.5, lw=2, label='Norway excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "FaroeIslandsNorwayEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Faroe Islands and Norway')
save_figure(fig19, "FaroeIslandsNorwayEM.png")
plt.close('all')

PYFNZLdata = genfromtxt('FrenchPolynesiaNewZealandEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in PYFNZLdata]
vax_PYF = [row[1] for row in PYFNZLdata]
vax_NZL = [row[2] for row in PYFNZLdata]
deaths_PYF = [row[3] for row in PYFNZLdata]
deaths_NZL = [row[4] for row in PYFNZLdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
PYFv = ax19.plot(dates, smooth(vax_PYF, 28), alpha=0.5, lw=2, label='French Polynesia vaccination rate')
NZLv = ax19.plot(dates, smooth(vax_NZL, 28), alpha=0.5, lw=2, label='New Zealand vaccination rate')
PYFem = ax20.plot(dates, smooth(deaths_PYF, 28), alpha=0.5, lw=2, label='French Polynesia excess mortality')
NZLem = ax20.plot(dates, smooth(deaths_NZL, 28), alpha=0.5, lw=2, label='New Zealand excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "FrenchPolynesiaNewZealandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for French Polynesia and New Zealand')
save_figure(fig19, "FrenchPolynesiaNewZealandEM.png")
plt.close('all')

GIBINDdata = genfromtxt('GibraltarIndiaCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in GIBINDdata]
vax_GIB = [row[1] for row in GIBINDdata]
vax_IND = [row[2] for row in GIBINDdata]
deaths_GIB = [row[3] for row in GIBINDdata]
deaths_IND = [row[4] for row in GIBINDdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
GIBv = ax19.plot(dates, smooth(vax_GIB, 28), alpha=0.5, lw=2, label='Gibraltar vaccination rate')
INDv = ax19.plot(dates, smooth(vax_IND, 28), alpha=0.5, lw=2, label='India vaccination rate')
GIBcd = ax20.plot(dates, smooth(deaths_GIB, 28), alpha=0.5, lw=2, label='Gibraltar COVID-19 death rate')
INDcd = ax20.plot(dates, smooth(deaths_IND, 28), alpha=0.5, lw=2, label='India COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "GibraltarIndiaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Gibraltar and India')
save_figure(fig19, "GibraltarIndiaCD.png")
plt.close('all')

HKGMACdata = genfromtxt('HongKongMacaoEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HKGMACdata]
vax_HKG = [row[1] for row in HKGMACdata]
vax_MAC = [row[2] for row in HKGMACdata]
deaths_HKG = [row[3] for row in HKGMACdata]
deaths_MAC = [row[4] for row in HKGMACdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HKGv = ax19.plot(dates, smooth(vax_HKG, 28), alpha=0.5, lw=2, label='Hong Kong vaccination rate')
MACv = ax19.plot(dates, smooth(vax_MAC, 28), alpha=0.5, lw=2, label='Macao vaccination rate')
HKGem = ax20.plot(dates, smooth(deaths_HKG, 28), alpha=0.5, lw=2, label='Hong Kong excess mortality')
MACem = ax20.plot(dates, smooth(deaths_MAC, 28), alpha=0.5, lw=2, label='Macao excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "HongKongMacaoEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Hong Kong and Macao')
save_figure(fig19, "HongKongMacaoEM.png")
plt.close('all')

HKGPHLdata = genfromtxt('HongKongPhilippinesCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HKGPHLdata]
vax_HKG = [row[1] for row in HKGPHLdata]
vax_PHL = [row[2] for row in HKGPHLdata]
deaths_HKG = [row[3] for row in HKGPHLdata]
deaths_PHL = [row[4] for row in HKGPHLdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HKGv = ax19.plot(dates, smooth(vax_HKG, 28), alpha=0.5, lw=2, label='Hong Kong vaccination rate')
PHLv = ax19.plot(dates, smooth(vax_PHL, 28), alpha=0.5, lw=2, label='Philippines vaccination rate')
HKGcd = ax20.plot(dates, smooth(deaths_HKG, 28), alpha=0.5, lw=2, label='Hong Kong COVID-19 death rate')
PHLcd = ax20.plot(dates, smooth(deaths_PHL, 28), alpha=0.5, lw=2, label='Philippines COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "HongKongPhilippinesCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Hong Kong and Philippines')
save_figure(fig19, "HongKongPhilippinesCD.png")
plt.close('all')

HKGPHLdata = genfromtxt('HongKongPhilippinesEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HKGPHLdata]
vax_HKG = [row[1] for row in HKGPHLdata]
vax_PHL = [row[2] for row in HKGPHLdata]
deaths_HKG = [row[3] for row in HKGPHLdata]
deaths_PHL = [row[4] for row in HKGPHLdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HKGv = ax19.plot(dates, smooth(vax_HKG, 28), alpha=0.5, lw=2, label='Hong Kong vaccination rate')
PHLv = ax19.plot(dates, smooth(vax_PHL, 28), alpha=0.5, lw=2, label='Philippines vaccination rate')
HKGem = ax20.plot(dates, smooth(deaths_HKG, 28), alpha=0.5, lw=2, label='Hong Kong excess mortality')
PHLem = ax20.plot(dates, smooth(deaths_PHL, 28), alpha=0.5, lw=2, label='Philippines excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "HongKongPhilippinesEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Hong Kong and Philippines')
save_figure(fig19, "HongKongPhilippinesEM.png")
plt.close('all')

HKGKORdata = genfromtxt('HongKongSouthKoreaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HKGKORdata]
vax_HKG = [row[1] for row in HKGKORdata]
vax_KOR = [row[2] for row in HKGKORdata]
deaths_HKG = [row[3] for row in HKGKORdata]
deaths_KOR = [row[4] for row in HKGKORdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HKGv = ax19.plot(dates, smooth(vax_HKG, 28), alpha=0.5, lw=2, label='Hong Kong vaccination rate')
KORv = ax19.plot(dates, smooth(vax_KOR, 28), alpha=0.5, lw=2, label='South Korea vaccination rate')
HKGem = ax20.plot(dates, smooth(deaths_HKG, 28), alpha=0.5, lw=2, label='Hong Kong excess mortality')
KORem = ax20.plot(dates, smooth(deaths_KOR, 28), alpha=0.5, lw=2, label='South Korea excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "HongKongSouthKoreaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Hong Kong and South Korea')
save_figure(fig19, "HongKongSouthKoreaEM.png")
plt.close('all')

HUNROUdata = genfromtxt('HungaryRomaniaCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HUNROUdata]
vax_HUN = [row[1] for row in HUNROUdata]
vax_ROU = [row[2] for row in HUNROUdata]
deaths_HUN = [row[3] for row in HUNROUdata]
deaths_ROU = [row[4] for row in HUNROUdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
ROUv = ax19.plot(dates, smooth(vax_ROU, 28), alpha=0.5, lw=2, label='Romania vaccination rate')
HUNcd = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary COVID-19 death rate')
ROUcd = ax20.plot(dates, smooth(deaths_ROU, 28), alpha=0.5, lw=2, label='Romania COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "HungaryRomaniaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Hungary and Romania')
save_figure(fig19, "HungaryRomaniaCD.png")
plt.close('all')

HUNROUdata = genfromtxt('HungaryRomaniaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in HUNROUdata]
vax_HUN = [row[1] for row in HUNROUdata]
vax_ROU = [row[2] for row in HUNROUdata]
deaths_HUN = [row[3] for row in HUNROUdata]
deaths_ROU = [row[4] for row in HUNROUdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
ROUv = ax19.plot(dates, smooth(vax_ROU, 28), alpha=0.5, lw=2, label='Romania vaccination rate')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ROUem = ax20.plot(dates, smooth(deaths_ROU, 28), alpha=0.5, lw=2, label='Romania excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "HungaryRomaniaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Hungary and Romania')
save_figure(fig19, "HungaryRomaniaEM.png")
plt.close('all')

IDNPHLdata = genfromtxt('IndonesiaPhilippinesCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in IDNPHLdata]
vax_IDN = [row[1] for row in IDNPHLdata]
vax_PHL = [row[2] for row in IDNPHLdata]
deaths_IDN = [row[3] for row in IDNPHLdata]
deaths_PHL = [row[4] for row in IDNPHLdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
IDNv = ax19.plot(dates, smooth(vax_IDN, 28), alpha=0.5, lw=2, label='Indonesia vaccination rate')
PHLv = ax19.plot(dates, smooth(vax_PHL, 28), alpha=0.5, lw=2, label='Philippines vaccination rate')
IDNcd = ax20.plot(dates, smooth(deaths_IDN, 28), alpha=0.5, lw=2, label='Indonesia COVID-19 death rate')
PHLcd = ax20.plot(dates, smooth(deaths_PHL, 28), alpha=0.5, lw=2, label='Philippines COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "IndonesiaPhilippinesCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Indonesia and Philippines')
save_figure(fig19, "IndonesiaPhilippinesCD.png")
plt.close('all')

IRQOMNdata = genfromtxt('IraqOmanCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in IRQOMNdata]
vax_IRQ = [row[1] for row in IRQOMNdata]
vax_OMN = [row[2] for row in IRQOMNdata]
deaths_IRQ = [row[3] for row in IRQOMNdata]
deaths_OMN = [row[4] for row in IRQOMNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
IRQv = ax19.plot(dates, smooth(vax_IRQ, 28), alpha=0.5, lw=2, label='Iraq vaccination rate')
OMNv = ax19.plot(dates, smooth(vax_OMN, 28), alpha=0.5, lw=2, label='Oman vaccination rate')
IRQcd = ax20.plot(dates, smooth(deaths_IRQ, 28), alpha=0.5, lw=2, label='Iraq COVID-19 death rate')
OMNcd = ax20.plot(dates, smooth(deaths_OMN, 28), alpha=0.5, lw=2, label='Oman COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "IraqOmanCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Iraq and Oman')
save_figure(fig19, "IraqOmanCD.png")
plt.close('all')

JAMVCTdata = genfromtxt('JamaicaSaintVincentandtheGrenadinesCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in JAMVCTdata]
vax_JAM = [row[1] for row in JAMVCTdata]
vax_VCT = [row[2] for row in JAMVCTdata]
deaths_JAM = [row[3] for row in JAMVCTdata]
deaths_VCT = [row[4] for row in JAMVCTdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
JAMv = ax19.plot(dates, smooth(vax_JAM, 28), alpha=0.5, lw=2, label='Jamaica vaccination rate')
VCTv = ax19.plot(dates, smooth(vax_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines vaccination rate')
JAMcd = ax20.plot(dates, smooth(deaths_JAM, 28), alpha=0.5, lw=2, label='Jamaica COVID-19 death rate')
VCTcd = ax20.plot(dates, smooth(deaths_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "JamaicaSaintVincentandtheGrenadinesCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths\nfor Jamaica and Saint Vincent and the Grenadines')
save_figure(fig19, "JamaicaSaintVincentandtheGrenadinesCD.png")
plt.close('all')

JAMVCTdata = genfromtxt('JamaicaSaintVincentandtheGrenadinesEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in JAMVCTdata]
vax_JAM = [row[1] for row in JAMVCTdata]
vax_VCT = [row[2] for row in JAMVCTdata]
deaths_JAM = [row[3] for row in JAMVCTdata]
deaths_VCT = [row[4] for row in JAMVCTdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
JAMv = ax19.plot(dates, smooth(vax_JAM, 28), alpha=0.5, lw=2, label='Jamaica vaccination rate')
VCTv = ax19.plot(dates, smooth(vax_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines vaccination rate')
JAMem = ax20.plot(dates, smooth(deaths_JAM, 28), alpha=0.5, lw=2, label='Jamaica excess mortality')
VCTem = ax20.plot(dates, smooth(deaths_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "JamaicaSaintVincentandtheGrenadinesEM.pdf")
ax19.set_title('Vaccinations and excess mortality\nfor Jamaica and Saint Vincent and the Grenadines')
save_figure(fig19, "JamaicaSaintVincentandtheGrenadinesEM.png")
plt.close('all')

JPNMYSdata = genfromtxt('JapanMalaysiaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in JPNMYSdata]
vax_JPN = [row[1] for row in JPNMYSdata]
vax_MYS = [row[2] for row in JPNMYSdata]
deaths_JPN = [row[3] for row in JPNMYSdata]
deaths_MYS = [row[4] for row in JPNMYSdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
JPNv = ax19.plot(dates, smooth(vax_JPN, 28), alpha=0.5, lw=2, label='Japan vaccination rate')
MYSv = ax19.plot(dates, smooth(vax_MYS, 28), alpha=0.5, lw=2, label='Malaysia vaccination rate')
JPNem = ax20.plot(dates, smooth(deaths_JPN, 28), alpha=0.5, lw=2, label='Japan excess mortality')
MYSem = ax20.plot(dates, smooth(deaths_MYS, 28), alpha=0.5, lw=2, label='Malaysia excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "JapanMalaysiaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Japan and Malaysia')
save_figure(fig19, "JapanMalaysiaEM.png")
plt.close('all')

JPNKORdata = genfromtxt('JapanSouthKoreaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in JPNKORdata]
vax_JPN = [row[1] for row in JPNKORdata]
vax_KOR = [row[2] for row in JPNKORdata]
deaths_JPN = [row[3] for row in JPNKORdata]
deaths_KOR = [row[4] for row in JPNKORdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
JPNv = ax19.plot(dates, smooth(vax_JPN, 28), alpha=0.5, lw=2, label='Japan vaccination rate')
KORv = ax19.plot(dates, smooth(vax_KOR, 28), alpha=0.5, lw=2, label='South Korea vaccination rate')
JPNem = ax20.plot(dates, smooth(deaths_JPN, 28), alpha=0.5, lw=2, label='Japan excess mortality')
KORem = ax20.plot(dates, smooth(deaths_KOR, 28), alpha=0.5, lw=2, label='South Korea excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "JapanSouthKoreaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Japan and South Korea')
save_figure(fig19, "JapanSouthKoreaEM.png")
plt.close('all')

KGZUZBdata = genfromtxt('KyrgyzstanUzbekistanCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in KGZUZBdata]
vax_KGZ = [row[1] for row in KGZUZBdata]
vax_UZB = [row[2] for row in KGZUZBdata]
deaths_KGZ = [row[3] for row in KGZUZBdata]
deaths_UZB = [row[4] for row in KGZUZBdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
KGZv = ax19.plot(dates, smooth(vax_KGZ, 28), alpha=0.5, lw=2, label='Kyrgyzstan vaccination rate')
UZBv = ax19.plot(dates, smooth(vax_UZB, 28), alpha=0.5, lw=2, label='Uzbekistan vaccination rate')
KGZcd = ax20.plot(dates, smooth(deaths_KGZ, 28), alpha=0.5, lw=2, label='Kyrgyzstan COVID-19 death rate')
UZBcd = ax20.plot(dates, smooth(deaths_UZB, 28), alpha=0.5, lw=2, label='Uzbekistan COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "KyrgyzstanUzbekistanCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Kyrgyzstan and Uzbekistan')
save_figure(fig19, "KyrgyzstanUzbekistanCD.png")
plt.close('all')

KGZUZBdata = genfromtxt('KyrgyzstanUzbekistanEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in KGZUZBdata]
vax_KGZ = [row[1] for row in KGZUZBdata]
vax_UZB = [row[2] for row in KGZUZBdata]
deaths_KGZ = [row[3] for row in KGZUZBdata]
deaths_UZB = [row[4] for row in KGZUZBdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
KGZv = ax19.plot(dates, smooth(vax_KGZ, 28), alpha=0.5, lw=2, label='Kyrgyzstan vaccination rate')
UZBv = ax19.plot(dates, smooth(vax_UZB, 28), alpha=0.5, lw=2, label='Uzbekistan vaccination rate')
KGZem = ax20.plot(dates, smooth(deaths_KGZ, 28), alpha=0.5, lw=2, label='Kyrgyzstan excess mortality')
UZBem = ax20.plot(dates, smooth(deaths_UZB, 28), alpha=0.5, lw=2, label='Uzbekistan excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "KyrgyzstanUzbekistanEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Kyrgyzstan and Uzbekistan')
save_figure(fig19, "KyrgyzstanUzbekistanEM.png")
plt.close('all')

LVAUKRdata = genfromtxt('LatviaUkraineCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in LVAUKRdata]
vax_LVA = [row[1] for row in LVAUKRdata]
vax_UKR = [row[2] for row in LVAUKRdata]
deaths_LVA = [row[3] for row in LVAUKRdata]
deaths_UKR = [row[4] for row in LVAUKRdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
LVAv = ax19.plot(dates, smooth(vax_LVA, 28), alpha=0.5, lw=2, label='Latvia vaccination rate')
UKRv = ax19.plot(dates, smooth(vax_UKR, 28), alpha=0.5, lw=2, label='Ukraine vaccination rate')
LVAcd = ax20.plot(dates, smooth(deaths_LVA, 28), alpha=0.5, lw=2, label='Latvia COVID-19 death rate')
UKRcd = ax20.plot(dates, smooth(deaths_UKR, 28), alpha=0.5, lw=2, label='Ukraine COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "LatviaUkraineCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Latvia and Ukraine')
save_figure(fig19, "LatviaUkraineCD.png")
plt.close('all')

LVAUKRdata = genfromtxt('LatviaUkraineEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in LVAUKRdata]
vax_LVA = [row[1] for row in LVAUKRdata]
vax_UKR = [row[2] for row in LVAUKRdata]
deaths_LVA = [row[3] for row in LVAUKRdata]
deaths_UKR = [row[4] for row in LVAUKRdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
LVAv = ax19.plot(dates, smooth(vax_LVA, 28), alpha=0.5, lw=2, label='Latvia vaccination rate')
UKRv = ax19.plot(dates, smooth(vax_UKR, 28), alpha=0.5, lw=2, label='Ukraine vaccination rate')
LVAem = ax20.plot(dates, smooth(deaths_LVA, 28), alpha=0.5, lw=2, label='Latvia excess mortality')
UKRem = ax20.plot(dates, smooth(deaths_UKR, 28), alpha=0.5, lw=2, label='Ukraine excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "LatviaUkraineEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Latvia and Ukraine')
save_figure(fig19, "LatviaUkraineEM.png")
plt.close('all')

LBNPSEdata = genfromtxt('LebanonPalestineEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in LBNPSEdata]
vax_LBN = [row[1] for row in LBNPSEdata]
vax_PSE = [row[2] for row in LBNPSEdata]
deaths_LBN = [row[3] for row in LBNPSEdata]
deaths_PSE = [row[4] for row in LBNPSEdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
LBNv = ax19.plot(dates, smooth(vax_LBN, 28), alpha=0.5, lw=2, label='Lebanon vaccination rate')
PSEv = ax19.plot(dates, smooth(vax_PSE, 28), alpha=0.5, lw=2, label='Palestine vaccination rate')
LBNem = ax20.plot(dates, smooth(deaths_LBN, 28), alpha=0.5, lw=2, label='Lebanon excess mortality')
PSEem = ax20.plot(dates, smooth(deaths_PSE, 28), alpha=0.5, lw=2, label='Palestine excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "LebanonPalestineEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Lebanon and Palestine')
save_figure(fig19, "LebanonPalestineEM.png")
plt.close('all')

MYSMNGdata = genfromtxt('MalaysiaMongoliaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in MYSMNGdata]
vax_MYS = [row[1] for row in MYSMNGdata]
vax_MNG = [row[2] for row in MYSMNGdata]
deaths_MYS = [row[3] for row in MYSMNGdata]
deaths_MNG = [row[4] for row in MYSMNGdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
MYSv = ax19.plot(dates, smooth(vax_MYS, 28), alpha=0.5, lw=2, label='Malaysia vaccination rate')
MNGv = ax19.plot(dates, smooth(vax_MNG, 28), alpha=0.5, lw=2, label='Mongolia vaccination rate')
MYSem = ax20.plot(dates, smooth(deaths_MYS, 28), alpha=0.5, lw=2, label='Malaysia excess mortality')
MNGem = ax20.plot(dates, smooth(deaths_MNG, 28), alpha=0.5, lw=2, label='Mongolia excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "MalaysiaMongoliaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Malaysia and Mongolia')
save_figure(fig19, "MalaysiaMongoliaEM.png")
plt.close('all')

MYSTWNdata = genfromtxt('MalaysiaTaiwanEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in MYSTWNdata]
vax_MYS = [row[1] for row in MYSTWNdata]
vax_TWN = [row[2] for row in MYSTWNdata]
deaths_MYS = [row[3] for row in MYSTWNdata]
deaths_TWN = [row[4] for row in MYSTWNdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
MYSv = ax19.plot(dates, smooth(vax_MYS, 28), alpha=0.5, lw=2, label='Malaysia vaccination rate')
TWNv = ax19.plot(dates, smooth(vax_TWN, 28), alpha=0.5, lw=2, label='Taiwan vaccination rate')
MYSem = ax20.plot(dates, smooth(deaths_MYS, 28), alpha=0.5, lw=2, label='Malaysia excess mortality')
TWNem = ax20.plot(dates, smooth(deaths_TWN, 28), alpha=0.5, lw=2, label='Taiwan excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "MalaysiaTaiwanEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Malaysia and Taiwan')
save_figure(fig19, "MalaysiaTaiwanEM.png")
plt.close('all')

MYSTHAdata = genfromtxt('MalaysiaThailandCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in MYSTHAdata]
vax_MYS = [row[1] for row in MYSTHAdata]
vax_THA = [row[2] for row in MYSTHAdata]
deaths_MYS = [row[3] for row in MYSTHAdata]
deaths_THA = [row[4] for row in MYSTHAdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
MYSv = ax19.plot(dates, smooth(vax_MYS, 28), alpha=0.5, lw=2, label='Malaysia vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
MYScd = ax20.plot(dates, smooth(deaths_MYS, 28), alpha=0.5, lw=2, label='Malaysia COVID-19 death rate')
THAcd = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "MalaysiaThailandCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Malaysia and Thailand')
save_figure(fig19, "MalaysiaThailandCD.png")
plt.close('all')

MYSTHAdata = genfromtxt('MalaysiaThailandEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in MYSTHAdata]
vax_MYS = [row[1] for row in MYSTHAdata]
vax_THA = [row[2] for row in MYSTHAdata]
deaths_MYS = [row[3] for row in MYSTHAdata]
deaths_THA = [row[4] for row in MYSTHAdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
MYSv = ax19.plot(dates, smooth(vax_MYS, 28), alpha=0.5, lw=2, label='Malaysia vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
MYSem = ax20.plot(dates, smooth(deaths_MYS, 28), alpha=0.5, lw=2, label='Malaysia excess mortality')
THAem = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "MalaysiaThailandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Malaysia and Thailand')
save_figure(fig19, "MalaysiaThailandEM.png")
plt.close('all')

MUSSYCdata = genfromtxt('MauritiusSeychellesCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in MUSSYCdata]
vax_MUS = [row[1] for row in MUSSYCdata]
vax_SYC = [row[2] for row in MUSSYCdata]
deaths_MUS = [row[3] for row in MUSSYCdata]
deaths_SYC = [row[4] for row in MUSSYCdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
MUSv = ax19.plot(dates, smooth(vax_MUS, 28), alpha=0.5, lw=2, label='Mauritius vaccination rate')
SYCv = ax19.plot(dates, smooth(vax_SYC, 28), alpha=0.5, lw=2, label='Seychelles vaccination rate')
MUScd = ax20.plot(dates, smooth(deaths_MUS, 28), alpha=0.5, lw=2, label='Mauritius COVID-19 death rate')
SYCcd = ax20.plot(dates, smooth(deaths_SYC, 28), alpha=0.5, lw=2, label='Seychelles COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "MauritiusSeychellesCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Mauritius and Seychelles')
save_figure(fig19, "MauritiusSeychellesCD.png")
plt.close('all')

MUSSYCdata = genfromtxt('MauritiusSeychellesEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in MUSSYCdata]
vax_MUS = [row[1] for row in MUSSYCdata]
vax_SYC = [row[2] for row in MUSSYCdata]
deaths_MUS = [row[3] for row in MUSSYCdata]
deaths_SYC = [row[4] for row in MUSSYCdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
MUSv = ax19.plot(dates, smooth(vax_MUS, 28), alpha=0.5, lw=2, label='Mauritius vaccination rate')
SYCv = ax19.plot(dates, smooth(vax_SYC, 28), alpha=0.5, lw=2, label='Seychelles vaccination rate')
MUSem = ax20.plot(dates, smooth(deaths_MUS, 28), alpha=0.5, lw=2, label='Mauritius excess mortality')
SYCem = ax20.plot(dates, smooth(deaths_SYC, 28), alpha=0.5, lw=2, label='Seychelles excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "MauritiusSeychellesEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Mauritius and Seychelles')
save_figure(fig19, "MauritiusSeychellesEM.png")
plt.close('all')

MNGTHAdata = genfromtxt('MongoliaThailandCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in MNGTHAdata]
vax_MNG = [row[1] for row in MNGTHAdata]
vax_THA = [row[2] for row in MNGTHAdata]
deaths_MNG = [row[3] for row in MNGTHAdata]
deaths_THA = [row[4] for row in MNGTHAdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
MNGv = ax19.plot(dates, smooth(vax_MNG, 28), alpha=0.5, lw=2, label='Mongolia vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
MNGcd = ax20.plot(dates, smooth(deaths_MNG, 28), alpha=0.5, lw=2, label='Mongolia COVID-19 death rate')
THAcd = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "MongoliaThailandCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Mongolia and Thailand')
save_figure(fig19, "MongoliaThailandCD.png")
plt.close('all')

MNGTHAdata = genfromtxt('MongoliaThailandEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in MNGTHAdata]
vax_MNG = [row[1] for row in MNGTHAdata]
vax_THA = [row[2] for row in MNGTHAdata]
deaths_MNG = [row[3] for row in MNGTHAdata]
deaths_THA = [row[4] for row in MNGTHAdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
MNGv = ax19.plot(dates, smooth(vax_MNG, 28), alpha=0.5, lw=2, label='Mongolia vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
MNGem = ax20.plot(dates, smooth(deaths_MNG, 28), alpha=0.5, lw=2, label='Mongolia excess mortality')
THAem = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "MongoliaThailandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Mongolia and Thailand')
save_figure(fig19, "MongoliaThailandEM.png")
plt.close('all')

PNGSLBdata = genfromtxt('PapuaNewGuineaSolomonIslandsCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in PNGSLBdata]
vax_PNG = [row[1] for row in PNGSLBdata]
vax_SLB = [row[2] for row in PNGSLBdata]
deaths_PNG = [row[3] for row in PNGSLBdata]
deaths_SLB = [row[4] for row in PNGSLBdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
PNGv = ax19.plot(dates, smooth(vax_PNG, 28), alpha=0.5, lw=2, label='Papua New Guinea vaccination rate')
SLBv = ax19.plot(dates, smooth(vax_SLB, 28), alpha=0.5, lw=2, label='Solomon Islands vaccination rate')
PNGcd = ax20.plot(dates, smooth(deaths_PNG, 28), alpha=0.5, lw=2, label='Papua New Guinea COVID-19 death rate')
SLBcd = ax20.plot(dates, smooth(deaths_SLB, 28), alpha=0.5, lw=2, label='Solomon Islands COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "PapuaNewGuineaSolomonIslandsCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Papua New Guinea and Solomon Islands')
save_figure(fig19, "PapuaNewGuineaSolomonIslandsCD.png")
plt.close('all')

PHLKORdata = genfromtxt('PhilippinesSouthKoreaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in PHLKORdata]
vax_PHL = [row[1] for row in PHLKORdata]
vax_KOR = [row[2] for row in PHLKORdata]
deaths_PHL = [row[3] for row in PHLKORdata]
deaths_KOR = [row[4] for row in PHLKORdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
PHLv = ax19.plot(dates, smooth(vax_PHL, 28), alpha=0.5, lw=2, label='Philippines vaccination rate')
KORv = ax19.plot(dates, smooth(vax_KOR, 28), alpha=0.5, lw=2, label='South Korea vaccination rate')
PHLem = ax20.plot(dates, smooth(deaths_PHL, 28), alpha=0.5, lw=2, label='Philippines excess mortality')
KORem = ax20.plot(dates, smooth(deaths_KOR, 28), alpha=0.5, lw=2, label='South Korea excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "PhilippinesSouthKoreaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Philippines and South Korea')
save_figure(fig19, "PhilippinesSouthKoreaEM.png")
plt.close('all')

ROURUSdata = genfromtxt('RomaniaRussiaCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in ROURUSdata]
vax_ROU = [row[1] for row in ROURUSdata]
vax_RUS = [row[2] for row in ROURUSdata]
deaths_ROU = [row[3] for row in ROURUSdata]
deaths_RUS = [row[4] for row in ROURUSdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
ROUv = ax19.plot(dates, smooth(vax_ROU, 28), alpha=0.5, lw=2, label='Romania vaccination rate')
RUSv = ax19.plot(dates, smooth(vax_RUS, 28), alpha=0.5, lw=2, label='Russia vaccination rate')
ROUcd = ax20.plot(dates, smooth(deaths_ROU, 28), alpha=0.5, lw=2, label='Romania COVID-19 death rate')
RUScd = ax20.plot(dates, smooth(deaths_RUS, 28), alpha=0.5, lw=2, label='Russia COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "RomaniaRussiaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Romania and Russia')
save_figure(fig19, "RomaniaRussiaCD.png")
plt.close('all')

ROURUSdata = genfromtxt('RomaniaRussiaEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in ROURUSdata]
vax_ROU = [row[1] for row in ROURUSdata]
vax_RUS = [row[2] for row in ROURUSdata]
deaths_ROU = [row[3] for row in ROURUSdata]
deaths_RUS = [row[4] for row in ROURUSdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
ROUv = ax19.plot(dates, smooth(vax_ROU, 28), alpha=0.5, lw=2, label='Romania vaccination rate')
RUSv = ax19.plot(dates, smooth(vax_RUS, 28), alpha=0.5, lw=2, label='Russia vaccination rate')
ROUem = ax20.plot(dates, smooth(deaths_ROU, 28), alpha=0.5, lw=2, label='Romania excess mortality')
RUSem = ax20.plot(dates, smooth(deaths_RUS, 28), alpha=0.5, lw=2, label='Russia excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "RomaniaRussiaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Romania and Russia')
save_figure(fig19, "RomaniaRussiaEM.png")
plt.close('all')

RWAUGAdata = genfromtxt('RwandaUgandaCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in RWAUGAdata]
vax_RWA = [row[1] for row in RWAUGAdata]
vax_UGA = [row[2] for row in RWAUGAdata]
deaths_RWA = [row[3] for row in RWAUGAdata]
deaths_UGA = [row[4] for row in RWAUGAdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
RWAv = ax19.plot(dates, smooth(vax_RWA, 28), alpha=0.5, lw=2, label='Rwanda vaccination rate')
UGAv = ax19.plot(dates, smooth(vax_UGA, 28), alpha=0.5, lw=2, label='Uganda vaccination rate')
RWAcd = ax20.plot(dates, smooth(deaths_RWA, 28), alpha=0.5, lw=2, label='Rwanda COVID-19 death rate')
UGAcd = ax20.plot(dates, smooth(deaths_UGA, 28), alpha=0.5, lw=2, label='Uganda COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "RwandaUgandaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Rwanda and Uganda')
save_figure(fig19, "RwandaUgandaCD.png")
plt.close('all')

SYCTZAdata = genfromtxt('SeychellesTanzaniaCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in SYCTZAdata]
vax_SYC = [row[1] for row in SYCTZAdata]
vax_TZA = [row[2] for row in SYCTZAdata]
deaths_SYC = [row[3] for row in SYCTZAdata]
deaths_TZA = [row[4] for row in SYCTZAdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
SYCv = ax19.plot(dates, smooth(vax_SYC, 28), alpha=0.5, lw=2, label='Seychelles vaccination rate')
TZAv = ax19.plot(dates, smooth(vax_TZA, 28), alpha=0.5, lw=2, label='Tanzania vaccination rate')
SYCcd = ax20.plot(dates, smooth(deaths_SYC, 28), alpha=0.5, lw=2, label='Seychelles COVID-19 death rate')
TZAcd = ax20.plot(dates, smooth(deaths_TZA, 28), alpha=0.5, lw=2, label='Tanzania COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "SeychellesTanzaniaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Seychelles and Tanzania')
save_figure(fig19, "SeychellesTanzaniaCD.png")
plt.close('all')

SGPTHAdata = genfromtxt('SingaporeThailandCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in SGPTHAdata]
vax_SGP = [row[1] for row in SGPTHAdata]
vax_THA = [row[2] for row in SGPTHAdata]
deaths_SGP = [row[3] for row in SGPTHAdata]
deaths_THA = [row[4] for row in SGPTHAdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
SGPv = ax19.plot(dates, smooth(vax_SGP, 28), alpha=0.5, lw=2, label='Singapore vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
SGPcd = ax20.plot(dates, smooth(deaths_SGP, 28), alpha=0.5, lw=2, label='Singapore COVID-19 death rate')
THAcd = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "SingaporeThailandCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Singapore and Thailand')
save_figure(fig19, "SingaporeThailandCD.png")
plt.close('all')

SGPTHAdata = genfromtxt('SingaporeThailandEM.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in SGPTHAdata]
vax_SGP = [row[1] for row in SGPTHAdata]
vax_THA = [row[2] for row in SGPTHAdata]
deaths_SGP = [row[3] for row in SGPTHAdata]
deaths_THA = [row[4] for row in SGPTHAdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
SGPv = ax19.plot(dates, smooth(vax_SGP, 28), alpha=0.5, lw=2, label='Singapore vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
SGPem = ax20.plot(dates, smooth(deaths_SGP, 28), alpha=0.5, lw=2, label='Singapore excess mortality')
THAem = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand excess mortality')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Excess mortality P-score ($\\%$)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "SingaporeThailandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Singapore and Thailand')
save_figure(fig19, "SingaporeThailandEM.png")
plt.close('all')

THAVNMdata = genfromtxt('ThailandVietnamCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in THAVNMdata]
vax_THA = [row[1] for row in THAVNMdata]
vax_VNM = [row[2] for row in THAVNMdata]
deaths_THA = [row[3] for row in THAVNMdata]
deaths_VNM = [row[4] for row in THAVNMdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
VNMv = ax19.plot(dates, smooth(vax_VNM, 28), alpha=0.5, lw=2, label='Vietnam vaccination rate')
THAcd = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand COVID-19 death rate')
VNMcd = ax20.plot(dates, smooth(deaths_VNM, 28), alpha=0.5, lw=2, label='Vietnam COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "ThailandVietnamCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Thailand and Vietnam')
save_figure(fig19, "ThailandVietnamCD.png")
plt.close('all')

ZMBZWEdata = genfromtxt('ZambiaZimbabweCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in ZMBZWEdata]
vax_ZMB = [row[1] for row in ZMBZWEdata]
vax_ZWE = [row[2] for row in ZMBZWEdata]
deaths_ZMB = [row[3] for row in ZMBZWEdata]
deaths_ZWE = [row[4] for row in ZMBZWEdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
sync_colour_cycle(ax20, ax19)
ZMBv = ax19.plot(dates, smooth(vax_ZMB, 28), alpha=0.5, lw=2, label='Zambia vaccination rate')
ZWEv = ax19.plot(dates, smooth(vax_ZWE, 28), alpha=0.5, lw=2, label='Zimbabwe vaccination rate')
ZMBcd = ax20.plot(dates, smooth(deaths_ZMB, 28), alpha=0.5, lw=2, label='Zambia COVID-19 death rate')
ZWEcd = ax20.plot(dates, smooth(deaths_ZWE, 28), alpha=0.5, lw=2, label='Zimbabwe COVID-19 death rate')
ax19.set_ylabel('Doses per 100 people per day')
ax20.set_ylabel('Deaths per million people per day')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
save_figure(fig19, "ZambiaZimbabweCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Zambia and Zimbabwe')
save_figure(fig19, "ZambiaZimbabweCD.png")
plt.close('all')




###############
# Output Data #
###############

f.write('IFR: ')
f.write(str(IFR))
f.write('\n')
f.write('gamma: ')
f.write(str(gamma))
f.write('\n')
f.write('\n')

f.write('UK_population: ')
f.write(str(pop_UK))
f.write('\n')
f.write('UK_datesD: ')

for t, y in enumerate(datesD_UK):
    datesD_UK[t] = datesD_UK[t].astype(dt.datetime)
    f.write(str(datesD_UK[t].strftime('%Y-%m-%d')))
    f.write(' ')
f.write('\n')
f.write('UK_datesI: ')
for t, y in enumerate(datesI_UK):
    datesI_UK[t] = datesI_UK[t].astype(dt.datetime)
    f.write(str(datesI_UK[t].strftime('%Y-%m-%d')))
    f.write(' ')
f.write('\n')
f.write('UK_deaths: ')
for t, y in enumerate(deaths_UK):
    f.write(str(deaths_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_deaths_smoothed: ')
for t, y in enumerate(deaths_UK_smoothed):
    f.write(str(deaths_UK_smoothed[t]))
    f.write(' ')
f.write('\n')
f.write('UK_deathrate: ')
for t, y in enumerate(deathrate_UK):
    f.write(str(deathrate_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_deathrate1: ')
for t, y in enumerate(deathrate1_UK):
    f.write(str(deathrate1_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_infections: ')
for t, y in enumerate(infections_UK):
    f.write(str(infections_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_S: ')
for t, y in enumerate(S_UK):
    f.write(str(S_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_I: ')
for t, y in enumerate(I_UK):
    f.write(str(I_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_R: ')
for t, y in enumerate(R_UK):
    f.write(str(R_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_S1: ')
for t, y in enumerate(S1_UK):
    f.write(str(S1_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_S_I_R: ')
for t, y in enumerate(S_UK):
    f.write(str(S_UK[t]+I_UK[t]+R_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_beta: ')
for t, y in enumerate(beta_UK):
    f.write(str(beta_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_dbetadt: ')
for t, y in enumerate(dbetadt_UK):
    f.write(str(dbetadt_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_dS1dt: ')
for t, y in enumerate(dS1dt_UK):
    f.write(str(dS1dt_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_dbetadS1: ')
for t, y in enumerate(dbetadS1_UK):
    f.write(str(dbetadS1_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_dbetadS1_smoothed: ')
for t, y in enumerate(dbetadS1_UK):
    f.write(str(smooth(dbetadS1_UK, 12)[t]))
    f.write(' ')
f.write('\n')
f.write('UK Gompertz parameters (A, r, t0): ')
f.write(str(A_UK)+' '+str(r_UK)+' '+str(t0_UK))
f.write('\n')
f.write('UK before lockdown Gompertz parameters (A, r, t0): ')
f.write(str(A_UKbefore)+' '+str(r_UKbefore)+' '+str(t0_UKbefore))
f.write('\n')
f.write('UK during lockdown Gompertz growth-rate coefficient (r): ')
f.write(str(r_UKlockdown))
f.write('\n')
f.write('\n')

f.write('Sweden_population: ')
f.write(str(pop_SE))
f.write('\n')




f.write('Sweden_datesD: ')
for t, y in enumerate(datesD_SE):
    datesD_SE[t] = datesD_SE[t].astype(dt.datetime)
    f.write(str(datesD_SE[t].strftime('%Y-%m-%d')))
    f.write(' ')
f.write('\n')
f.write('Sweden_datesI: ')
for t, y in enumerate(datesI_SE):
    datesI_SE[t] = datesI_SE[t].astype(dt.datetime)
    f.write(str(datesI_SE[t].strftime('%Y-%m-%d')))
    f.write(' ')
f.write('\n')
f.write('Sweden_deaths: ')
for t, y in enumerate(deaths_SE):
    f.write(str(deaths_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_deaths_smoothed: ')
for t, y in enumerate(deaths_SE_smoothed):
    f.write(str(deaths_SE_smoothed[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_deathrate: ')
for t, y in enumerate(deathrate_SE):
    f.write(str(deathrate_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_deathrate1: ')
for t, y in enumerate(deathrate1_SE):
    f.write(str(deathrate1_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_infections: ')
for t, y in enumerate(infections_SE):
    f.write(str(infections_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_S: ')
for t, y in enumerate(S_SE):
    f.write(str(S_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_I: ')
for t, y in enumerate(I_SE):
    f.write(str(I_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_R: ')
for t, y in enumerate(R_SE):
    f.write(str(R_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_S1: ')
for t, y in enumerate(S1_SE):
    f.write(str(S1_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_S_I_R: ')
for t, y in enumerate(S_SE):
    f.write(str(S_SE[t]+I_SE[t]+R_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_beta: ')
for t, y in enumerate(beta_SE):
    f.write(str(beta_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_beta_smoothed: ')
for t, y in enumerate(beta_SE_smoothed):
    f.write(str(beta_SE_smoothed[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_dbetadt: ')
for t, y in enumerate(dbetadt_SE):
    f.write(str(dbetadt_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_dS1dt: ')
for t, y in enumerate(dS1dt_SE):
    f.write(str(dS1dt_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_dbetadS1: ')
for t, y in enumerate(dbetadS1_SE):
    f.write(str(dbetadS1_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_dbetadS1_smoothed: ')
for t, y in enumerate(dbetadS1_SE):
    f.write(str(dbetadS1_SE_smoothed[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden Gompertz parameters (A, r, t0): ')
f.write(str(A_SE)+' '+str(r_SE)+' '+str(t0_SE))
f.write('\n')
f.write('Sweden before lockdown Gompertz parameters (A, r, t0): ')
f.write(str(A_SEbefore)+' '+str(r_SEbefore)+' '+str(t0_SEbefore))
f.write('\n')
f.write('Sweden during UK lockdown Gompertz growth-rate coefficient (r): ')
f.write(str(r_SElockdown))
f.write('\n')
f.write('\n')

f.write('England_population: ')
f.write(str(pop_EN))
f.write('\n')
f.write('England_datesD: ')
for t, y in enumerate(datesD_EN):
    f.write(str(datesD_EN[t]))
    f.write(' ')
f.write('\n')
f.write('England_datesI: ')
for t, y in enumerate(datesI_EN):
    f.write(str(datesI_EN[t]))
    f.write(' ')
f.write('\n')
f.write('England_deaths: ')
for t, y in enumerate(deaths_EN):
    f.write(str(deaths_EN[t]))
    f.write(' ')
f.write('\nEngland_infections: ')
for t, y in enumerate(infections_EN):
    f.write(str(infections_EN[t]))
    f.write(' ')
f.write('\n')
f.write('England_S: ')
for t, y in enumerate(S_EN):
    f.write(str(S_EN[t]))
    f.write(' ')
f.write('\n')
f.write('England_I: ')
for t, y in enumerate(I_EN):
    f.write(str(I_EN[t]))
    f.write(' ')
f.write('\n')
f.write('England_R: ')
for t, y in enumerate(R_EN):
    f.write(str(R_EN[t]))
    f.write(' ')
f.write('\n')
f.write('England_beta: ')
for t, y in enumerate(beta_EN):
    f.write(str(beta_EN[t]))
    f.write(' ')
f.write('\n')
f.write('England_beta_smoothed: ')
for t, y in enumerate(beta_EN_smoothed):
    f.write(str(beta_EN_smoothed[t]))
    f.write(' ')
f.write('\n')
f.write('\n')

f.write('Control: ')
for t, y in enumerate(control):
    f.write(str(control[t]))
    f.write(' ')
f.write('\n')
f.write('Control_smoothed: ')
for t, y in enumerate(control_smoothed):
    f.write(str(control_smoothed[t]))
    f.write(' ')
f.write('\n')
f.write('dbetadS1_UK - control: ')
for t, y in enumerate(epi_dbetadS1_UK_smoothed):
    f.write(str(epi_dbetadS1_UK_smoothed[t] - control_smoothed[t]))
    f.write(' ')
f.write('\n')
f.write('\n')

f.write('Statistic 1\n')
f.write('UK Gompertz function growth-rate coefficient during the lockdown minus the equivalent for Sweden (both per day).\n')
f.write('This is a descriptive growth-parameter contrast; its sign does not establish a causal lockdown effect. A moving-block bootstrap 95% confidence interval is reported at the end of this file.\n')
f.write(str(r_UKlockdown - r_SElockdown))
f.write('\n')
f.write('\n')

f.write('Statistic 2\n')
f.write('Mean difference between d(beta)/ds for the UK and the matched Sweden control over infection dates 28 February 2020 to 17 April 2020 (50 daily observations spanning 49 elapsed days), where s = 100*S/N is the percentage of the population susceptible, so the units are per day per percentage point of s. 12-day-smoothed series; the control is smoothed after being matched onto UK dates.\n')
f.write('This is a descriptive derivative contrast; its sign does not establish a causal lockdown effect. A moving-block bootstrap 95% confidence interval is reported at the end of this file.\n')
f.write(str(statistics.mean(epi_dbetadS1_UK_smoothed[25:74+1]) - statistics.mean(control_smoothed[25:74+1])))
f.write('\n')
f.write('Statistic 2 without the 12-day smoothing: ')
f.write(str(statistics.mean(epi_dbetadS1_UK[25:74+1]) - statistics.mean(control[25:74+1])))
f.write('\n')
f.write('\n')

f.write('Statistic 3\n')
f.write('Change in 12-day-smoothed beta in England between infection dates 29 June 2020 and 17 August 2020, minus the equivalent for Sweden (infection dates 8 July 2020 and 26 August 2020; the two series are aligned on each country\'s first death).\n')
f.write('This contrast has units of per day; it is descriptive and does not establish a causal mask-policy effect. A moving-block bootstrap 95% confidence interval is reported at the end of this file.\n')
f.write(str((beta_EN_smoothed[196] - beta_EN_smoothed[147]) - (beta_SE_smoothed[196] - beta_SE_smoothed[147])))
f.write('\n')
f.write('Statistic 3 without the 12-day smoothing: ')
f.write(str((beta_EN[196] - beta_EN[147]) - (beta_SE[196] - beta_SE[147])))


####################################
# Bootstrap Uncertainty Analysis   #
####################################
# Moving-block bootstrap on the daily deaths series, propagated through the full
# pipeline, giving percentile confidence intervals for Statistics 1-3.
# Method: for each country, take the 7-day moving average of daily deaths as the
# trend m_t; form Pearson-scaled residuals e_t = (y_t - m_t)/sqrt(max(m_t, 1));
# resample the residuals in weekday-aligned moving blocks of BOOT_BLOCK days;
# reconstruct synthetic deaths y*_t = round(max(0, m_t + sqrt(max(m_t, 1))*e*_t));
# and recompute the entire pipeline (7-day smoothing, inference of infections,
# SIR reconstruction, Gompertz fits with re-estimated peak alignment, beta,
# matching of the Sweden control, 12-day smoothing) and the three statistics on
# each replicate.  The intervals quantify sampling/reporting noise conditional
# on the modelling choices (the IFR, the 35% pre-existing-immunity assumption,
# the 27-day infection-to-death lag and the matching designs); they do not
# capture uncertainty in those choices themselves.
# The UK, Sweden and England series are resampled independently; no statistic
# combines the UK and England series, so their overlap is immaterial.
# Block-length sensitivity requires separate runs; it is not established by this script.

BOOT_B = args.bootstrap_replicates   # default 10,000; smaller values are for validation
BOOT_BLOCK = 14  # block length in days (two weekly cycles)
BOOT_SEED = 0    # for reproducibility

boot_rng = np.random.default_rng(BOOT_SEED)
boot_deaths_UK = np.asarray(deaths_UK, dtype=float)
boot_deaths_SE = np.asarray(deaths_SE, dtype=float)
boot_deaths_EN = np.asarray(deaths_EN, dtype=float)

def boot_gompertz(tt, A, r, t0):
    return A*r*np.exp(-np.exp(-r*(tt-t0))-r*(tt-t0))

def boot_fit3(y, guess, fallback):
    tt = np.arange(len(y))
    def sse(p):
        return np.sum((y - boot_gompertz(tt, p[0], p[1], p[2]))**2)
    res = optimize.minimize(sse, guess, method='Nelder-Mead')
    if not res.success:
        res = optimize.minimize(sse, fallback, method='Nelder-Mead')
    return res.x, res.success

def boot_fitr(y, i0, A, t0, guess_r):
    tt = np.arange(len(y)) + i0
    def sse(p):
        return np.sum((y - boot_gompertz(tt, A, p[0], t0))**2)
    res = optimize.minimize(sse, [guess_r], method='Nelder-Mead')
    return res.x[0], res.success

# Warm starts: the parameters fitted to the observed data (above); fallbacks: the original initial guesses.
boot_warm = {'UK': [A_UK, r_UK, t0_UK], 'SE': [A_SE, r_SE, t0_SE],
             'UKb': [A_UKbefore, r_UKbefore, t0_UKbefore], 'UKl': r_UKlockdown,
             'SEb': [A_SEbefore, r_SEbefore, t0_SEbefore], 'SEl': r_SElockdown}

def boot_pipeline(yUK, ySE, yEN):
    """Recompute Statistics 1-3 from (possibly synthetic) daily deaths series.
    Returns five descriptive contrasts, or None if a fit or contrast is invalid."""
    iUK = smooth(yUK, 7)/IFR
    iSE = smooth(ySE, 7)/IFR
    iEN = smooth(yEN, 7)/IFR

    def sir_beta(i, pop):
        n = len(i)
        S = 0.65*pop - np.cumsum(i)              # S[0] = 0.65N - c_0; S[t] = S[t-1] - c_t
        I = np.convolve(i, np.ones(9))[:n]       # fixed 9-day infectious period (1/gamma)
        beta = inferred_beta(i, S, I, pop)
        return S, beta

    S_UKb_, beta_UKb_ = sir_beta(iUK, pop_UK)
    S_SEb_, beta_SEb_ = sir_beta(iSE, pop_SE)
    S_ENb_, beta_ENb_ = sir_beta(iEN, pop_EN)

    # Statistic 1: Gompertz growth-rate difference, with peak alignment re-estimated
    pUK, ok1 = boot_fit3(iUK[:last_UK_epi_index+1], boot_warm['UK'], initial_guess_UK)
    pSE, ok2 = boot_fit3(iSE[:last_SE_epi_index+1], boot_warm['SE'], initial_guess_SE)
    if not (ok1 and ok2):
        return None
    maxUK = int(np.argmax(boot_gompertz(np.arange(last_UK_epi_index+1), pUK[0], pUK[1], pUK[2])))
    maxSE = int(np.argmax(boot_gompertz(np.arange(last_SE_epi_index+1), pSE[0], pSE[1], pSE[2])))
    off = maxSE - maxUK
    fSE = first_day_UK_lockdown_index + off
    lSE = last_day_UK_lockdown_index + off
    if fSE < 5 or lSE + 1 > len(iSE):
        return None
    pUKb, ok3 = boot_fit3(iUK[:first_day_UK_lockdown_index], boot_warm['UKb'], initial_guess_UKbefore)
    if not ok3:
        return None
    rUKl, ok4 = boot_fitr(iUK[first_day_UK_lockdown_index:last_day_UK_lockdown_index+1],
                          first_day_UK_lockdown_index, pUKb[0], pUKb[2], boot_warm['UKl'])
    pSEb, ok5 = boot_fit3(iSE[:fSE], boot_warm['SEb'], initial_guess_SEbefore)
    if not (ok4 and ok5):
        return None
    rSEl, ok6 = boot_fitr(iSE[fSE:lSE+1], fSE, pSEb[0], pSEb[2], boot_warm['SEl'])
    if not ok6:
        return None
    s1 = rUKl - rSEl
    s4 = (rUKl - pUKb[1]) - (rSEl - pSEb[1])  # Gompertz difference-in-differences

    # Statistic 2: d(beta)/ds with matched Sweden control, 12-day smoothing after matching
    def dbds(beta, S, pop):
        s = 100.0*S/pop
        db = np.empty(len(beta)); dsv = np.empty(len(beta))
        db[1:] = np.diff(beta); db[0] = beta[1]-beta[0]
        dsv[1:] = np.diff(s);   dsv[0] = s[1]-s[0]
        out = np.full(len(beta), np.nan)
        nz = dsv != 0
        out[nz] = db[nz]/dsv[nz]
        return out, s

    dbdsUK, sUK = dbds(beta_UKb_, S_UKb_, pop_UK)
    dbdsSE, sSE = dbds(beta_SEb_, S_SEb_, pop_SE)
    epi_sUK = sUK[:last_UK_epi_index+1]
    epi_sSE = sSE[:last_SE_epi_index+1]
    epi_dbdsUK = dbdsUK[:last_UK_epi_index+1]
    closest = np.argmin((epi_sUK[:, None] - epi_sSE[None, :])**2, axis=1)
    ctrl = dbdsSE[closest]
    s2 = np.mean(smooth(epi_dbdsUK, 12)[25:75]) - np.mean(smooth(ctrl, 12)[25:75])

    # Statistic 3: masks difference-in-differences on 12-day-smoothed beta
    bENs = smooth(beta_ENb_, 12)
    bSEs = smooth(beta_SEb_, 12)
    s3 = (bENs[196]-bENs[147]) - (bSEs[196]-bSEs[147])
    # C10: the same mask contrast with England and Sweden aligned on common calendar
    # dates (Swedish indices 138 and 187 fall on the English calendar dates 147 and 196).
    s5 = (bENs[196]-bENs[147]) - (bSEs[187]-bSEs[138])

    if not (np.isfinite(s1) and np.isfinite(s2) and np.isfinite(s3) and np.isfinite(s4) and np.isfinite(s5)):
        return None
    return s1, s2, s3, s4, s5

def boot_resample(y, rng, block):
    """Weekday-aligned moving-block bootstrap of Pearson-scaled residuals
    around the 7-day moving-average trend."""
    n = len(y)
    trend = smooth(y, 7)
    sd = np.sqrt(np.maximum(trend, 1.0))
    resid = (y - trend)/sd
    out = np.empty(n)
    pos = 0
    while pos < n:
        cands = np.arange(pos % 7, n - block + 1, 7)
        if len(cands) == 0:
            cands = np.arange(0, max(1, n - block + 1))
        start = rng.choice(cands)
        ln = min(block, n - pos)
        out[pos:pos+ln] = resid[start:start+ln]
        pos += ln
    return np.rint(np.maximum(0.0, trend + sd*out))

# Internal consistency check: the bootstrap's pipeline mirror must reproduce the main statistics.
boot_stat1_main = r_UKlockdown - r_SElockdown
boot_stat2_main = statistics.mean(epi_dbetadS1_UK_smoothed[25:74+1]) - statistics.mean(control_smoothed[25:74+1])
boot_stat3_main = (beta_EN_smoothed[196] - beta_EN_smoothed[147]) - (beta_SE_smoothed[196] - beta_SE_smoothed[147])
boot_stat4_main = (r_UKlockdown - r_UKbefore) - (r_SElockdown - r_SEbefore)  # Gompertz difference-in-differences
boot_stat5_main = (beta_EN_smoothed[196] - beta_EN_smoothed[147]) - (beta_SE_smoothed[187] - beta_SE_smoothed[138])  # mask contrast, common calendar dates (C10)
boot_point = boot_pipeline(boot_deaths_UK, boot_deaths_SE, boot_deaths_EN)
if boot_point is None:
    raise RuntimeError('The bootstrap pipeline could not reproduce the observed-data fit.')
boot_check = max(abs(boot_point[0]-boot_stat1_main), abs(boot_point[1]-boot_stat2_main), abs(boot_point[2]-boot_stat3_main), abs(boot_point[3]-boot_stat4_main), abs(boot_point[4]-boot_stat5_main))

if not np.isfinite(boot_check) or boot_check > 1e-5:
    raise RuntimeError('The bootstrap pipeline disagrees with the main statistics.')

print("Running moving-block bootstrap (B = " + str(BOOT_B) + ")...")
boot_results = []
boot_failures = 0
for boot_b in range(BOOT_B):
    boot_r = boot_pipeline(boot_resample(boot_deaths_UK, boot_rng, BOOT_BLOCK),
                           boot_resample(boot_deaths_SE, boot_rng, BOOT_BLOCK),
                           boot_resample(boot_deaths_EN, boot_rng, BOOT_BLOCK))
    if boot_r is None:
        boot_failures += 1
    else:
        boot_results.append(boot_r)
    if (boot_b+1) % 1000 == 0:
        print("  bootstrap replicate " + str(boot_b+1) + "/" + str(BOOT_B))
if len(boot_results) < 2:
    raise RuntimeError('Fewer than two valid bootstrap replicates; confidence intervals cannot be calculated.')
boot_arr = np.array(boot_results)

f.write('\n')
f.write('\n')
f.write('Bootstrap Uncertainty Analysis\n')
f.write('Moving-block bootstrap: B = ' + str(BOOT_B) + ' replicates, block length ' + str(BOOT_BLOCK) + ' days (weekday-aligned), seed ' + str(BOOT_SEED) + '.\n')
f.write('Trend: 7-day moving average of daily deaths; Pearson-scaled residuals resampled; full pipeline recomputed per replicate.\n')
f.write('95% percentile intervals, conditional on the modelling choices (IFR, 35% pre-existing immunity, 27-day infection-to-death lag, matching designs).\n')
f.write('Internal consistency check (pipeline mirror vs main code, max abs diff): ' + str(boot_check) + '\n')
f.write('Replicates discarded due to failed fits: ' + str(boot_failures) + '\n')
for boot_i, (boot_name, boot_pt) in enumerate([('Statistic 1', boot_stat1_main), ('Statistic 2', boot_stat2_main), ('Statistic 3', boot_stat3_main), ('Statistic 1 difference-in-differences', boot_stat4_main), ('Statistic 3 common-calendar sensitivity', boot_stat5_main)]):
    boot_lo, boot_hi = np.percentile(boot_arr[:, boot_i], [2.5, 97.5])
    boot_se = boot_arr[:, boot_i].std(ddof=1)
    boot_units = 'per day per percentage point' if boot_i == 1 else 'per day'
    boot_verdict = 'interval includes zero' if (boot_lo <= 0.0 <= boot_hi) else 'interval excludes zero'
    f.write(boot_name + ' (' + boot_units + '): ' + str(boot_pt) + ', bootstrap SE ' + str(boot_se) + ', 95% CI [' + str(boot_lo) + ', ' + str(boot_hi) + '] (' + boot_verdict + ')\n')

f.flush()
os.fsync(f.fileno())
f.close()
print("Finished!")