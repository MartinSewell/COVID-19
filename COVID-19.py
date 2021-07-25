# The Efficacy of Lockdowns and Face Masks Vis-à-Vis Mitigating COVID-19
# Martin Sewell
# martin.sewell@cantab.net
# 25 July 2021

import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.optimize as optimize
import statistics
from numpy import genfromtxt

# Abbreviations:
# UK United Kingdom
# SE Sweden
# EN England
# IFR infection fatality rate

# Constants
pop_UK = 67886011 # Worldometer (2020) https://www.worldometers.info/world-population/uk-population/
pop_SE = 10099265 # Worldometer (2020) https://www.worldometers.info/world-population/sweden-population/
pop_EN = 56550000 # ONS (2020) https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/bulletins/annualmidyearpopulationestimates/mid2020
IFR = 0.51/100    # Oke and Heneghan (2021) https://www.cebm.net/covid-19/global-covid-19-case-fatality-rates/
gamma = 1/9       # Van Beusekom (2020) https://www.cidrap.umn.edu/news-perspective/2020/11/covid-19-most-contagious-first-5-days-illness-study-finds

# Data sources:
# UK: https://coronavirus.data.gov.uk/details/deaths
# Sweden: https://fohm.maps.arcgis.com/sharing/rest/content/items/b5e7488e117749c19881cce45db13f7e/data Antal avlidna per dag
# England: https://coronavirus.data.gov.uk/details/deaths?areaType=nation&areaName=England

# Enable LaTeX maths symbols in figures
plt.rc('text', usetex = True)
plt.rc('font', **{'family' : "sans-serif"})
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{commath}']}
plt.rcParams.update(params)

# Function to parse dates in input files
def date_parser(d_bytes):
    s = d_bytes.decode('utf-8')
    return np.datetime64(dt.datetime.strptime(s, '%d/%m/%Y'))

# Function to smooth data
def smooth(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, mode='same')

# output file
f = open("output.txt","w")

# Extract dates and deaths from input files
# datesD refers to dates with respect to deaths
data_UK = genfromtxt('UKdeaths.csv', delimiter='\t', converters = {0: date_parser})
datesD_UK = [row[0] for row in data_UK]
deaths_UK = [row[1] for row in data_UK]
num_UK_dates = len(datesD_UK)

data_SE = genfromtxt('Swedendeaths.csv', delimiter='\t', converters = {0: date_parser})
datesD_SE = [row[0] for row in data_SE]
deaths_SE = [row[1] for row in data_SE]
num_SE_dates = len(datesD_SE)

data_EN = genfromtxt('Englanddeaths.csv', delimiter='\t', converters = {0: date_parser})
datesD_EN = [row[0] for row in data_EN]
deaths_EN = [row[1] for row in data_EN]
num_EN_dates = len(datesD_EN)

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
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7*8))
ax1.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig1.autofmt_xdate()
ax1.grid(b=True)
ax1.set_title('UK COVID-19 deaths')
legend = ax1.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax1.spines[spine].set_visible(False)
fig1.savefig("fig1.png")

# Sweden COVID-19 deaths
fig2 = plt.figure(facecolor='w')
ax2 = fig2.add_subplot(111, axisbelow=True)
ax2.plot(datesD_SE, deaths_SE, alpha=0.5, lw=2, label='Daily')
ax2.plot(datesD_SE, deaths_SE_smoothed, alpha=0.5, lw=2, label='7-day moving average')
ax2.set_ylabel('Daily deaths')
ax2.yaxis.set_tick_params(length=0)
ax2.xaxis.set_tick_params(length=0)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7*8))
ax2.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig2.autofmt_xdate()
ax2.grid(b=True)
ax2.set_title('Sweden COVID-19 deaths')
legend = ax2.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax2.spines[spine].set_visible(False)
fig2.savefig("fig2.png")

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

# For days since first COVID-19 death in figure 3, we need to delete the first number.
deathrate1_UK = deathrate_UK[:]
deathrate1_SE = deathrate_SE[:]
deathrate1_UK.pop(0)
deathrate1_SE.pop(0)

# COVID-19 death rate in the UK and Sweden
fig3 = plt.figure(facecolor='w')
ax3 = fig3.add_subplot(111, axisbelow=True)
ax3.plot(deathrate1_UK, alpha=0.5, lw=2, label='UK')
ax3.plot(deathrate1_SE, alpha=0.5, lw=2, label='Sweden')
ax3.set_xlabel('Days since first COVID-19 death')
ax3.set_ylabel('Deaths per 100,000 population per day')
ax3.yaxis.set_tick_params(length=0)
ax3.xaxis.set_tick_params(length=0)
ax3.grid(b=True)
legend = ax3.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax3.spines[spine].set_visible(False)
fig3.savefig("fig1.pdf") # first figure in paper
ax3.set_title('COVID-19 death rate in the UK and Sweden')
fig3.savefig("fig3.png")

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
        S_UK[t] = pop_UK - infections_UK[t]
    else:
        S_UK[t] = S_UK[t-1] - infections_UK[t]

for t, y in enumerate(S_SE):
    if t == 0:
        S_SE[t] = pop_SE - infections_SE[t]
    else:
        S_SE[t] = S_SE[t-1] - infections_SE[t]

for t, y in enumerate(S_EN):
    if t == 0:
        S_EN[t] = pop_EN - infections_EN[t]
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
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax4.xaxis.set_major_locator(mdates.DayLocator(interval=17*4))
ax4.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig4.autofmt_xdate()
ax4.grid(b=True)
ax4.set_title('UK COVID-19 SIR model')
legend = ax4.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax4.spines[spine].set_visible(False)
fig4.savefig("fig4.png")

# Sweden COVID-19 SIR model
fig5 = plt.figure(facecolor='w')
ax5 = fig5.add_subplot(111, axisbelow=True)
ax5.plot(datesI_SE, S_SE, alpha=0.5, lw=2, label='S')
ax5.plot(datesI_SE, I_SE, alpha=0.5, lw=2, label='I')
ax5.plot(datesI_SE, R_SE, alpha=0.5, lw=2, label='R')
ax5.set_ylabel('Number of people')
ax5.yaxis.set_tick_params(length=0)
ax5.xaxis.set_tick_params(length=0)
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax5.xaxis.set_major_locator(mdates.DayLocator(interval=7*8))
ax5.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig5.autofmt_xdate()
ax5.grid(b=True)
ax5.set_title('Sweden COVID-19 SIR model')
legend = ax5.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax5.spines[spine].set_visible(False)
fig5.savefig("fig5.png")


##############################
# Gompertz Function Analysis #
##############################

# Fit Gompertz function to UK data

# Only use epidemic (first wave) data, so truncate data
epi_infections_UK = infections_UK[0 : last_UK_epi_index+1] # does not include last value
epi_datesI_UK = datesI_UK[0 : last_UK_epi_index+1] # does not include last value

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
    raise ValueError(result.message)

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
ax6.plot(epi_datesI_UK, GompUK, alpha=0.5, lw=2, label='Gompertz')
ax6.set_ylabel('New infections per day')
ax6.yaxis.set_tick_params(length=0)
ax6.xaxis.set_tick_params(length=0)
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax6.xaxis.set_major_locator(mdates.DayLocator(interval=7*4))
ax6.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig6.autofmt_xdate()
ax6.grid(b=True)
ax6.set_title('UK infections and fitted first derivative of the Gompertz function')
legend = ax6.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax6.spines[spine].set_visible(False)
fig6.savefig("fig6.png")


# Fit Gompertz function to Sweden data

# Only use epidemic (first wave) data, so truncate data
epi_infections_SE = infections_SE[0 : last_SE_epi_index+1] # does not include last value
epi_datesI_SE = datesI_SE[0 : last_SE_epi_index+1] # does not include last value

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
ax7.plot(epi_datesI_SE, GompSE, alpha=0.5, lw=2, label='Gompertz')
ax7.set_ylabel('New infections per day')
ax7.yaxis.set_tick_params(length=0)
ax7.xaxis.set_tick_params(length=0)
ax7.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax7.xaxis.set_major_locator(mdates.DayLocator(interval=7*4))
ax7.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig7.autofmt_xdate()
ax7.grid(b=True)
legend = ax7.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax7.spines[spine].set_visible(False)
fig7.savefig("fig2.pdf") # second figure in paper
ax7.set_title('Sweden infections and fitted first derivative of the Gompertz function')
fig7.savefig("fig7.png")


# Gompertz function - UK before lockdown

# We only want epidemic (first wave) data, so truncate data
epi_infections_UKbefore = infections_UK[0 : (first_day_UK_lockdown_index-1)+1] # does not include last value
epi_datesI_UKbefore = datesI_UK[0 : (first_day_UK_lockdown_index-1)+1] # does not include last value

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
    raise ValueError(result.message)

GompUKbefore = [0.0] * ((first_day_UK_lockdown_index-1)+1)
for t, y in enumerate(GompUKbefore):
    GompUKbefore[t] = A_UKbefore*r_UKbefore*np.exp(-np.exp(-r_UKbefore*(t-t0_UKbefore))-r_UKbefore*(t-t0_UKbefore))

# UK infections before lockdown and fitted first derivative of the Gompertz function
fig8 = plt.figure(facecolor='w')
ax8 = fig8.add_subplot(111, axisbelow=True)
ax8.plot(epi_datesI_UKbefore, epi_infections_UKbefore, alpha=0.5, lw=2, label='Data')
ax8.plot(epi_datesI_UKbefore, GompUKbefore, alpha=0.5, lw=2, label='Gompertz')
ax8.set_ylabel('New infections per day')
ax8.yaxis.set_tick_params(length=0)
ax8.xaxis.set_tick_params(length=0)
ax8.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax8.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax8.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig8.autofmt_xdate()
ax8.grid(b=True)
ax8.set_title('UK infections before lockdown and fitted first derivative of the Gompertz function')
legend = ax8.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax8.spines[spine].set_visible(False)
fig8.savefig("fig8.png")


# Gompertz function - UK during lockdown

# only use epidemic data, so truncate data
epi_infections_UKlockdown = infections_UK[first_day_UK_lockdown_index : last_day_UK_lockdown_index+1] # does not include last value
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
ax9.plot(epi_datesI_UKlockdown, GompUKlockdown, alpha=0.5, lw=2, label='Fitted Gompertz function')
ax9.set_ylabel('New infections per day')
ax9.yaxis.set_tick_params(length=0)
ax9.xaxis.set_tick_params(length=0)
ax9.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax9.xaxis.set_major_locator(mdates.DayLocator(interval=7*2))
ax9.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig9.autofmt_xdate()
ax9.grid(b=True)
ax9.set_title('UK infections during lockdown and fitted first derivative of the Gompertz function')
legend = ax9.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax9.spines[spine].set_visible(False)
fig9.savefig("fig9.png")


# Gompertz function - Sweden before UK lockdown

# Sweden didn't have a lockdown, but we wish to identify the part of Sweden's epidemic curve that corresponds to the UK's epidemic curve during the UK lockdown by matching the peaks of the curves.
first_SE_lockdown_index = first_day_UK_lockdown_index + (maxGompSEindex - maxGompUKindex)
last_SE_lockdown_index = last_day_UK_lockdown_index + (maxGompSEindex - maxGompUKindex)

# Only use epidemic data, so truncate data
epi_infections_SEbefore = infections_SE[0 : (first_SE_lockdown_index-1)+1] # does not include last value
epi_datesI_SEbefore = datesI_SE[0 : (first_SE_lockdown_index-1)+1] # does not include last value

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
ax10.plot(epi_datesI_SEbefore, GompSEbefore, alpha=0.5, lw=2, label='Gompertz')
ax10.set_ylabel('New infections per day')
ax10.yaxis.set_tick_params(length=0)
ax10.xaxis.set_tick_params(length=0)
ax10.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax10.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax10.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig10.autofmt_xdate()
ax10.grid(b=True)
ax10.set_title('Sweden infections before UK lockdown and fitted first derivative of the Gompertz function')
legend = ax10.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax10.spines[spine].set_visible(False)
fig10.savefig("fig10.png")


# Gompertz function - Sweden during UK lockdown

# only use lockdown data, so truncate data
epi_infections_SElockdown = infections_SE[first_SE_lockdown_index : last_SE_lockdown_index + 1] # does not include last value
epi_datesI_SElockdown = datesI_SE[first_SE_lockdown_index : last_SE_lockdown_index + 1] # does not include last value

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
ax11.plot(epi_datesI_SElockdown, GompSElockdown, alpha=0.5, lw=2, label='Gompertz')
ax11.set_ylabel('New infections per day')
ax11.yaxis.set_tick_params(length=0)
ax11.xaxis.set_tick_params(length=0)
ax11.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax11.xaxis.set_major_locator(mdates.DayLocator(interval=7*2))
ax11.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig11.autofmt_xdate()
ax11.grid(b=True)
ax11.set_title('Sweden infections during UK lockdown and fitted first derivative of the Gompertz function')
legend = ax11.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax11.spines[spine].set_visible(False)
fig11.savefig("fig11.png")


#################
# Beta Analysis #
#################

# Calculate the proportion (%) of the population who remain susceptible, S1
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

for t, y in enumerate(beta_UK):
    if t == 0:
        beta_UK[t] = -(pop_UK/(I_UK[t]*S_UK[t])) * (S_UK[t]-pop_UK)
    else:
        beta_UK[t] = -(pop_UK/(I_UK[t]*S_UK[t])) * (S_UK[t]-S_UK[t-1])

for t, y in enumerate(beta_SE):
    if t == 0:
        beta_SE[t] = -(pop_SE/(I_SE[t]*S_SE[t])) * (S_SE[t]-pop_SE)
    else:
        beta_SE[t] = -(pop_SE/(I_SE[t]*S_SE[t])) * (S_SE[t]-S_SE[t-1])

for t, y in enumerate(beta_EN):
    if t == 0:
        beta_EN[t] = -(pop_EN/(I_EN[t]*S_EN[t])) * (S_EN[t]-pop_EN)
    else:
        beta_EN[t] = -(pop_EN/(I_EN[t]*S_EN[t])) * (S_EN[t]-S_EN[t-1])

# beta
fig12 = plt.figure(facecolor='w')
ax12 = fig12.add_subplot(111, axisbelow=True)
ax12.plot(datesI_UK, beta_UK, alpha=0.5, lw=2, label='UK')
ax12.plot(datesI_SE, beta_SE, alpha=0.5, lw=2, label='Sweden')
ax12.set_ylabel(r'$\beta$')
ax12.yaxis.set_tick_params(length=0)
ax12.xaxis.set_tick_params(length=0)
ax12.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax12.xaxis.set_major_locator(mdates.DayLocator(interval=7*8))
ax12.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig12.autofmt_xdate()
ax12.grid(b=True)
ax12.set_ylim(0,0.4)
ax12.set_title(r'$\beta$ from the SIR models for the UK and Sweden')
legend = ax12.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax12.spines[spine].set_visible(False)
fig12.savefig("fig12.png")

# S/N
fig13 = plt.figure(facecolor='w')
ax13 = fig13.add_subplot(111, axisbelow=True)
ax13.plot(datesI_UK, S1_UK, alpha=0.5, lw=2, label='UK')
ax13.plot(datesI_SE, S1_SE, alpha=0.5, lw=2, label='Sweden')
ax13.set_ylabel(r'$\frac{S}{N}$')
ax13.yaxis.set_tick_params(length=0)
ax13.xaxis.set_tick_params(length=0)
ax13.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax13.xaxis.set_major_locator(mdates.DayLocator(interval=7*8))
ax13.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig13.autofmt_xdate()
ax13.grid(b=True)
ax13.set_title(r'$\frac{S}{N}$ from the SIR models for the UK and Sweden')
legend = ax13.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax13.spines[spine].set_visible(False)
fig13.savefig("fig12.png")

# Identify the three UK/England lockdowns for figure 14.
# '1' masks the value, so '0' signifies lockdown.
S1_EN_lockdown = ma.masked_array(S1_EN, mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Beta vs percentage of population susceptible in the COVID-19 SIR model
fig14 = plt.figure(facecolor='w')
ax14 = fig14.add_subplot(111, axisbelow=True)
ax14.plot(S1_EN, beta_EN, 'g', alpha=0.5, lw=2, label='England without lockdown')
ax14.plot(S1_EN_lockdown, beta_EN, 'r', alpha=0.5, lw=2, label='England under lockdown')
ax14.plot(S1_SE, beta_SE, 'b', alpha=0.5, lw=2, label='Sweden')
ax14.set_xlabel(r'Percentage of population susceptible ($\frac{S}{N}$)')
ax14.set_ylabel(r'$\beta$')
ax14.yaxis.set_tick_params(length=0)
ax14.xaxis.set_tick_params(length=0)
ax14.grid(b=True)
ax14.set_xlim(100, 60)
ax14.set_ylim(0,0.3)
legend = ax14.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax14.spines[spine].set_visible(False)
fig14.savefig("fig3.pdf") # third figure in paper
ax14.set_title(r'$\beta$ vs $\frac{S}{N}$ in the COVID-19 SIR model, data up to 30 June 2021')
fig14.savefig("fig14.png")

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
        dbetadS1_UK[t] = 0

for t, y in enumerate(dbetadS1_SE):
    if dS1dt_SE[t] != 0:
        dbetadS1_SE[t] = dbetadt_SE[t]/dS1dt_SE[t]
    else:
        dbetadS1_SE[t] = 0

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
ax15.set_ylabel(r'$\od{\beta}{\frac{S}{N}}$')
ax15.yaxis.set_tick_params(length=0)
ax15.xaxis.set_tick_params(length=0)
ax15.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax15.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax15.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig15.autofmt_xdate()
ax15.grid(b=True)
ax15.set_xlim([dt.date(2020, 2, 28), dt.date(2020, 4, 17)])
ax15.set_ylim(-0.2,0.2)
legend = ax15.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax15.spines[spine].set_visible(False)
fig15.savefig("fig4.pdf") # fourth figure in paper
ax15.set_title(r'$\od{\beta}{\frac{S}{N}}$ as the UK transitioned into its first lockdown')
fig15.savefig("fig15.png")

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
ax16.set_ylabel(r'$\od{\beta}{\frac{S}{N}}$')
ax16.yaxis.set_tick_params(length=0)
ax16.xaxis.set_tick_params(length=0)
ax16.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax16.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax16.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig16.autofmt_xdate()
ax16.grid(b=True)
ax16.set_xlim([dt.date(2020, 2, 28), dt.date(2020, 4, 17)])
ax16.set_ylim(-0.2,0.2)
legend = ax16.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax16.spines[spine].set_visible(False)
fig16.savefig("fig4.pdf") # fourth figure in paper
ax16.set_title(r'Smoothed $\od{\beta}{\frac{S}{N}}$ as the UK transitioned into its first lockdown')
fig16.savefig("fig16.png")


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
ax17.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax17.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax17.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig17.autofmt_xdate()
ax17.grid(b=True)
ax17.set_xlim([dt.date(2020, 6, 29), dt.date(2020, 8, 17)])
ax17.set_ylim(0,4000)
ax17.set_title('The effect of making masks mandatory in enclosed public places in England')
legend = ax17.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax17.spines[spine].set_visible(False)
fig17.savefig("fig17.png")

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
ax18.set_ylabel(r'$\beta$ in the SIR model inferred from deaths')
ax18.yaxis.set_tick_params(length=0)
ax18.xaxis.set_tick_params(length=0)
ax18.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax18.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax18.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig18.autofmt_xdate()
ax18.grid(b=True)
ax18.set_xlim([dt.date(2020, 6, 29), dt.date(2020, 8, 17)])
ax18.set_ylim(0,0.2)
legend = ax18.legend()
legend.get_frame().set_alpha(1)
for spine in ('top', 'right', 'bottom', 'left'):
    ax18.spines[spine].set_visible(False)
fig18.savefig("fig5.pdf") # fifth figure in paper
ax18.set_title('The effect of making masks mandatory in enclosed public places in England')
fig18.savefig("fig18.png")


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
    f.write(str(datesD_UK[t]))
    f.write(' ')
f.write('\n')
f.write('UK_datesI: ')
for t, y in enumerate(datesI_UK):
    f.write(str(datesI_UK[t]))
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
    f.write(str(dbetadS1_UK[t]))
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
    f.write(str(datesD_SE[t]))
    f.write(' ')
f.write('\n')
f.write('Sweden_datesI: ')
for t, y in enumerate(datesI_SE):
    f.write(str(datesI_SE[t]))
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
    f.write(str(dbetadS1_SE[t]))
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
f.write('England_infections: ')
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
f.write('UK Gompertz function growth-rate coefficient during the lockdown minus the equivalent for Sweden.\n')
f.write('If the UK lockdown worked, this number should be negative and significantly different from zero.\n')
f.write(str(r_UKlockdown - r_SElockdown))
f.write('\n')
f.write('\n')

f.write('Statistic 2\n')
f.write('Change in beta relative to the proportion of the population that is susceptible in the UK during the transition from before the lockdown to during the lockdown minus the equivalent for Sweden.\n')
f.write('If the UK lockdown worked, this number should be positive and significantly different from zero.\n')
f.write(str(statistics.mean(epi_dbetadS1_UK_smoothed[25:74+1]) - statistics.mean(control_smoothed[25:74+1])))
f.write('\n')
f.write('\n')

f.write('Statistic 3\n')
f.write('Change in beta in England between 29 June 2020 and 17 August 2020, minus the equivalent for Sweden.\n')
f.write('If the mask policy was effective, this number should be negative and significantly different from zero.\n')
f.write(str((beta_EN_smoothed[196] - beta_EN_smoothed[147]) - (beta_SE_smoothed[196] - beta_SE_smoothed[147])))

f.close()