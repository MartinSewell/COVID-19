# The Effectiveness of Lockdowns, Face Masks and Vaccination Programmes Vis-à-Vis Mitigating COVID-19
# Martin Sewell
# martin.sewell@cantab.net
# 23 October 2024

import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import scipy.optimize as optimize
import statistics
from numpy import genfromtxt

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
plt.rcParams['text.usetex'] = True

# Function to parse dates in input files
def date_parser(d_bytes):
    s = d_bytes.decode('utf-8')
    return np.datetime64(dt.datetime.strptime(s, '%d/%m/%Y'))

# Function to smooth data
def smooth(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, mode='same')

# output file
f = open("output.txt","w", encoding='ascii')

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
ax1.grid(visible=True)
ax1.set_title('UK COVID-19 deaths')
legend = ax1.legend()
legend.get_frame().set_alpha(1)

#    ax1.spines[spine].set_visible(False)
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
ax2.grid(visible=True)
ax2.set_title('Sweden COVID-19 deaths')
legend = ax2.legend()
legend.get_frame().set_alpha(1)

#    ax2.spines[spine].set_visible(False)
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
ax3.grid(visible=True)
legend = ax3.legend()
legend.get_frame().set_alpha(1)

#    ax3.spines[spine].set_visible(False)
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
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax4.xaxis.set_major_locator(mdates.DayLocator(interval=17*4))
ax4.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig4.autofmt_xdate()
ax4.grid(visible=True)
ax4.set_title('UK COVID-19 SIR model')
legend = ax4.legend()
legend.get_frame().set_alpha(1)

#    ax4.spines[spine].set_visible(False)
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
ax5.grid(visible=True)
ax5.set_title('Sweden COVID-19 SIR model')
legend = ax5.legend()
legend.get_frame().set_alpha(1)

#    ax5.spines[spine].set_visible(False)
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
ax6.plot(epi_datesI_UK, GompUK, alpha=0.5, lw=2, label='Gompertz')
ax6.set_ylabel('New infections per day')
ax6.yaxis.set_tick_params(length=0)
ax6.xaxis.set_tick_params(length=0)
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax6.xaxis.set_major_locator(mdates.DayLocator(interval=7*4))
ax6.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig6.autofmt_xdate()
ax6.grid(visible=True)
ax6.set_title('UK infections and fitted first derivative of the Gompertz function')
legend = ax6.legend()
legend.get_frame().set_alpha(1)

#    ax6.spines[spine].set_visible(False)
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
ax7.grid(visible=True)
legend = ax7.legend()
legend.get_frame().set_alpha(1)
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
    raise ValueError(resultUKbefore.message)

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
ax8.grid(visible=True)
ax8.set_title('UK infections before lockdown and fitted first derivative of the Gompertz function')
legend = ax8.legend()
legend.get_frame().set_alpha(1)

#    ax8.spines[spine].set_visible(False)
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
ax9.grid(visible=True)
ax9.set_title('UK infections during lockdown and fitted first derivative of the Gompertz function')
legend = ax9.legend()
legend.get_frame().set_alpha(1)

#    ax9.spines[spine].set_visible(False)
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
ax10.grid(visible=True)
ax10.set_title('Sweden infections before UK lockdown and fitted first derivative of the Gompertz function')
legend = ax10.legend()
legend.get_frame().set_alpha(1)

#    ax10.spines[spine].set_visible(False)
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
ax11.grid(visible=True)
ax11.set_title('Sweden infections during UK lockdown and fitted first derivative of the Gompertz function')
legend = ax11.legend()
legend.get_frame().set_alpha(1)

#    ax11.spines[spine].set_visible(False)
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
ax12.grid(visible=True)
ax12.set_ylim(0,0.6) # ax12.set_ylim(0,0.4)
ax12.set_title(r'$\beta$ from the SIR models for the UK and Sweden')
legend = ax12.legend()
legend.get_frame().set_alpha(1)

#    ax12.spines[spine].set_visible(False)
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
ax13.grid(visible=True)
ax13.set_title(r'$\frac{S}{N}$ from the SIR models for the UK and Sweden')
legend = ax13.legend()
legend.get_frame().set_alpha(1)

#    ax13.spines[spine].set_visible(False)
fig13.savefig("fig13.png")

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
ax14.grid(visible=True)
ax14.set_xlim(80, 20) #ax14.set_xlim(100, 60)
ax14.set_ylim(0,0.6) #ax14.set_ylim(0,0.3)
legend = ax14.legend()
legend.get_frame().set_alpha(1)
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
ax15.set_ylabel(r'$\frac{\textrm{d}\beta}{\textrm{d}\frac{S}{N}}$')
ax15.yaxis.set_tick_params(length=0)
ax15.xaxis.set_tick_params(length=0)
ax15.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax15.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax15.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig15.autofmt_xdate()
ax15.grid(visible=True)
ax15.set_xlim([dt.date(2020, 2, 28), dt.date(2020, 4, 17)])
ax15.set_ylim(-0.25,0.25) # ax15.set_ylim(-0.2,0.2)
legend = ax15.legend()
legend.get_frame().set_alpha(1)

#    ax15.spines[spine].set_visible(False)
ax15.set_title(r'$\frac{\textrm{d}\beta}{\textrm{d}\frac{S}{N}}$ as the UK transitioned into its first lockdown')
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
ax16.set_ylabel(r'$\frac{\textrm{d}\beta}{\textrm{d}\frac{S}{N}}$')
ax16.yaxis.set_tick_params(length=0)
ax16.xaxis.set_tick_params(length=0)
ax16.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax16.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax16.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig16.autofmt_xdate()
ax16.grid(visible=True)
ax16.set_xlim([dt.date(2020, 2, 28), dt.date(2020, 4, 17)])
ax16.set_ylim(-0.25,0.25) # ax16.set_ylim(-0.2,0.2)
legend = ax16.legend()
legend.get_frame().set_alpha(1)
fig16.savefig("fig4.pdf") # fourth figure in paper
ax16.set_title(r'Smoothed $\frac{\textrm{d}\beta}{\textrm{d}\frac{S}{N}}$ as the UK transitioned into its first lockdown')
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
ax17.grid(visible=True)
ax17.set_xlim([dt.date(2020, 6, 29), dt.date(2020, 8, 17)])
ax17.set_ylim(0,3500) # ax17.set_ylim(0,4000)
ax17.set_title('The effect of making masks mandatory in enclosed public places in England')
legend = ax17.legend()
legend.get_frame().set_alpha(1)

#    ax17.spines[spine].set_visible(False)
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
ax18.grid(visible=True)
ax18.set_xlim([dt.date(2020, 6, 29), dt.date(2020, 8, 17)])
ax18.set_ylim(0,0.3) #ax18.set_ylim(0,0.2)
legend = ax18.legend()
legend.get_frame().set_alpha(1)
fig18.savefig("fig5.pdf") # fifth figure in paper
ax18.set_title('The effect of making masks mandatory in enclosed public places in England')
fig18.savefig("fig18.png")


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
axLV._get_lines.prop_cycler = axLV._get_lines.prop_cycler
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with a high vaccination rate')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with a medium vaccination rate')
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with a low vaccination rate')
axLV.set_ylabel('Excess mortality P-scores')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
figLV.savefig("lockdownlowvax.pdf")
axLV.set_title('Excess mortality for countries with low lockdown stringency')
figLV.savefig("lockdownlowvax.png")

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
axLV._get_lines.prop_cycler = axLV._get_lines.prop_cycler
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with a high vaccination rate')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with a medium vaccination rate')
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with a low vaccination rate')
axLV.set_ylabel('Excess mortality P-scores')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
figLV.savefig("lockdownmedvax.pdf")
axLV.set_title('Excess mortality for countries with medium lockdown stringency')
figLV.savefig("lockdownmedvax.png")


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
axLV._get_lines.prop_cycler = axLV._get_lines.prop_cycler
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with a high vaccination rate')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with a medium vaccination rate')
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with a low vaccination rate')
axLV.set_ylabel('Excess mortality P-scores')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
figLV.savefig("lockdownhighvax.pdf")
axLV.set_title('Excess mortality for countries with high lockdown stringency')
figLV.savefig("lockdownhighvax.png")



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
axLV._get_lines.prop_cycler = axLV._get_lines.prop_cycler
axLV.plot(dates, smooth(emhl, 28), alpha=0.5, lw=2, label='Countries with high lockdown stringency')
axLV.plot(dates, smooth(emml, 28), alpha=0.5, lw=2, label='Countries with medium lockdown stringency')
axLV.plot(dates, smooth(emll, 28), alpha=0.5, lw=2, label='Countries with low lockdown stringency')
axLV.set_ylabel('Excess mortality P-scores')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
figLV.savefig("lockdownlowvax.pdf")
axLV.set_title('Excess mortality for countries with low vaccination rates')
figLV.savefig("lockdownlowvax.png")

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
axLV._get_lines.prop_cycler = axLV._get_lines.prop_cycler
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low lockdown stringency')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium lockdown stringency')
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high lockdown stringency')
axLV.set_ylabel('Excess mortality P-scores')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
figLV.savefig("lockdownmedvax.pdf")
axLV.set_title('Excess mortality for countries with medium vaccination rates')
figLV.savefig("lockdownmedvax.png")


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
axLV._get_lines.prop_cycler = axLV._get_lines.prop_cycler
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low lockdown stringency')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium lockdown stringency')
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high lockdown stringency')
axLV.set_ylabel('Excess mortality P-scores')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
figLV.savefig("lockdownhighvax.pdf")
axLV.set_title('Excess mortality for countries with high vaccination rates')
figLV.savefig("lockdownhighvax.png")



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
axLV._get_lines.prop_cycler = axLV._get_lines.prop_cycler
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high mask stringency')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium mask stringency')
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low mask stringency')
axLV.set_ylabel('Excess mortality P-scores')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
figLV.savefig("maskslowvax.pdf")
axLV.set_title('Excess mortality for countries with low vaccination rates')
figLV.savefig("maskslowvax.png")

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
axLV._get_lines.prop_cycler = axLV._get_lines.prop_cycler
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high mask stringency')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium mask stringency')
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low mask stringency')
axLV.set_ylabel('Excess mortality P-scores')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
figLV.savefig("masksmedvax.pdf")
axLV.set_title('Excess mortality for countries with medium vaccination rates')
figLV.savefig("masksmedvax.png")

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
axLV._get_lines.prop_cycler = axLV._get_lines.prop_cycler
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with high mask stringency')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium mask stringency')
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with low mask stringency')
axLV.set_ylabel('Excess mortality P-scores')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
figLV.savefig("maskshighvax.pdf")
axLV.set_title('Excess mortality for countries with high vaccination rates')
figLV.savefig("maskshighvax.png")

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
axLV._get_lines.prop_cycler = axLV._get_lines.prop_cycler
axLV.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Countries with high vaccination rates')
axLV.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Countries with medium vaccination rates')
axLV.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Countries with low vaccination rates')
axLV.set_ylabel('Excess mortality P-scores')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
figLV.savefig("coviddeathslowvax.pdf")
axLV.set_title('Excess mortality for countries with a low COVID-19 death rate')
figLV.savefig("coviddeathslowvax.png")

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
axLV._get_lines.prop_cycler = axLV._get_lines.prop_cycler
axLV.plot(dates, smooth(emh, 28), alpha=0.5, lw=2, label='Countries with high vaccination rates')
axLV.plot(dates, smooth(emm, 28), alpha=0.5, lw=2, label='Countries with medium vaccination rates')
axLV.plot(dates, smooth(eml, 28), alpha=0.5, lw=2, label='Countries with low vaccination rates')
axLV.set_ylabel('Excess mortality P-scores')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
figLV.savefig("coviddeathsmedvax.pdf")
axLV.set_title('Excess mortality for countries with a medium COVID-19 death rate')
figLV.savefig("coviddeathsmedvax.png")

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
axLV._get_lines.prop_cycler = axLV._get_lines.prop_cycler
axLV.plot(dates, smooth(emh, 28), alpha=0.5, lw=2, label='Highly-vaccinated countries')
axLV.plot(dates, smooth(emm, 28), alpha=0.5, lw=2, label='Medium-vaccinated countries')
axLV.plot(dates, smooth(eml, 28), alpha=0.5, lw=2, label='Low-vaccinated countries')
axLV.set_ylabel('Excess mortality P-scores')
axLV.xaxis.set_tick_params(length=0)
axLV.yaxis.set_tick_params(length=0)
axLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figLV.autofmt_xdate()
axLV.grid(visible=True)
legend = axLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
figLV.savefig("coviddeathshighvax.pdf")
axLV.set_title('Excess mortality for countries with a high COVID-19 death rate')
figLV.savefig("coviddeathshighvax.png")


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
axCDLV._get_lines.prop_cycler = axCDLV._get_lines.prop_cycler
axCDLV.plot(dates, smooth(emlll, 28), alpha=0.5, lw=2, label='COVID-19 death rate low, lockdown stringency low, vaccination rate low')
axCDLV.plot(dates, smooth(emhll, 28), alpha=0.5, lw=2, label='COVID-19 death rate high, lockdown stringency low, vaccination rate low')
axCDLV.plot(dates, smooth(emlhl, 28), alpha=0.5, lw=2, label='COVID-19 death rate low, lockdown stringency high, vaccination rate low')
axCDLV.plot(dates, smooth(emllh, 28), alpha=0.5, lw=2, label='COVID-19 death rate low, lockdown stringency low, vaccination rate high')
axCDLV.set_ylabel('Excess mortality P-scores')
axCDLV.xaxis.set_tick_params(length=0)
axCDLV.yaxis.set_tick_params(length=0)
axCDLV.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
axCDLV.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axCDLV.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
figCDLV.autofmt_xdate()
axCDLV.grid(visible=True)
legend = axCDLV.legend(fancybox=True, framealpha=0.8)
legend.get_frame().set_alpha(1)
#    axCDLV.spines[spine].set_visible(False)
figCDLV.savefig("coviddeathslockdownvax.pdf")
axCDLV.set_title('COVID-19 death rate, lockdown stringency and vaccination rates')
figCDLV.savefig("coviddeathslockdownvax.png")


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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
p1 = ax19.plot(dates, smooth(cdh, 28), alpha=0.5, lw=2, label='COVID-19 death rate for countries with a high COVID-19 death rate')
p2 = ax19.plot(dates, smooth(cdl, 28), alpha=0.5, lw=2, label='COVID-19 death rate for countries with a low COVID-19 death rate')
p3 = ax20.plot(dates, smooth(emh, 28), alpha=0.5, lw=2, label='Excess mortality for countries with high COVID-19 death rate')
p4 = ax20.plot(dates, smooth(eml, 28), alpha=0.5, lw=2, label='Excess mortality for countries with low COVID-19 death rate')
ax19.set_ylabel('Daily COVID-19 deaths per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax20.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("coviddeaths.pdf")
ax19.set_title('COVID-19 death rate and excess mortality')
fig19.savefig("coviddeaths.png")


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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
p1 = ax19.plot(dates, smooth(ldh, 28), alpha=0.5, lw=2, label='Lockdown stringency in high-stringency countries')
p2 = ax19.plot(dates, smooth(ldl, 28), alpha=0.5, lw=2, label='Lockdown stringency in low-stringency countries')
p3 = ax20.plot(dates, smooth(emh, 28), alpha=0.5, lw=2, label='Excess mortality in high-stringency countries')
p4 = ax20.plot(dates, smooth(eml, 28), alpha=0.5, lw=2, label='Excess mortality in low-stringency countries')
ax19.set_ylabel('Lockdown stringency')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax20.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("lockdown.pdf") # sixth figure in paper
ax19.set_title('Lockdown stringency and excess mortality')
fig19.savefig("lockdown.png")



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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
p1 = ax19.plot(dates, smooth(mh, 28), alpha=0.5, lw=2, label='Mask index in high mask stringency countries')
p2 = ax19.plot(dates, smooth(ml, 28), alpha=0.5, lw=2, label='Mask index in low mask stringency countries')
p3 = ax20.plot(dates, smooth(emh, 28), alpha=0.5, lw=2, label='Excess mortality in high mask stringency countries')
p4 = ax20.plot(dates, smooth(eml, 28), alpha=0.5, lw=2, label='Excess mortality in low mask stringency countries')
ax19.set_ylabel('Face covering policies index')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax20.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("masks.pdf")
ax19.set_title('Face masks')
fig19.savefig("masks.png")


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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
p1 = ax19.plot(dates, smooth(vh, 28), alpha=0.5, lw=2, label='Vaccination rate in highly-vaccinated countries')
p2 = ax19.plot(dates, smooth(vm, 28), alpha=0.5, lw=2, label='Vaccination rate in medium-vaccinated countries')
p3 = ax19.plot(dates, smooth(vl, 28), alpha=0.5, lw=2, label='Vaccination rate in low-vaccinated countries')
p4 = ax20.plot(dates, smooth(cdh, 28), alpha=0.5, lw=2, label='COVID-19 death rate in highly-vaccinated countries')
p5 = ax20.plot(dates, smooth(cdm, 28), alpha=0.5, lw=2, label='COVID-19 death rate in medium-vaccinated countries')
p6 = ax20.plot(dates, smooth(cdl, 28), alpha=0.5, lw=2, label='COVID-19 death rate in low-vaccinated countries')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax20.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("vaccinationsCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths')
fig19.savefig("vaccinationsCD.png")


numclust =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
clust =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']

for c, d in enumerate(numclust):
    for a, b in enumerate(clust):
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
        ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
        if vaxhighvx.size > 0:
            cd1 = np.empty(len(dates))
            cd1[:] = np.nan
            for t, y in enumerate(cd1):
                cd1[t] = np.nanmean(vaxhighvx[t], axis=0)
            p1 = ax19.plot(dates, smooth(cd1, 28), alpha=0.5, lw=2, label='Vaccination rate in highly-vaccinated countries') # Vaccination rate in highly vaccinated countries
        if vaxmedvx.size > 0:
            cd2 = np.empty(len(dates))
            cd2[:] = np.nan
            for t, y in enumerate(cd2):
                cd2[t] = np.nanmean(vaxmedvx[t], axis=0)
            p2 = ax19.plot(dates, smooth(cd2, 28), alpha=0.5, lw=2, label='Vaccination rate in medium-vaccinated countries') # Vaccination rate in medium vaccinated countries
        if vaxlowvx.size > 0:
            cd3 = np.empty(len(dates))
            cd3[:] = np.nan
            for t, y in enumerate(cd3):
                cd3[t] = np.nanmean(vaxlowvx[t], axis=0)
            p3 = ax19.plot(dates, smooth(cd3, 28), alpha=0.5, lw=2, label='Vaccination rate in low-vaccinated countries') # Vaccination rate in low vaccinated countries
        if vaxhighcd.size > 0:
            cd4 = np.empty(len(dates))
            cd4[:] = np.nan
            for t, y in enumerate(cd4):
                cd4[t] = np.nanmean(vaxhighcd[t], axis=0)
            p4 = ax20.plot(dates, smooth(cd4, 28), alpha=0.5, lw=2, label='COVID-19 deaths in highly-vaccinated countries') # COVID-19 deaths for highly vaccinated countries
        if vaxmedcd.size > 0:
            cd5 = np.empty(len(dates))
            cd5[:] = np.nan
            for t, y in enumerate(cd5):
                cd5[t] = np.nanmean(vaxmedcd[t], axis=0)
            p5 = ax20.plot(dates, smooth(cd5, 28), alpha=0.5, lw=2, label='COVID-19 deaths in medium-vaccinated countries') # COVID-19 deaths for medium vaccinated countries
        if vaxlowcd.size > 0:
            cd6 = np.empty(len(dates))
            cd6[:] = np.nan
            for t, y in enumerate(cd6):
                cd6[t] = np.nanmean(vaxlowcd[t], axis=0)
            p6 = ax20.plot(dates, smooth(cd6, 28), alpha=0.5, lw=2, label='COVID-19 deaths in low-vaccinated countries') # COVID-19 deaths for low vaccinated countries
        if (vaxhighvx.size > 0 or vaxmedvx.size > 0 or vaxlowvx.size > 0 or vaxhighcd.size > 0 or vaxmedcd.size > 0 or vaxlowcd.size > 0):
            ax19.set_ylabel('Daily vaccination doses per million')
            ax20.set_ylabel('Daily COVID-19 deaths per million')
            ax19.yaxis.set_tick_params(length=0)
            ax20.xaxis.set_tick_params(length=0)
            ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
            ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
            fig19.autofmt_xdate()
            ax19.grid(visible=True)
            legend.get_frame().set_alpha(1)
            lines, labels = ax19.get_legend_handles_labels()
            lines2, labels2 = ax20.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
            fig19.savefig(d+"_vaccinationsCD"+b+".pdf")
            titlecd = 'Vaccinations and COVID-19 deaths for countries in cluster '+b
            ax19.set_title(titlecd)
            fig19.savefig(d+"_vaccinationsCD"+b+".png")
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
        ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
        if vaxhighvx.size > 0:
            em1 = np.empty(len(dates))
            em1[:] = np.nan
            for t, y in enumerate(em1):
                em1[t] = np.nanmean(vaxhighvx[t], axis=0)
            p1 = ax19.plot(dates, smooth(em1, 28), alpha=0.5, lw=2, label='Vaccination rate in highly-vaccinated countries') # Vaccination rate in highly vaccinated countries
        if vaxmedvx.size > 0:
            em2 = np.empty(len(dates))
            em2[:] = np.nan
            for t, y in enumerate(em2):
                em2[t] = np.nanmean(vaxmedvx[t], axis=0)
            p2 = ax19.plot(dates, smooth(em2, 28), alpha=0.5, lw=2, label='Vaccination rate in medium-vaccinated countries') # Vaccination rate in medium vaccinated countries
        if vaxlowvx.size > 0:
            em3 = np.empty(len(dates))
            em3[:] = np.nan
            for t, y in enumerate(em3):
                em3[t] = np.nanmean(vaxlowvx[t], axis=0)
            p3 = ax19.plot(dates, smooth(em3, 28), alpha=0.5, lw=2, label='Vaccination rate in low-vaccinated countries') # Vaccination rate in low vaccinated countries
        if vaxhighem.size > 0:
            em4 = np.empty(len(dates))
            em4[:] = np.nan
            for t, y in enumerate(em4):
                em4[t] = np.nanmean(vaxhighem[t], axis=0)
            p4 = ax20.plot(dates, smooth(em4, 28), alpha=0.5, lw=2, label='Excess mortality in highly-vaccinated countries') # Excess mortality for highly vaccinated countries
        if vaxmedem.size > 0:
            em5 = np.empty(len(dates))
            em5[:] = np.nan
            for t, y in enumerate(em5):
                em5[t] = np.nanmean(vaxmedem[t], axis=0)
            p5 = ax20.plot(dates, smooth(em5, 28), alpha=0.5, lw=2, label='Excess mortality in medium-vaccinated countries') # Excess mortality for medium vaccinated countries
        if vaxlowem.size > 0:
            em6 = np.empty(len(dates))
            em6[:] = np.nan
            for t, y in enumerate(em6):
                em6[t] = np.nanmean(vaxlowem[t], axis=0)
            p6 = ax20.plot(dates, smooth(em6, 28), alpha=0.5, lw=2, label='Excess mortality in low-vaccinated countries') # Excess mortality for low vaccinated countries
        if (vaxhighvx.size > 0 or vaxmedvx.size > 0 or vaxlowvx.size > 0 or vaxhighcd.size > 0 or vaxmedcd.size > 0 or vaxlowcd.size > 0):
            ax19.set_ylabel('Daily vaccination doses per million')
            ax20.set_ylabel('Daily COVID-19 deaths per million')
            ax19.yaxis.set_tick_params(length=0)
            ax20.xaxis.set_tick_params(length=0)
            ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
            ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
            fig19.autofmt_xdate()
            ax19.grid(visible=True)
            legend.get_frame().set_alpha(1)
            lines, labels = ax19.get_legend_handles_labels()
            lines2, labels2 = ax20.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
            fig19.savefig(d+"_vaccinationsEM"+b+".pdf")
            titleem = 'Vaccinations and excess mortality for countries in cluster '+b
            ax19.set_title(titleem)
            fig19.savefig(d+"_vaccinationsEM"+b+".png")
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
eml = np.empty(len(dates))
eml[:] = np.nan
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
p1 = ax19.plot(dates, smooth(vh, 28), alpha=0.5, lw=2, label='Vaccination rate in highly-vaccinated countries')
p2 = ax19.plot(dates, smooth(vm, 28), alpha=0.5, lw=2, label='Vaccination rate in medium-vaccinated countries')
p3 = ax19.plot(dates, smooth(vl, 28), alpha=0.5, lw=2, label='Vaccination rate in low-vaccinated countries')
p4 = ax20.plot(dates, smooth(emh, 28), alpha=0.5, lw=2, label='Excess mortality in highly-vaccinated countries')
p5 = ax20.plot(dates, smooth(emm, 28), alpha=0.5, lw=2, label='Excess mortality in medium-vaccinated countries')
p6 = ax20.plot(dates, smooth(eml, 28), alpha=0.5, lw=2, label='Excess mortality in low-vaccinated countries')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax20.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("vaccinationsEM.pdf")
ax19.set_title('Vaccinations and excess mortality')
fig19.savefig("vaccinationsEM.png")
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
        ls = ms = vs = cds = ems = ''
        if sum(CNTRYlockdown0) > 0:
            CNTRY1 = axc.plot(dates, CNTRYlockdown, alpha=0.5, lw=2, label='Lockdown stringency', color='grey')
            if sum(CNTRYmasks0) > 0 or sum(CNTRYvax0) > 0 or sum(CNTRYcoviddeaths0) > 0:
                ls = 'Lockdown stringency, '
            else:
                ls = 'Lockdown stringency'
        if sum(CNTRYmasks0) > 0:
            CNTRYmasks2 = np.multiply(CNTRYmasks, 10)
            CNTRY2 = axc.plot(dates, CNTRYmasks2, alpha=0.5, lw=2, label='Mask index', color='c')
            if (not (sum(CNTRYlockdown0) > 0)):
                ms = r'Mask index $\times$10'
            else:
                ms = r'mask index $\times$10'
            if sum(CNTRYvax0) > 0 and sum(CNTRYcoviddeaths0) > 0:
                ms = ms + ', '
            if sum(CNTRYvax0) > 0 and (not (sum(CNTRYcoviddeaths0) > 0)):
                ms = ms + ' and'
            if (not (sum(CNTRYvax0) > 0)) and sum(CNTRYcoviddeaths0) > 0:
                ms = ms + ' and'
        if sum(CNTRYvax0) > 0:
            CNTRYvax2 = np.multiply(CNTRYvax, 100)
            CNTRY3 = axc.plot(dates, smooth(CNTRYvax2, 28), alpha=0.5, lw=2, label='Vaccination rate', color='b')
        if sum(CNTRYcoviddeaths0) > 0:
            CNTRYcoviddeaths2 = np.multiply(CNTRYcoviddeaths, 10)
            CNTRY4 = axc.plot(dates, smooth(CNTRYcoviddeaths2, 28), alpha=0.5, lw=2, label='COVID-19 death rate', color='k')
             
        if sum(CNTRYvax0) > 0:
            if (sum(CNTRYlockdown0) > 0 or sum(CNTRYmasks0) > 0) and sum(CNTRYcoviddeaths0) > 0:
                vs = 'daily vaccination doses/10,000 and '
            if (sum(CNTRYlockdown0) > 0 or sum(CNTRYmasks0) > 0) and (not sum(CNTRYcoviddeaths0) > 0):
                vs = 'daily vaccination doses/10,000'
            if not(sum(CNTRYlockdown0) > 0 or sum(CNTRYmasks0) > 0) and sum(CNTRYcoviddeaths0) > 0:
                vs = 'Daily vaccination doses/10,000 and '
            if not(sum(CNTRYlockdown0) > 0 or sum(CNTRYmasks0) > 0) and (not sum(CNTRYcoviddeaths0) > 0):
                vs = 'Daily vaccination doses/10,000'

        if sum(CNTRYcoviddeaths0) > 0:
            if (sum(CNTRYlockdown0) > 0 or sum(CNTRYmasks0) > 0 or sum(CNTRYvax0) > 0):
                cds = 'daily COVID-19 deaths/100,000'
            else:
                cds = 'Daily COVID-19 deaths/100,000'

        ylabeltext = ""
        if ls != "":
            ylabeltext = ylabeltext + ls
        if ms != "":
            ylabeltext = ylabeltext + ms
        if vs != "":
            ylabeltext = ylabeltext + "\n" + vs
        if cds != "":
            ylabeltext = ylabeltext + "\n" + cds
        
        axc.set_ylabel(ylabeltext)



        if not all(val == 0 for val in CNTRYexcessmortality0):
            axem = axc.twinx()
            CNTRY5 = axem.plot(dates, smooth(CNTRYexcessmortality, 28), alpha=0.5, lw=2, label='Excess mortality', color='r')
            axem._get_lines.prop_cycler = axc._get_lines.prop_cycler
            axem.set_ylabel('Excess mortality P-scores')
            lines, labels = axc.get_legend_handles_labels()
            lines2, labels2 = axem.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
        else:
            lines, labels = axc.get_legend_handles_labels()
            plt.legend(lines, labels, fancybox=True, framealpha=0.8)
        axc.yaxis.set_tick_params(length=0)
        axc.xaxis.set_tick_params(length=0)
        axc.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
        if (c == 'Falkland Islands') or (c == 'Saint Helena'):
            axc.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        else:
            axc.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axc.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
        figc.autofmt_xdate()
        axc.grid(visible=True)
        legend.get_frame().set_alpha(1)
        figc.savefig(cf + ".pdf")
        axc.set_title(c)
        figc.savefig(cf + ".png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
SWEl = ax19.plot(dates, lockdown_SWE, alpha=0.5, lw=2, label='Sweden')
GBRl = ax19.plot(dates, lockdown_GBR, alpha=0.5, lw=2, label='UK')
SWEem = ax20.plot(dates, smooth(deaths_SWE, 28), alpha=0.5, lw=2, label='Sweden')
GBRem = ax20.plot(dates, smooth(deaths_GBR, 28), alpha=0.5, lw=2, label='UK')
ax19.set_ylabel('Lockdown')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("SWEGBR.pdf")
ax19.set_title('Lockdowns and excess mortality for Sweden and the United Kingdom')
fig19.savefig("SWEGBR.png")




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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
DZAv = ax19.plot(dates, smooth(vax_DZA, 28), alpha=0.5, lw=2, label='Antigua and Barbuda vaccination rate')
EGYv = ax19.plot(dates, smooth(vax_EGY, 28), alpha=0.5, lw=2, label='Egypt vaccination rate')
DZAcd = ax20.plot(dates, smooth(deaths_DZA, 28), alpha=0.5, lw=2, label='Antigua and Barbuda COVID-19 death rate')
EGYcd = ax20.plot(dates, smooth(deaths_EGY, 28), alpha=0.5, lw=2, label='Egypt COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("AlgeriaEgyptCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Antigua and Barbuda and Egypt')
fig19.savefig("AlgeriaEgyptCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
DZAv = ax19.plot(dates, smooth(vax_DZA, 28), alpha=0.5, lw=2, label='Antigua and Barbuda vaccination rate')
EGYv = ax19.plot(dates, smooth(vax_EGY, 28), alpha=0.5, lw=2, label='Egypt vaccination rate')
DZAem = ax20.plot(dates, smooth(deaths_DZA, 28), alpha=0.5, lw=2, label='Antigua and Barbuda excess mortality')
EGYem = ax20.plot(dates, smooth(deaths_EGY, 28), alpha=0.5, lw=2, label='Egypt excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("AlgeriaEgyptEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Antigua and Barbuda and Egypt')
fig19.savefig("AlgeriaEgyptEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
ATGv = ax19.plot(dates, smooth(vax_ATG, 28), alpha=0.5, lw=2, label='Antigua and Barbuda vaccination rate')
CUBv = ax19.plot(dates, smooth(vax_CUB, 28), alpha=0.5, lw=2, label='Cuba vaccination rate')
ATGem = ax20.plot(dates, smooth(deaths_ATG, 28), alpha=0.5, lw=2, label='Antigua and Barbuda excess mortality')
CUBem = ax20.plot(dates, smooth(deaths_CUB, 28), alpha=0.5, lw=2, label='Cuba excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("AntiguaandBarbudaCubaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Antigua and Barbuda and Cuba')
fig19.savefig("AntiguaandBarbudaCubaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
ARGv = ax19.plot(dates, smooth(vax_ARG, 28), alpha=0.5, lw=2, label='Argentina vaccination rate')
PRYv = ax19.plot(dates, smooth(vax_PRY, 28), alpha=0.5, lw=2, label='Paraguay vaccination rate')
ARGem = ax20.plot(dates, smooth(deaths_ARG, 28), alpha=0.5, lw=2, label='Argentina excess mortality')
PRYem = ax20.plot(dates, smooth(deaths_PRY, 28), alpha=0.5, lw=2, label='Paraguay excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("ArgentinaParaguayEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Argentina and Paraguay')
fig19.savefig("ArgentinaParaguayEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
ARMv = ax19.plot(dates, smooth(vax_ARM, 28), alpha=0.5, lw=2, label='Armenia vaccination rate')
AZEv = ax19.plot(dates, smooth(vax_AZE, 28) , alpha=0.5, lw=2, label='Azerbaijan vaccination rate')
ARMcd = ax20.plot(dates, smooth(deaths_ARM, 28), alpha=0.5, lw=2, label='Armenia COVID-19 death rate')
AZEcd = ax20.plot(dates, smooth(deaths_AZE, 28), alpha=0.5, lw=2, label='Azerbaijan COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("ArmeniaAzerbaijanCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Armenia and Azerbaijan')
fig19.savefig("ArmeniaAzerbaijanCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
ARMv = ax19.plot(dates, smooth(vax_ARM, 28), alpha=0.5, lw=2, label='Armenia vaccination rate')
AZEv = ax19.plot(dates, smooth(vax_AZE, 28) , alpha=0.5, lw=2, label='Azerbaijan vaccination rate')
ARMem = ax20.plot(dates, smooth(deaths_ARM, 28), alpha=0.5, lw=2, label='Armenia excess mortality')
AZEem = ax20.plot(dates, smooth(deaths_AZE, 28), alpha=0.5, lw=2, label='Azerbaijan excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("ArmeniaAzerbaijanEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Armenia and Azerbaijan')
fig19.savefig("ArmeniaAzerbaijanEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
AUSv = ax19.plot(dates, smooth(vax_AUS, 28), alpha=0.5, lw=2, label='Australia vaccination rate')
NZLv = ax19.plot(dates, smooth(vax_NZL, 28), alpha=0.5, lw=2, label='New Zealand vaccination rate')
AUScd = ax20.plot(dates, smooth(deaths_AUS, 28), alpha=0.5, lw=2, label='Australia COVID-19 death rate')
NZLcd = ax20.plot(dates, smooth(deaths_NZL, 28), alpha=0.5, lw=2, label='New Zealand COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 death rate')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("AustraliaNewZealandCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Australia and New Zealand')
fig19.savefig("AustraliaNewZealandCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
AUSv = ax19.plot(dates, smooth(vax_AUS, 28), alpha=0.5, lw=2, label='Australia vaccination rate')
NZLv = ax19.plot(dates, smooth(vax_NZL, 28), alpha=0.5, lw=2, label='New Zealand vaccination rate')
AUSem = ax20.plot(dates, smooth(deaths_AUS, 28), alpha=0.5, lw=2, label='Australia excess mortality')
NZLem = ax20.plot(dates, smooth(deaths_NZL, 28), alpha=0.5, lw=2, label='New Zealand excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("AustraliaNewZealandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Australia and New Zealand')
fig19.savefig("AustraliaNewZealandEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
BLRv = ax19.plot(dates, smooth(vax_BLR, 28), alpha=0.5, lw=2, label='Belarus vaccination rate')
DNKv = ax19.plot(dates, smooth(vax_DNK, 28), alpha=0.5, lw=2, label='Denmark vaccination rate')
BLRcd = ax20.plot(dates, smooth(deaths_BLR, 28), alpha=0.5, lw=2, label='Belarus COVID-19 death rate')
DNKcd = ax20.plot(dates, smooth(deaths_DNK, 28), alpha=0.5, lw=2, label='Denmark COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("BelarusDenmarkCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Belarus and Denmark')
fig19.savefig("BelarusDenmarkCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
BLRv = ax19.plot(dates, smooth(vax_BLR, 28), alpha=0.5, lw=2, label='Belarus vaccination rate')
DNKv = ax19.plot(dates, smooth(vax_DNK, 28), alpha=0.5, lw=2, label='Denmark vaccination rate')
BLRem = ax20.plot(dates, smooth(deaths_BLR, 28), alpha=0.5, lw=2, label='Belarus excess mortality')
DNKem = ax20.plot(dates, smooth(deaths_DNK, 28), alpha=0.5, lw=2, label='Denmark excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("BelarusDenmarkEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Belarus and Denmark')
fig19.savefig("BelarusDenmarkEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
BTNv = ax19.plot(dates, smooth(vax_BTN, 28), alpha=0.5, lw=2, label='Bhutan vaccination rate')
SGPv = ax19.plot(dates, smooth(vax_SGP, 28), alpha=0.5, lw=2, label='Singapore vaccination rate')
BTNcd = ax20.plot(dates, smooth(deaths_BTN, 28), alpha=0.5, lw=2, label='Bhutan COVID-19 death rate')
SGPcd = ax20.plot(dates, smooth(deaths_SGP, 28), alpha=0.5, lw=2, label='Singapore COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("BhutanSingaporeCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Bhutan and Singapore')
fig19.savefig("BhutanSingaporeCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
BTNv = ax19.plot(dates, smooth(vax_BTN, 28), alpha=0.5, lw=2, label='Bhutan vaccination rate')
VNMv = ax19.plot(dates, smooth(vax_VNM, 28), alpha=0.5, lw=2, label='Vietnam vaccination rate')
BTNcd = ax20.plot(dates, smooth(deaths_BTN, 28), alpha=0.5, lw=2, label='Bhutan COVID-19 death rate')
VNMcd = ax20.plot(dates, smooth(deaths_VNM, 28), alpha=0.5, lw=2, label='Vietnam COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("BhutanVietnamCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Bhutan and Vietnam')
fig19.savefig("BhutanVietnamCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
BTNv = ax19.plot(dates, smooth(vax_BTN, 28), alpha=0.5, lw=2, label='Bhutan vaccination rate')
SGPv = ax19.plot(dates, smooth(vax_SGP, 28), alpha=0.5, lw=2, label='Singapore vaccination rate')
BTNem = ax20.plot(dates, smooth(deaths_BTN, 28), alpha=0.5, lw=2, label='Bhutan excess mortality')
SGPem = ax20.plot(dates, smooth(deaths_SGP, 28), alpha=0.5, lw=2, label='Singapore excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("BhutanSingaporeEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Bhutan and Singapore')
fig19.savefig("BhutanSingaporeEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
BTNv = ax19.plot(dates, smooth(vax_BTN, 28), alpha=0.5, lw=2, label='Bhutan vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
BTNcd = ax20.plot(dates, smooth(deaths_BTN, 28), alpha=0.5, lw=2, label='Bhutan COVID-19 death rate')
THAcd = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("BhutanThailandCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Bhutan and Thailand')
fig19.savefig("BhutanThailandCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
BTNv = ax19.plot(dates, smooth(vax_BTN, 28), alpha=0.5, lw=2, label='Bhutan vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
BTNem = ax20.plot(dates, smooth(deaths_BTN, 28), alpha=0.5, lw=2, label='Bhutan excess mortality')
THAem = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("BhutanThailandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Bhutan and Thailand')
fig19.savefig("BhutanThailandEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
BIHv = ax19.plot(dates, smooth(vax_BIH, 28), alpha=0.5, lw=2, label='Bosnia and Herzegovina vaccination rate')
ROUv = ax19.plot(dates, smooth(vax_ROU, 28), alpha=0.5, lw=2, label='Romania vaccination rate')
BIHem = ax20.plot(dates, smooth(deaths_BIH, 28), alpha=0.5, lw=2, label='Bosnia and Herzegovina excess mortality')
ROUem = ax20.plot(dates, smooth(deaths_ROU, 28), alpha=0.5, lw=2, label='Romania excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("BosniaandHerzegovinaRomaniaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Bosnia and Herzegovina and Romania')
fig19.savefig("BosniaandHerzegovinaRomaniaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
BGRv = ax19.plot(dates, smooth(vax_BGR, 28), alpha=0.5, lw=2, label='Bulgaria vaccination rate')
SRBv = ax19.plot(dates, smooth(vax_SRB, 28), alpha=0.5, lw=2, label='Serbia vaccination rate')
BGRem = ax20.plot(dates, smooth(deaths_BGR, 28), alpha=0.5, lw=2, label='Bulgaria excess mortality')
SRBem = ax20.plot(dates, smooth(deaths_SRB, 28), alpha=0.5, lw=2, label='Serbia excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("BulgariaSerbiaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Bulgaria and Serbia')
fig19.savefig("BulgariaSerbiaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
BDIv = ax19.plot(dates, smooth(vax_BDI, 28), alpha=0.5, lw=2, label='Burundi vaccination rate')
ERIv = ax19.plot(dates, smooth(vax_ERI, 28) , alpha=0.5, lw=2, label='Eritrea vaccination rate')
BDIcd = ax20.plot(dates, smooth(deaths_BDI, 28), alpha=0.5, lw=2, label='Burundi COVID-19 death rate')
ERIcd = ax20.plot(dates, smooth(deaths_ERI, 28), alpha=0.5, lw=2, label='Eritrea COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("BurundiEritreaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Burundi and Eritrea')
fig19.savefig("BurundiEritreaCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
BDIv = ax19.plot(dates, smooth(vax_BDI, 28), alpha=0.5, lw=2, label='Burundi vaccination rate')
SYCv = ax19.plot(dates, smooth(vax_SYC, 28), alpha=0.5, lw=2, label='Seychelles vaccination rate')
BDIcd = ax20.plot(dates, smooth(deaths_BDI, 28), alpha=0.5, lw=2, label='Burundi COVID-19 death rate')
SYCcd = ax20.plot(dates, smooth(deaths_SYC, 28), alpha=0.5, lw=2, label='Seychelles COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("BurundiSeychellesCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Burundi and Seychelles')
fig19.savefig("BurundiSeychellesCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
KHMv = ax19.plot(dates, smooth(vax_KHM, 28), alpha=0.5, lw=2, label='Cambodia vaccination rate')
VNMv = ax19.plot(dates, smooth(vax_VNM, 28), alpha=0.5, lw=2, label='Vietnam vaccination rate')
KHMcd = ax20.plot(dates, smooth(deaths_KHM, 28), alpha=0.5, lw=2, label='Cambodia COVID-19 death rate')
VNMcd = ax20.plot(dates, smooth(deaths_VNM, 28), alpha=0.5, lw=2, label='Vietnam COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CambodiaVietnamCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Cambodia and Vietnam')
fig19.savefig("CambodiaVietnamCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
CANv = ax19.plot(dates, smooth(vax_CAN, 28), alpha=0.5, lw=2, label='Canada vaccination rate')
CUBv = ax19.plot(dates, smooth(vax_CUB, 28), alpha=0.5, lw=2, label='Cuba vaccination rate')
CANem = ax20.plot(dates, smooth(deaths_CAN, 28), alpha=0.5, lw=2, label='Canada excess mortality')
CUBem = ax20.plot(dates, smooth(deaths_CUB, 28), alpha=0.5, lw=2, label='Cuba excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CanadaCubaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Canada and Cuba')
fig19.savefig("CanadaCubaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
CANv = ax19.plot(dates, smooth(vax_CAN, 28), alpha=0.5, lw=2, label='Canada vaccination rate')
VCTv = ax19.plot(dates, smooth(vax_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines vaccination rate')
CANem = ax20.plot(dates, smooth(deaths_CAN, 28), alpha=0.5, lw=2, label='Canada excess mortality')
VCTem = ax20.plot(dates, smooth(deaths_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CanadaSaintVincentandtheGrenadinesEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Canada and Saint Vincent and the Grenadines')
fig19.savefig("CanadaSaintVincentandtheGrenadinesEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
CPVv = ax19.plot(dates, smooth(vax_CPV, 28), alpha=0.5, lw=2, label='Cape Verde vaccination rate')
TUNv = ax19.plot(dates, smooth(vax_TUN, 28), alpha=0.5, lw=2, label='Tunisia vaccination rate')
CPVem = ax20.plot(dates, smooth(deaths_CPV, 28), alpha=0.5, lw=2, label='Cape Verde excess mortality')
TUNem = ax20.plot(dates, smooth(deaths_TUN, 28), alpha=0.5, lw=2, label='Tunisia excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CapeVerdeTunisiaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Cape Verde and Tunisia')
fig19.savefig("CapeVerdeTunisiaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVcd = ax20.plot(dates, smooth(deaths_HRVs, 28), alpha=0.5, lw=2, label='Croatia COVID-19 death rate')
HUNcd = ax20.plot(dates, smooth(deaths_HUNs, 28), alpha=0.5, lw=2, label='Hungary COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungaryCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Croatia and Hungary')
fig19.savefig("CroatiaHungaryCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVcd = ax20.plot(dates, smooth(deaths_HRVs, 28), alpha=0.5, lw=2, label='Croatia COVID-19 death rate')
HUNcd = ax20.plot(dates, smooth(deaths_HUNs, 28), alpha=0.5, lw=2, label='Hungary COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)]) # zoom in on this period
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungaryCDz.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Croatia and Hungary')
fig19.savefig("CroatiaHungaryCDz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNcd = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary COVID-19 death rate minus Croatia COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungaryCDdiffz.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Croatia and Hungary (differences)')
fig19.savefig("CroatiaHungaryCDdiffz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel('Excess mortality P-scores (all ages)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungaryEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Croatia and Hungary')
fig19.savefig("CroatiaHungaryEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel('Excess mortality P-scores (all ages)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)]) # zoom in on this period
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungaryEMz.pdf")
ax19.set_title('Vaccinations and excess mortality for Croatia and Hungary')
fig19.savefig("CroatiaHungaryEMz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel(r'Excess mortality P-scores (ages 0--14)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_0_14EM.pdf")
ax19.set_title(r'Vaccinations and excess mortality for ages 0--14 for Croatia and Hungary')
fig19.savefig("CroatiaHungary_0_14EM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel(r'Excess mortality P-scores (ages 0--14)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_0_14EMz.pdf")
ax19.set_title(r'Vaccinations and excess mortality for ages 0--14 for Croatia and Hungary')
fig19.savefig("CroatiaHungary_0_14EMz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNcd = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel(r'Excess mortality P-scores (ages 0--14)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_0_14EMdiffz.pdf")
ax19.set_title(r'Vaccinations and excess mortality for ages 0--14 for Croatia and Hungary (differences)')
fig19.savefig("CroatiaHungary_0_14EMdiffz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel(r'Excess mortality P-scores (ages 15--64)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_15_64EM.pdf")
ax19.set_title(r'Vaccinations and excess mortality for ages 15--64 for Croatia and Hungary')
fig19.savefig("CroatiaHungary_15_64EM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel(r'Excess mortality P-scores (ages 15--64)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_15_64EMz.pdf")
ax19.set_title(r'Vaccinations and excess mortality for ages 15--64 for Croatia and Hungary')
fig19.savefig("CroatiaHungary_15_64EMz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNcd = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel(r'Excess mortality P-scores (ages 15--64)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_15_64EMdiffz.pdf")
ax19.set_title(r'Vaccinations and excess mortality for ages 15--64 for Croatia and Hungary (differences)')
fig19.savefig("CroatiaHungary_15_64EMdiffz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel(r'Excess mortality P-scores (ages 65--74)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_65_74EM.pdf")
ax19.set_title(r'Vaccinations and excess mortality for ages 65--74 for Croatia and Hungary')
fig19.savefig("CroatiaHungary_65_74EM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel(r'Excess mortality P-scores (ages 65--74)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_65_74EMz.pdf")
ax19.set_title(r'Vaccinations and excess mortality for ages 65--74 for Croatia and Hungary')
fig19.savefig("CroatiaHungary_65_74EMz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNcd = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary COVID-19 death rate minus Croatia COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel(r'Excess mortality P-scores (ages 65--74)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_65_74EMdiffz.pdf")
ax19.set_title(r'Vaccinations and excess mortality for ages 65--74 for Croatia and Hungary (differences)')
fig19.savefig("CroatiaHungary_65_74EMdiffz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel(r'Excess mortality P-scores (ages 75--84)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_75_84EM.pdf")
ax19.set_title(r'Vaccinations and excess mortality for ages 75--84 for Croatia and Hungary')
fig19.savefig("CroatiaHungary_75_84EM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel(r'Excess mortality P-scores (ages 75--84)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_75_84EMz.pdf")
ax19.set_title(r'Vaccinations and excess mortality for ages 75--84 for Croatia and Hungary')
fig19.savefig("CroatiaHungary_75_84EMz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNcd = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary COVID-19 excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel(r'Excess mortality P-scores (ages 75--84)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_75_84EMdiffz.pdf")
ax19.set_title(r'Vaccinations and excess mortality for ages 75--84 for Croatia and Hungary (differences)')
fig19.savefig("CroatiaHungary_75_84EMzdiff.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel('Excess mortality P-scores (ages 85+)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_85pEM.pdf")
ax19.set_title('Vaccinations and excess mortality for ages 85+ for Croatia and Hungary')
fig19.savefig("CroatiaHungary_85pEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVv = ax19.plot(dates, smooth(vax_HRV, 28), alpha=0.5, lw=2, label='Croatia vaccination rate')
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
HRVem = ax20.plot(dates, smooth(deaths_HRV, 28), alpha=0.5, lw=2, label='Croatia excess mortality')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel('Excess mortality P-scores (ages 85+)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_85pEMz.pdf")
ax19.set_title('Vaccinations and excess mortality for ages 85+ for Croatia and Hungary')
fig19.savefig("CroatiaHungary_85pEMz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNcd = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Daily vaccination doses per million (all ages)')
ax20.set_ylabel('Excess mortality P-scores (ages 85+)')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungary_85pEMdiffz.pdf")
ax19.set_title('Vaccinations and excess mortality for ages 85+ for Croatia and Hungary (differences)')
fig19.savefig("CroatiaHungary_85pEMdiffz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNem = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungaryEMdiff.pdf")
ax19.set_title('Vaccinations and excess mortality for Croatia and Hungary (differences)')
fig19.savefig("CroatiaHungaryEMdiff.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HRVHUNv = ax19.plot(dates, smooth(vax_diff, 28), alpha=0.5, lw=2, label='Hungary vaccination rate minus Croatia vaccination rate')
HRVHUNem = ax20.plot(dates, smooth(deaths_diff, 28), alpha=0.5, lw=2, label='Hungary excess mortality minus Croatia excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.set_xlim([dt.date(2020, 1, 1), dt.date(2021, 6, 5)])
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CroatiaHungaryEMdiffz.pdf")
ax19.set_title('Vaccinations and excess mortality for Croatia and Hungary (differences)')
fig19.savefig("CroatiaHungaryEMdiffz.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
CUBv = ax19.plot(dates, smooth(vax_CUB, 28), alpha=0.5, lw=2, label='Cuba vaccination rate')
JAMv = ax19.plot(dates, smooth(vax_JAM, 28), alpha=0.5, lw=2, label='Jamaica vaccination rate')
CUBcd = ax20.plot(dates, smooth(deaths_CUB, 28), alpha=0.5, lw=2, label='Cuba COVID-19 death rate')
JAMcd = ax20.plot(dates, smooth(deaths_JAM, 28), alpha=0.5, lw=2, label='Jamaica COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CubaJamaicaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Cuba and Jamaica')
fig19.savefig("CubaJamaicaCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
CUBv = ax19.plot(dates, smooth(vax_CUB, 28), alpha=0.5, lw=2, label='Cuba vaccination rate')
JAMv = ax19.plot(dates, smooth(vax_JAM, 28), alpha=0.5, lw=2, label='Jamaica vaccination rate')
CUBem = ax20.plot(dates, smooth(deaths_CUB, 28), alpha=0.5, lw=2, label='Cuba excess mortality')
JAMem = ax20.plot(dates, smooth(deaths_JAM, 28), alpha=0.5, lw=2, label='Jamaica excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CubaJamaicaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Cuba and Jamaica')
fig19.savefig("CubaJamaicaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
CUBv = ax19.plot(dates, smooth(vax_CUB, 28), alpha=0.5, lw=2, label='Cuba vaccination rate')
VCTv = ax19.plot(dates, smooth(vax_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines vaccination rate')
CUBem = ax20.plot(dates, smooth(deaths_CUB, 28), alpha=0.5, lw=2, label='Cuba excess mortality')
VCTem = ax20.plot(dates, smooth(deaths_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("CubaSaintVincentandtheGrenadinesEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Cuba and Saint Vincent and the Grenadines')
fig19.savefig("CubaSaintVincentandtheGrenadinesEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
DNKv = ax19.plot(dates, smooth(vax_DNK, 28), alpha=0.5, lw=2, label='Denmark vaccination rate')
FINv = ax19.plot(dates, smooth(vax_FIN, 28), alpha=0.5, lw=2, label='Finland vaccination rate')
DNKem = ax20.plot(dates, smooth(deaths_DNK, 28), alpha=0.5, lw=2, label='Denmark excess mortality')
FINem = ax20.plot(dates, smooth(deaths_FIN, 28), alpha=0.5, lw=2, label='Finland excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("DenmarkFinlandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Denmark and Finland')
fig19.savefig("DenmarkFinlandEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
DNKv = ax19.plot(dates, smooth(vax_DNK, 28), alpha=0.5, lw=2, label='Denmark vaccination rate')
DEUv = ax19.plot(dates, smooth(vax_DEU, 28), alpha=0.5, lw=2, label='Germany vaccination rate')
DNKem = ax20.plot(dates, smooth(deaths_DNK, 28), alpha=0.5, lw=2, label='Denmark excess mortality')
DEUem = ax20.plot(dates, smooth(deaths_DEU, 28), alpha=0.5, lw=2, label='Germany excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("DenmarkGermanyEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Denmark and Germany')
fig19.savefig("DenmarkGermanyEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
DNKv = ax19.plot(dates, smooth(vax_DNK, 28), alpha=0.5, lw=2, label='Denmark vaccination rate')
NORv = ax19.plot(dates, smooth(vax_NOR, 28), alpha=0.5, lw=2, label='Norway vaccination rate')
DNKcd = ax20.plot(dates, smooth(deaths_DNK, 28), alpha=0.5, lw=2, label='Denmark COVID-19 death rate')
NORcd = ax20.plot(dates, smooth(deaths_NOR, 28), alpha=0.5, lw=2, label='Norway COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("DenmarkNorwayCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Denmark and Norway')
fig19.savefig("DenmarkNorwayCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
DNKv = ax19.plot(dates, smooth(vax_DNK, 28), alpha=0.5, lw=2, label='Denmark vaccination rate')
NORv = ax19.plot(dates, smooth(vax_NOR, 28), alpha=0.5, lw=2, label='Norway vaccination rate')
DNKem = ax20.plot(dates, smooth(deaths_DNK, 28), alpha=0.5, lw=2, label='Denmark excess mortality')
NORem = ax20.plot(dates, smooth(deaths_NOR, 28), alpha=0.5, lw=2, label='Norway excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("DenmarkNorwayEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Denmark and Norway')
fig19.savefig("DenmarkNorwayEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
DOMv = ax19.plot(dates, smooth(vax_DOM, 28), alpha=0.5, lw=2, label='Dominican Republic vaccination rate')
HTIv = ax19.plot(dates, smooth(vax_HTI, 28), alpha=0.5, lw=2, label='Haiti vaccination rate')
DOMcd = ax20.plot(dates, smooth(deaths_DOM, 28), alpha=0.5, lw=2, label='Dominican Republic COVID-19 death rate')
HTIcd = ax20.plot(dates, smooth(deaths_HTI, 28), alpha=0.5, lw=2, label='Haiti COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("DominicanRepublicHaitiCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Dominican Republic and Haiti')
fig19.savefig("DominicanRepublicHaitiCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
FROv = ax19.plot(dates, smooth(vax_FRO, 28), alpha=0.5, lw=2, label='Faroe Islands vaccination rate')
NORv = ax19.plot(dates, smooth(vax_NOR, 28), alpha=0.5, lw=2, label='Norway vaccination rate')
FROcd = ax20.plot(dates, smooth(deaths_FRO, 28), alpha=0.5, lw=2, label='Faroe Islands COVID-19 death rate')
NORcd = ax20.plot(dates, smooth(deaths_NOR, 28), alpha=0.5, lw=2, label='Norway COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("FaroeIslandsNorwayCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Faroe Islands and Norway')
fig19.savefig("FaroeIslandsNorwayCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
FROv = ax19.plot(dates, smooth(vax_FRO, 28), alpha=0.5, lw=2, label='Faroe Islands vaccination rate')
NORv = ax19.plot(dates, smooth(vax_NOR, 28), alpha=0.5, lw=2, label='Norway vaccination rate')
FROem = ax20.plot(dates, smooth(deaths_FRO, 28), alpha=0.5, lw=2, label='Faroe Islands excess mortality')
NORem = ax20.plot(dates, smooth(deaths_NOR, 28), alpha=0.5, lw=2, label='Norway excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("FaroeIslandsNorwayEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Faroe Islands and Norway')
fig19.savefig("FaroeIslandsNorwayEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
PYFv = ax19.plot(dates, smooth(vax_PYF, 28), alpha=0.5, lw=2, label='French Polynesia vaccination rate')
NZLv = ax19.plot(dates, smooth(vax_NZL, 28), alpha=0.5, lw=2, label='New Zealand vaccination rate')
PYFem = ax20.plot(dates, smooth(deaths_PYF, 28), alpha=0.5, lw=2, label='French Polynesia excess mortality')
NZLem = ax20.plot(dates, smooth(deaths_NZL, 28), alpha=0.5, lw=2, label='New Zealand excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("FrenchPolynesiaNewZealandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for French Polynesia and New Zealand')
fig19.savefig("FrenchPolynesiaNewZealandEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
GIBv = ax19.plot(dates, smooth(vax_GIB, 28), alpha=0.5, lw=2, label='Gibraltar vaccination rate')
INDv = ax19.plot(dates, smooth(vax_IND, 28), alpha=0.5, lw=2, label='India vaccination rate')
GIBcd = ax20.plot(dates, smooth(deaths_GIB, 28), alpha=0.5, lw=2, label='Gibraltar COVID-19 death rate')
INDcd = ax20.plot(dates, smooth(deaths_IND, 28), alpha=0.5, lw=2, label='India COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("GibraltarIndiaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Gibraltar and India')
fig19.savefig("GibraltarIndiaCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HKGv = ax19.plot(dates, smooth(vax_HKG, 28), alpha=0.5, lw=2, label='Hong Kong vaccination rate')
MACv = ax19.plot(dates, smooth(vax_MAC, 28), alpha=0.5, lw=2, label='Macao vaccination rate')
HKGem = ax20.plot(dates, smooth(deaths_HKG, 28), alpha=0.5, lw=2, label='Hong Kong excess mortality')
MACem = ax20.plot(dates, smooth(deaths_MAC, 28), alpha=0.5, lw=2, label='Macao excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("HongKongMacaoEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Hong Kong and Macao')
fig19.savefig("HongKongMacaoEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HKGv = ax19.plot(dates, smooth(vax_HKG, 28), alpha=0.5, lw=2, label='Hong Kong vaccination rate')
PHLv = ax19.plot(dates, smooth(vax_PHL, 28), alpha=0.5, lw=2, label='Philippines vaccination rate')
HKGcd = ax20.plot(dates, smooth(deaths_HKG, 28), alpha=0.5, lw=2, label='Hong Kong COVID-19 death rate')
PHLcd = ax20.plot(dates, smooth(deaths_PHL, 28), alpha=0.5, lw=2, label='Philippines COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("HongKongPhilippinesCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Hong Kong and Philippines')
fig19.savefig("HongKongPhilippinesCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HKGv = ax19.plot(dates, smooth(vax_HKG, 28), alpha=0.5, lw=2, label='Hong Kong vaccination rate')
PHLv = ax19.plot(dates, smooth(vax_PHL, 28), alpha=0.5, lw=2, label='Philippines vaccination rate')
HKGem = ax20.plot(dates, smooth(deaths_HKG, 28), alpha=0.5, lw=2, label='Hong Kong excess mortality')
PHLem = ax20.plot(dates, smooth(deaths_PHL, 28), alpha=0.5, lw=2, label='Philippines excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("HongKongPhilippinesEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Hong Kong and Philippines')
fig19.savefig("HongKongPhilippinesEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HKGv = ax19.plot(dates, smooth(vax_HKG, 28), alpha=0.5, lw=2, label='Hong Kong vaccination rate')
KORv = ax19.plot(dates, smooth(vax_KOR, 28), alpha=0.5, lw=2, label='South Korea vaccination rate')
HKGem = ax20.plot(dates, smooth(deaths_HKG, 28), alpha=0.5, lw=2, label='Hong Kong excess mortality')
KORem = ax20.plot(dates, smooth(deaths_KOR, 28), alpha=0.5, lw=2, label='South Korea excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("HongKongSouthKoreaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Hong Kong and South Korea')
fig19.savefig("HongKongSouthKoreaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
ROUv = ax19.plot(dates, smooth(vax_ROU, 28), alpha=0.5, lw=2, label='Romania vaccination rate')
HUNcd = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary COVID-19 death rate')
ROUcd = ax20.plot(dates, smooth(deaths_ROU, 28), alpha=0.5, lw=2, label='Romania COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("HungaryRomaniaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Hungary and Romania')
fig19.savefig("HungaryRomaniaCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
HUNv = ax19.plot(dates, smooth(vax_HUN, 28), alpha=0.5, lw=2, label='Hungary vaccination rate')
ROUv = ax19.plot(dates, smooth(vax_ROU, 28), alpha=0.5, lw=2, label='Romania vaccination rate')
HUNem = ax20.plot(dates, smooth(deaths_HUN, 28), alpha=0.5, lw=2, label='Hungary excess mortality')
ROUem = ax20.plot(dates, smooth(deaths_ROU, 28), alpha=0.5, lw=2, label='Romania excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("HungaryRomaniaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Hungary and Romania')
fig19.savefig("HungaryRomaniaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
IDNv = ax19.plot(dates, smooth(vax_IDN, 28), alpha=0.5, lw=2, label='Indonesia vaccination rate')
PHLv = ax19.plot(dates, smooth(vax_PHL, 28), alpha=0.5, lw=2, label='Philippines vaccination rate')
IDNcd = ax20.plot(dates, smooth(deaths_IDN, 28), alpha=0.5, lw=2, label='Indonesia COVID-19 death rate')
PHLcd = ax20.plot(dates, smooth(deaths_PHL, 28), alpha=0.5, lw=2, label='Philippines COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("IndonesiaPhilippinesCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Indonesia and Philippines')
fig19.savefig("IndonesiaPhilippinesCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
IRQv = ax19.plot(dates, smooth(vax_IRQ, 28), alpha=0.5, lw=2, label='Iraq vaccination rate')
OMNv = ax19.plot(dates, smooth(vax_OMN, 28), alpha=0.5, lw=2, label='Oman vaccination rate')
IRQcd = ax20.plot(dates, smooth(deaths_IRQ, 28), alpha=0.5, lw=2, label='Iraq COVID-19 death rate')
OMNcd = ax20.plot(dates, smooth(deaths_OMN, 28), alpha=0.5, lw=2, label='Oman COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("IraqOmanCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Iraq and Oman')
fig19.savefig("IraqOmanCD.png")
plt.close('all')

JAMPRTdata = genfromtxt('JamaicaSaintVincentandtheGrenadinesCD.txt', delimiter='\t', converters = {0: date_parser})
dates = [row[0] for row in JAMPRTdata]
vax_JAM = [row[1] for row in JAMPRTdata]
vax_PRT = [row[2] for row in JAMPRTdata]
deaths_JAM = [row[3] for row in JAMPRTdata]
deaths_PRT = [row[4] for row in JAMPRTdata]
fig19 = plt.figure(facecolor='w')
ax19 = fig19.add_subplot(111, axisbelow=True)
ax20 = ax19.twinx()
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
JAMv = ax19.plot(dates, smooth(vax_JAM, 28), alpha=0.5, lw=2, label='Jamaica vaccination rate')
PRTv = ax19.plot(dates, smooth(vax_PRT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines vaccination rate')
JAMcd = ax20.plot(dates, smooth(deaths_JAM, 28), alpha=0.5, lw=2, label='Jamaica COVID-19 death rate')
PRTcd = ax20.plot(dates, smooth(deaths_PRT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("JamaicaSaintVincentandtheGrenadinesCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths\nfor Antigua and Barbuda and Saint Vincent and the Grenadines')
fig19.savefig("JamaicaSaintVincentandtheGrenadinesCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
JAMv = ax19.plot(dates, smooth(vax_JAM, 28), alpha=0.5, lw=2, label='Jamaica vaccination rate')
VCTv = ax19.plot(dates, smooth(vax_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines vaccination rate')
JAMem = ax20.plot(dates, smooth(deaths_JAM, 28), alpha=0.5, lw=2, label='Jamaica excess mortality')
VCTem = ax20.plot(dates, smooth(deaths_VCT, 28), alpha=0.5, lw=2, label='Saint Vincent and the Grenadines excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("JamaicaSaintVincentandtheGrenadinesEM.pdf")
ax19.set_title('Vaccinations and excess mortality\nfor Antigua and Barbuda and Saint Vincent and the Grenadines')
fig19.savefig("JamaicaSaintVincentandtheGrenadinesEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
JPNv = ax19.plot(dates, smooth(vax_JPN, 28), alpha=0.5, lw=2, label='Japan vaccination rate')
MYSv = ax19.plot(dates, smooth(vax_MYS, 28), alpha=0.5, lw=2, label='Malaysia vaccination rate')
JPNem = ax20.plot(dates, smooth(deaths_JPN, 28), alpha=0.5, lw=2, label='Japan excess mortality')
MYSem = ax20.plot(dates, smooth(deaths_MYS, 28), alpha=0.5, lw=2, label='Malaysia excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("JapanMalaysiaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Japan and Malaysia')
fig19.savefig("JapanMalaysiaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
JPNv = ax19.plot(dates, smooth(vax_JPN, 28), alpha=0.5, lw=2, label='Japan vaccination rate')
KORv = ax19.plot(dates, smooth(vax_KOR, 28), alpha=0.5, lw=2, label='South Korea vaccination rate')
JPNem = ax20.plot(dates, smooth(deaths_JPN, 28), alpha=0.5, lw=2, label='Japan excess mortality')
KORem = ax20.plot(dates, smooth(deaths_KOR, 28), alpha=0.5, lw=2, label='South Korea excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("JapanSouthKoreaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Japan and South Korea')
fig19.savefig("JapanSouthKoreaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
KGZv = ax19.plot(dates, smooth(vax_KGZ, 28), alpha=0.5, lw=2, label='Kyrgyzstan vaccination rate')
UZBv = ax19.plot(dates, smooth(vax_UZB, 28), alpha=0.5, lw=2, label='Uzbekistan vaccination rate')
KGZcd = ax20.plot(dates, smooth(deaths_KGZ, 28), alpha=0.5, lw=2, label='Kyrgyzstan COVID-19 death rate')
UZBcd = ax20.plot(dates, smooth(deaths_UZB, 28), alpha=0.5, lw=2, label='Uzbekistan COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("KyrgyzstanUzbekistanCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Kyrgyzstan and Uzbekistan')
fig19.savefig("KyrgyzstanUzbekistanCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
KGZv = ax19.plot(dates, smooth(vax_KGZ, 28), alpha=0.5, lw=2, label='Kyrgyzstan vaccination rate')
UZBv = ax19.plot(dates, smooth(vax_UZB, 28), alpha=0.5, lw=2, label='Uzbekistan vaccination rate')
KGZem = ax20.plot(dates, smooth(deaths_KGZ, 28), alpha=0.5, lw=2, label='Kyrgyzstan excess mortality')
UZBem = ax20.plot(dates, smooth(deaths_UZB, 28), alpha=0.5, lw=2, label='Uzbekistan excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("KyrgyzstanUzbekistanEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Kyrgyzstan and Uzbekistan')
fig19.savefig("KyrgyzstanUzbekistanEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
LVAv = ax19.plot(dates, smooth(vax_LVA, 28), alpha=0.5, lw=2, label='Latvia vaccination rate')
UKRv = ax19.plot(dates, smooth(vax_UKR, 28), alpha=0.5, lw=2, label='Ukraine vaccination rate')
LVAcd = ax20.plot(dates, smooth(deaths_LVA, 28), alpha=0.5, lw=2, label='Latvia COVID-19 death rate')
UKRcd = ax20.plot(dates, smooth(deaths_UKR, 28), alpha=0.5, lw=2, label='Ukraine COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('COVID-19 death rate')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("LatviaUkraineCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Latvia and Ukraine')
fig19.savefig("LatviaUkraineCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
LVAv = ax19.plot(dates, smooth(vax_LVA, 28), alpha=0.5, lw=2, label='Latvia vaccination rate')
UKRv = ax19.plot(dates, smooth(vax_UKR, 28), alpha=0.5, lw=2, label='Ukraine vaccination rate')
LVAem = ax20.plot(dates, smooth(deaths_LVA, 28), alpha=0.5, lw=2, label='Latvia excess mortality')
UKRem = ax20.plot(dates, smooth(deaths_UKR, 28), alpha=0.5, lw=2, label='Ukraine excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("LatviaUkraineEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Latvia and Ukraine')
fig19.savefig("LatviaUkraineEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
LBNv = ax19.plot(dates, smooth(vax_LBN, 28), alpha=0.5, lw=2, label='Lebanon vaccination rate')
PSEv = ax19.plot(dates, smooth(vax_PSE, 28), alpha=0.5, lw=2, label='Palestine vaccination rate')
LBNem = ax20.plot(dates, smooth(deaths_LBN, 28), alpha=0.5, lw=2, label='Lebanon excess mortality')
PSEem = ax20.plot(dates, smooth(deaths_PSE, 28), alpha=0.5, lw=2, label='Palestine excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("LebanonPalestineEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Lebanon and Palestine')
fig19.savefig("LebanonPalestineEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
MYSv = ax19.plot(dates, smooth(vax_MYS, 28), alpha=0.5, lw=2, label='Malaysia vaccination rate')
MNGv = ax19.plot(dates, smooth(vax_MNG, 28), alpha=0.5, lw=2, label='Mongolia vaccination rate')
MYSem = ax20.plot(dates, smooth(deaths_MYS, 28), alpha=0.5, lw=2, label='Malaysia excess mortality')
MNGem = ax20.plot(dates, smooth(deaths_MNG, 28), alpha=0.5, lw=2, label='Mongolia excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("MalaysiaMongoliaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Malaysia and Mongolia')
fig19.savefig("MalaysiaMongoliaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
MYSv = ax19.plot(dates, smooth(vax_MYS, 28), alpha=0.5, lw=2, label='Malaysia vaccination rate')
TWNv = ax19.plot(dates, smooth(vax_TWN, 28), alpha=0.5, lw=2, label='Taiwan vaccination rate')
MYSem = ax20.plot(dates, smooth(deaths_MYS, 28), alpha=0.5, lw=2, label='Malaysia excess mortality')
TWNem = ax20.plot(dates, smooth(deaths_TWN, 28), alpha=0.5, lw=2, label='Taiwan excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("MalaysiaTaiwanEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Malaysia and Taiwan')
fig19.savefig("MalaysiaTaiwanEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
MYSv = ax19.plot(dates, smooth(vax_MYS, 28), alpha=0.5, lw=2, label='Malaysia vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
MYScd = ax20.plot(dates, smooth(deaths_MYS, 28), alpha=0.5, lw=2, label='Malaysia COVID-19 death rate')
THAcd = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("MalaysiaThailandCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Malaysia and Thailand')
fig19.savefig("MalaysiaThailandCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
MYSv = ax19.plot(dates, smooth(vax_MYS, 28), alpha=0.5, lw=2, label='Malaysia vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
MYSem = ax20.plot(dates, smooth(deaths_MYS, 28), alpha=0.5, lw=2, label='Malaysia excess mortality')
THAem = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("MalaysiaThailandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Malaysia and Thailand')
fig19.savefig("MalaysiaThailandEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
MUSv = ax19.plot(dates, smooth(vax_MUS, 28), alpha=0.5, lw=2, label='Mauritius vaccination rate')
SYCv = ax19.plot(dates, smooth(vax_SYC, 28), alpha=0.5, lw=2, label='Seychelles vaccination rate')
MUScd = ax20.plot(dates, smooth(deaths_MUS, 28), alpha=0.5, lw=2, label='Mauritius COVID-19 death rate')
SYCcd = ax20.plot(dates, smooth(deaths_SYC, 28), alpha=0.5, lw=2, label='Seychelles COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("MauritiusSeychellesCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Mauritius and Seychelles')
fig19.savefig("MauritiusSeychellesCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
MUSv = ax19.plot(dates, smooth(vax_MUS, 28), alpha=0.5, lw=2, label='Mauritius vaccination rate')
SYCv = ax19.plot(dates, smooth(vax_SYC, 28), alpha=0.5, lw=2, label='Seychelles vaccination rate')
MUSem = ax20.plot(dates, smooth(deaths_MUS, 28), alpha=0.5, lw=2, label='Mauritius excess mortality')
SYCem = ax20.plot(dates, smooth(deaths_SYC, 28), alpha=0.5, lw=2, label='Seychelles excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("MauritiusSeychellesEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Mauritius and Seychelles')
fig19.savefig("MauritiusSeychellesEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
MNGv = ax19.plot(dates, smooth(vax_MNG, 28), alpha=0.5, lw=2, label='Mongolia vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
MNGcd = ax20.plot(dates, smooth(deaths_MNG, 28), alpha=0.5, lw=2, label='Mongolia COVID-19 death rate')
THAcd = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("MongoliaThailandCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Mongolia and Thailand')
fig19.savefig("MongoliaThailandCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
MNGv = ax19.plot(dates, smooth(vax_MNG, 28), alpha=0.5, lw=2, label='Mongolia vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
MNGem = ax20.plot(dates, smooth(deaths_MNG, 28), alpha=0.5, lw=2, label='Mongolia excess mortality')
THAem = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("MongoliaThailandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Mongolia and Thailand')
fig19.savefig("MongoliaThailandEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
PNGv = ax19.plot(dates, smooth(vax_PNG, 28), alpha=0.5, lw=2, label='Papua New Guinea vaccination rate')
SLBv = ax19.plot(dates, smooth(vax_SLB, 28), alpha=0.5, lw=2, label='Solomon Islands vaccination rate')
PNGcd = ax20.plot(dates, smooth(deaths_PNG, 28), alpha=0.5, lw=2, label='Papua New Guinea COVID-19 death rate')
SLBcd = ax20.plot(dates, smooth(deaths_SLB, 28), alpha=0.5, lw=2, label='Solomon Islands COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("PapuaNewGuineaSolomonIslandsCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for PapuaNewGuinea and SolomonIslands')
fig19.savefig("PapuaNewGuineaSolomonIslandsCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
PHLv = ax19.plot(dates, smooth(vax_PHL, 28), alpha=0.5, lw=2, label='Philippines vaccination rate')
KORv = ax19.plot(dates, smooth(vax_KOR, 28), alpha=0.5, lw=2, label='South Korea vaccination rate')
PHLem = ax20.plot(dates, smooth(deaths_PHL, 28), alpha=0.5, lw=2, label='Philippines excess mortality')
KORem = ax20.plot(dates, smooth(deaths_KOR, 28), alpha=0.5, lw=2, label='South Korea excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("PhilippinesSouthKoreaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Philippines and South Korea')
fig19.savefig("PhilippinesSouthKoreaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
ROUv = ax19.plot(dates, smooth(vax_ROU, 28), alpha=0.5, lw=2, label='Romania vaccination rate')
RUSv = ax19.plot(dates, smooth(vax_RUS, 28), alpha=0.5, lw=2, label='Russia vaccination rate')
ROUcd = ax20.plot(dates, smooth(deaths_ROU, 28), alpha=0.5, lw=2, label='Romania COVID-19 death rate')
RUScd = ax20.plot(dates, smooth(deaths_RUS, 28), alpha=0.5, lw=2, label='Russia COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("RomaniaRussiaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Romania and Russia')
fig19.savefig("RomaniaRussiaCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
ROUv = ax19.plot(dates, smooth(vax_ROU, 28), alpha=0.5, lw=2, label='Romania vaccination rate')
RUSv = ax19.plot(dates, smooth(vax_RUS, 28), alpha=0.5, lw=2, label='Russia vaccination rate')
ROUem = ax20.plot(dates, smooth(deaths_ROU, 28), alpha=0.5, lw=2, label='Romania excess mortality')
RUSem = ax20.plot(dates, smooth(deaths_RUS, 28), alpha=0.5, lw=2, label='Russia excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("RomaniaRussiaEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Romania and Russia')
fig19.savefig("RomaniaRussiaEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
RWAv = ax19.plot(dates, smooth(vax_RWA, 28), alpha=0.5, lw=2, label='Rwanda vaccination rate')
UGAv = ax19.plot(dates, smooth(vax_UGA, 28), alpha=0.5, lw=2, label='Uganda vaccination rate')
RWAcd = ax20.plot(dates, smooth(deaths_RWA, 28), alpha=0.5, lw=2, label='Rwanda COVID-19 death rate')
UGAcd = ax20.plot(dates, smooth(deaths_UGA, 28), alpha=0.5, lw=2, label='Uganda COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("RwandaUgandaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Rwanda and Uganda')
fig19.savefig("RwandaUgandaCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
SYCv = ax19.plot(dates, smooth(vax_SYC, 28), alpha=0.5, lw=2, label='Seychelles vaccination rate')
TZAv = ax19.plot(dates, smooth(vax_TZA, 28), alpha=0.5, lw=2, label='Tanzania vaccination rate')
SYCcd = ax20.plot(dates, smooth(deaths_SYC, 28), alpha=0.5, lw=2, label='Seychelles COVID-19 death rate')
TZAcd = ax20.plot(dates, smooth(deaths_TZA, 28), alpha=0.5, lw=2, label='Tanzania COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("SeychellesTanzaniaCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Seychelles and Tanzania')
fig19.savefig("SeychellesTanzaniaCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
SGPv = ax19.plot(dates, smooth(vax_SGP, 28), alpha=0.5, lw=2, label='Singapore vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
SGPcd = ax20.plot(dates, smooth(deaths_SGP, 28), alpha=0.5, lw=2, label='Singapore COVID-19 death rate')
THAcd = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("SingaporeThailandCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Singapore and Thailand')
fig19.savefig("SingaporeThailandCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
SGPv = ax19.plot(dates, smooth(vax_SGP, 28), alpha=0.5, lw=2, label='Singapore vaccination rate')
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
SGPem = ax20.plot(dates, smooth(deaths_SGP, 28), alpha=0.5, lw=2, label='Singapore excess mortality')
THAem = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand excess mortality')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Excess mortality P-scores')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("SingaporeThailandEM.pdf")
ax19.set_title('Vaccinations and excess mortality for Singapore and Thailand')
fig19.savefig("SingaporeThailandEM.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
THAv = ax19.plot(dates, smooth(vax_THA, 28), alpha=0.5, lw=2, label='Thailand vaccination rate')
VNMv = ax19.plot(dates, smooth(vax_VNM, 28), alpha=0.5, lw=2, label='Vietnam vaccination rate')
THAcd = ax20.plot(dates, smooth(deaths_THA, 28), alpha=0.5, lw=2, label='Thailand COVID-19 death rate')
VNMcd = ax20.plot(dates, smooth(deaths_VNM, 28), alpha=0.5, lw=2, label='Vietnam COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("ThailandVietnamCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Thailand and Vietnam')
fig19.savefig("ThailandVietnamCD.png")
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
ax20._get_lines.prop_cycler = ax19._get_lines.prop_cycler
ZMBv = ax19.plot(dates, smooth(vax_ZMB, 28), alpha=0.5, lw=2, label='Zambia vaccination rate')
ZWEv = ax19.plot(dates, smooth(vax_ZWE, 28), alpha=0.5, lw=2, label='Zimbabwe vaccination rate')
ZMBcd = ax20.plot(dates, smooth(deaths_ZMB, 28), alpha=0.5, lw=2, label='Zambia COVID-19 death rate')
ZWEcd = ax20.plot(dates, smooth(deaths_ZWE, 28), alpha=0.5, lw=2, label='Zimbabwe COVID-19 death rate')
ax19.set_ylabel('Daily vaccination doses per million')
ax20.set_ylabel('Daily COVID-19 deaths per million')
ax19.yaxis.set_tick_params(length=0)
ax19.xaxis.set_tick_params(length=0)
ax19.xaxis.set_major_formatter(mdates.DateFormatter('%#d %b %Y'))
ax19.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax19.fmt_xdata = mdates.DateFormatter('%#d %b %Y')
fig19.autofmt_xdate()
ax19.grid(visible=True)
legend.get_frame().set_alpha(1)
lines, labels = ax19.get_legend_handles_labels()
lines2, labels2 = ax20.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, fancybox=True, framealpha=0.8)
fig19.savefig("ZambiaZimbabweCD.pdf")
ax19.set_title('Vaccinations and COVID-19 deaths for Zambia and Zimbabwe')
fig19.savefig("ZambiaZimbabweCD.png")
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

f.flush()
os.fsync(f.fileno())
f.close()
print("Finished!")