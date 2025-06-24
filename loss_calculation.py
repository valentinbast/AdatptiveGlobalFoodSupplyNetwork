############## SHOCK ADAPTATION FOOD SUPPLY MODEL #############
#
#
#  ADAPTED BY CSM_SEXY_GRP_ - 2025, ORIGIN: SOPHIA BAUM - 2024


### PARAMETERS ### PAK RUS HOA URU ALL

scenario = 'HOA'

input_folder  = './input/'   
output_folder = './results/'             # Folder with parameters             # Folder to write results to
losses        = './evaluation/'          # Folder to store the refined results to


### IMPORTs ###

import pandas as pd
from pandas import IndexSlice as idx

import scipy.io as io
import scipy.sparse as sprs

import numpy as np


### LOADING DATA ###

X_base  = pd.read_csv(output_folder + scenario + '_base_.csv', index_col = [0, 1], header = [0])
XS_comp = pd.read_csv(output_folder + scenario + '.csv',       index_col = [0, 1], header = [0])

    # Load further information on countries
a_frame  = pd.read_csv(input_folder + 'a_frame.csv')

### GENERAL LOSS CALCULATIONS ###

x_i = XS_comp['amount [t]']                                                         # Extract values to transform

XS_comp['absolute_losses [t]'] = (X_base['base'] - x_i).fillna(0)                       # Absolute loss calculation
XS_comp['relative_losses'] = (1 - x_i / X_base['base']).fillna(0).clip(lower = -1)  # Realative loss calculation and Manipulation


### PER CAPITA EVALUATION ###

    # Create dictionary, keys: countries, values: populations
populations = {}                                                   

countries  = a_frame['area']
population = a_frame['population']

for index, country in enumerate(countries):
    populations[country] = population[index]                       # Construct dictionary

    # Write corresponding populations in evluation sheet - needed for pc_losses
for row in XS_comp.itertuples():                                    # Iterate over all rows
    index = row.Index                                               # Indentify country name
    country = index[0]                                              # Indentify country name
    XS_comp.loc[index, 'al/capita [kg]'] = populations[country]     # Write corresponding population in new cells for pc calc

XS_comp['al/capita [kg]'] = 1000 * XS_comp['absolute_losses [t]'] / XS_comp['al/capita [kg]'] # Calc pc_losses + convert tons in kg

#Save
XS_comp.to_csv(losses + scenario + '-Losses.csv') 


### FIND COUNTRIES WITH HIGHEST CHANGES ###

pc_losses = XS_comp.iloc[:, 3]

XS_highest_losses = XS_comp[pc_losses > 500]                                                       # Only keep Data for sectors that reach treshold of 100kg
XS_highest_losses = XS_highest_losses.sort_values(by = XS_comp.columns[3], ascending = False)    # Sort values
XS_highest_losses.to_csv(losses + scenario + '-highestLosses.csv')                               # Save

XS_highest_profits = XS_comp[pc_losses < -500]                                                     # Only keep Data for sectors that reach tresholdof -1kg
XS_highest_profits = XS_highest_profits.sort_values(by = XS_comp.columns[3], ascending = False)  # Sort values
XS_highest_profits.to_csv(losses + scenario + '-highestProfits.csv')                             # Save
