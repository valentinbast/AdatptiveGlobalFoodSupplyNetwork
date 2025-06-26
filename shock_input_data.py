############## SHOCK ADAPTATION FOOD SUPPLY MODEL #############
#
#
#  ADD ON BY CSM_SEXY_GRP_ -  2025, ORIGIN: SOPHIA BAUM # -  2024


### IMPORTS ###

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pandas import IndexSlice as idx

import scipy.io as io
import scipy.sparse as sprs
import numpy as np


### EMPIRICAL SHOCK INTENSITY DATA ###
 
possible_scenarios = { 
    
    'PAK': {
                ('Pakistan','Rice and products'):                0.2146,
                ('Pakistan','Cottonseed'):                       0.4124,
                ('Pakistan','Rape and Mustardseed'):             0.27,
                ('Pakistan', 'Peas'):                            0.29,
                ('Pakistan', 'Dates'):                           0.7274
            },

    'RUS': {
                ('Russian Federation', 'Wheat and products'):    0.3276,
                ('Russian Federation', 'Barley and products'):   0.5330,
                ('Russian Federation', 'Cereals, Other'):        0.3985,
                ('Russian Federation', 'Maize and products'):    0.2216,
                ('Russian Federation', 'Oats'):                  0.4039,
                ('Russian Federation', 'Peas'):                  0.2344,
                ('Russian Federation', 'Potatoes and products'): 0.3208,
                ('Russian Federation', 'Rye and products'):      0.6225
            },

    'HOA': {
                ('Kenya', 'Sugar cane'):                        36.8610650242051,   # -  2023 worst year 
                ('Ethiopia', 'Sorghum and products'):           21.332861080057995, # -  2021 worst year
                ('Kenya', 'Tea (including mate)'):              20.298360655737703, # -  2021 worst year
                ('Kenya', 'Tomatoes and products'):             32.87920997090733,  # -  2021 worst year
                ('Ethiopia', 'Milk - Excluding Butter'):        17.62077684309846,  # -  2021 worst year
                ('Ethiopia', 'Sugar cane'):                     39.400315180764856, # -  2021 worst year
                ('Ethiopia', 'Sweet potatoes'):                 42.84694759881593,  # -  2021 worst year
                ('Kenya', 'Pineapples and products'):           61.45627225959635,  # -  2023 worst year
                ('Ethiopia', 'Millet and products'):            22.680840873397766, # -  2022 worst year
                ('Kenya', 'Sorghum and products'):              55.904444444444444, # -  2021 worst year
                ('Kenya', 'Eggs'):                              95.9267938846111,   # -  2022 worst year
                ('Kenya', 'Wheat and products'):                16.057673530496675, # -  2023 worst year
                ('Ethiopia', 'Groundnuts'):                     32.081281073435655, # -  2021 worst year
                ('Kenya', 'Millet and products'):               61.3202614379085,   # -  2021 worst year
                ('Ethiopia', 'Rice and products'):              25.881845684577364, # -  2022 worst year
                ('Ethiopia', 'Sesame seed'):                    47.04977525538828,  # -  2021 worst year
                ('Ethiopia', 'Honey'):                          59.7574651394807,   # -  2021 worst year
                ('Ethiopia', 'Onions'):                         39.990983909746625, # -  2021 worst year
                ('Kenya', 'Lemons, Limes and products'):        30.93136991085126,  # -  2021 worst year
                ('Kenya', 'Onions'):                            16.75193564238785,  # -  2021 worst year
                ('Kenya', 'Coconuts - Incl Copra'):             21.323843545762777, # -  2021 worst year
                ('Djibouti', 'Beans'):                          25.0,               # -  2021 worst year
                ('Somalia', 'Bovine Meat'):                      2.1653381604079702 # -  2023 worst year
            },
    
    'URU': {
                ('Uruguay', 'Soyabeans'):                        0.7667, # -  2022 worst year
                ('Uruguay', 'Maize and products'):               0.0442, # -  2021 worst year
                ('Uruguay', 'Milk - Excluding Butter'):          0.0478, # -  2022 worst year
                ('Uruguay', 'Sorghum and products'):             0.7157, # -  2022 worst year
                ('Uruguay', 'Lemons, Limes and products'):       0.2746, # -  2022 worst year
                ('Uruguay', 'Rice and products'):                0.0180, # -  2023 worst year
                ('Uruguay', 'Oranges, Mandarines'):              0.0728  # -  2023 worst year
    }
}

# !Important! - noramlise and round values for HOA scenario
possible_scenarios['HOA'] = {k: np.round(v*0.01, 4) for k, v in possible_scenarios['HOA'].items()} 

#shock intensity curve constuction
mu = - 0.5                                              # Assumption
def phi(i_0, mu, t):                                      
    return np.round(i_0 * np.exp(mu * t), 2)            # Assumed exponential decay of shock intensity over time

def create_shock_scaling_marix(scenario, tau):
    if scenario not in ['PAK', 'RUS', 'HOA', 'URU', 'ALL']:
        raise ValueError("Please choose a valid scenrio.")
    
                        # Extract the information needed.
    elif scenario == 'ALL':
        shock_sectors = []
        phi_0 = []

        for key in possible_scenarios.keys():
            sectors = possible_scenarios[key]
            for k, v in sectors.items():
                shock_sectors.append(k) 
                phi_0.append(v)

    else:
        sectors = possible_scenarios[scenario]
        shock_sectors = list(sectors.keys())
        phi_0 = list(sectors.values())


                            # Construct shock scaling matrix
    shock_scaling = np.zeros((len(shock_sectors), tau )) #[1 - phi(t) for t in range(tau)] # Create values to scale production-output
    for row_index, row in enumerate(shock_scaling):
        shock_scaling[row_index, : ] = [1 - phi(phi_0[row_index], mu, t) for t in range(tau)]
        
    return shock_sectors, shock_scaling
