############## SHOCK ADAPTATION FOOD SUPPLY MODEL #############
#
# ADAPTED BY CSM_SEXY_GRP_ - 2025, ORIGIN: SOPHIA BAUM - 2024

### IMPORTs ###
import pandas as pd
from pandas import IndexSlice as idx
import numpy as np
import os
from pathlib import Path


### PARAMETERS ###
###SCENSRIOS : PAK RUS HOA URU ALL
scenario = 'PAK'
production_cap = True  # Add this parameter to control cap behavior

#working directory
script_dir = Path(__file__).parent.absolute()


input_folder = script_dir / 'input'   
output_folder = script_dir / 'results'
losses_folder = script_dir /  'evaluation'



### LOADING DATA ###
def load_data(scenario, production_cap):
    """Load data with appropriate filename based on production_cap"""
    base_file = output_folder / f"{scenario}_base_.csv"
    scenario_file = output_folder / f"{scenario}{'_capped' if production_cap else '_no_cap'}.csv"
    
    X_base = pd.read_csv(base_file, index_col=[0, 1], header=[0])
    XS_comp = pd.read_csv(scenario_file, index_col=[0, 1], header=[0])
    
    return X_base, XS_comp

X_base, XS_comp = load_data(scenario, production_cap)
a_frame = pd.read_csv(input_folder / 'a_frame.csv')

### GENERAL LOSS CALCULATIONS ###
def calculate_losses(X_base, XS_comp):
    """Calculate absolute and relative losses"""
    x_i = XS_comp['amount [t]']
    
    XS_comp['absolute_losses [t]'] = (X_base['base'] - x_i).fillna(0)
    XS_comp['relative_losses'] = (1 - x_i / X_base['base']).fillna(0).clip(lower=-1)
    
    return XS_comp

XS_comp = calculate_losses(X_base, XS_comp)

### PER CAPITA EVALUATION ###
def add_per_capita(XS_comp, a_frame):
    """Add per capita calculations"""
    populations = dict(zip(a_frame['area'], a_frame['population']))
    
    # Calculate per capita losses (convert tons to kg)
    XS_comp['population'] = XS_comp.index.get_level_values(0).map(populations)
    XS_comp['al/capita [kg]'] = 1000 * XS_comp['absolute_losses [t]'] / XS_comp['population']
    
    return XS_comp

XS_comp = add_per_capita(XS_comp, a_frame)

### SAVE RESULTS WITH PRODUCTION CAP INDICATOR ###
def save_results(XS_comp, scenario, production_cap, losses_folder):
    """Save results with appropriate filenames"""
    cap_suffix = "_capped" if production_cap else "_no_cap"
    
    # Save complete losses - using pathlib's / operator
    complete_path = losses_folder / f"{scenario}-Losses{cap_suffix}.csv"
    XS_comp.to_csv(complete_path)
    print(f"Saved complete losses to: {complete_path}")
    
    # Save highest losses (>100 kg/capita)
    XS_highest_losses = XS_comp[XS_comp['al/capita [kg]'] > 50]
    if not XS_highest_losses.empty:
        highest_losses_path = losses_folder / f"{scenario}-highestLosses{cap_suffix}.csv"
        XS_highest_losses.sort_values('al/capita [kg]', ascending=False).to_csv(highest_losses_path)
        print(f"Saved highest losses to: {highest_losses_path}")
    
    # Save highest profits (<-100 kg/capita)
    XS_highest_profits = XS_comp[XS_comp['al/capita [kg]'] < -50]
    if not XS_highest_profits.empty:
        highest_profits_path = losses_folder / f"{scenario}-highestProfits{cap_suffix}.csv"
        XS_highest_profits.sort_values('al/capita [kg]', ascending=False).to_csv(highest_profits_path)
        print(f"Saved highest profits to: {highest_profits_path}")

save_results(XS_comp, scenario, production_cap, losses_folder)

print("Loss calculations completed successfully.")