#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Load and prepare data
production_overshoot = pd.read_csv('results/production_overshoot.csv')


cap_line_df = production_overshoot[production_overshoot['scenario'] == production_overshoot['scenario'].unique()[0]][['time_step', 'cap']].copy()

cap_line_df.dropna(subset=['cap'], inplace=True)
cap_line_df = cap_line_df[cap_line_df['cap'] > 0] # Filter out non-positive values



df_melted = production_overshoot.melt(id_vars=['time_step', 'scenario'], 
                            value_vars=['total_prod'], 
                            var_name='variable', value_name='total_production')



plt.figure(figsize=(12, 7))

# Plot total_prod per scenario
sns.lineplot(
    data=df_melted, 
    x='time_step', 
    y='total_production', 
    hue='scenario', 
    marker='o', 
    alpha=0.8,
    linewidth=2
)

# Plot the cap line
if not cap_line_df.empty:
    plt.plot(
        cap_line_df['time_step'], 
        cap_line_df['cap'], 
        label='Production Cap', 
        linestyle=':', 
        color='black', 
        linewidth=2
    )
else:
    print("WARNING: 'cap' line not plotted. No valid data.")

# Final touches
plt.title('Total Production vs. Cap Over Time by Scenario')
plt.xlabel('Time Step')
plt.ylabel('Units Produced')
plt.ylim(0, df_melted['total_production'].max() * 1.1)
plt.grid(True)

# Place legend outside
plt.legend(title='Scenario', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig("evaluation/plots/production_capacity_production.jpg")
plt.show()
