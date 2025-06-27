import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from matplotlib.ticker import FuncFormatter

# Configuration
production_cap = True  # Change as needed
scenario = ["ALL", "HOA", "RUS", "URU", "PAK"]
output_dir = Path("evaluation/plots")
output_dir.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load data for current production_cap setting"""
    data = {'losses': {}, 'profits': {}}
    cap_status = "capped" if production_cap else "no_cap"
    
    for s in scenario:
        # Load losses
        loss_file = Path(f"evaluation/{s}-highestLosses_{cap_status}.csv")
        if loss_file.exists():
            loss_df = pd.read_csv(loss_file)
            if 'absolute_losses [t]' in loss_df.columns:
                data['losses'][s] = loss_df
        
        # Load profits 
        profit_file = Path(f"evaluation/{s}-highestProfits_{cap_status}.csv")
        if profit_file.exists():
            profit_df = pd.read_csv(profit_file)
            if 'absolute_losses [t]' in profit_df.columns:
                data['profits'][s] = profit_df
    
    return data

def create_plots(data_dict):
    """Create and save loss/profit comparison plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Losses plot
    if data_dict['losses']:
        losses = {s: df['absolute_losses [t]'].sum() for s, df in data_dict['losses'].items()}
        group_loss = sum(v for k, v in losses.items() if k != "ALL")
        losses["GROUP"] = group_loss
        
        sns.barplot(x=list(losses.keys()), y=list(losses.values()), color='red', ax=ax1)
        ax1.set_title(f"Losses (Cap: {'ON' if production_cap else 'OFF'})")
        ax1.set_ylabel("Tonnes")
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax1.tick_params(axis='x', rotation=45)
    
    # Profits plot
    if data_dict['profits']:
        profits = {s: df['absolute_losses [t]'].abs().sum() for s, df in data_dict['profits'].items()}
        group_profit = sum(v for k, v in profits.items() if k != "ALL")
        profits["GROUP"] = group_profit
        
        sns.barplot(x=list(profits.keys()), y=list(profits.values()), color='green', ax=ax2)
        ax2.set_title(f"Profits (Cap: {'ON' if production_cap else 'OFF'})")
        ax2.set_ylabel("Tonnes")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir/f"results_{'capped' if production_cap else 'uncapped'}.png")
    plt.show()

if __name__ == "__main__":
    data = load_data()
    if data['losses'] or data['profits']:
        create_plots(data)
    else:
        print("No data available for visualization")

