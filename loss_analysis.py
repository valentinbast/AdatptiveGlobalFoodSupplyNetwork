import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# Configuration
scenarios = ["ALL", "HOA", "RUS", "URU", "PAK"]
output_dir = Path("evaluation/plots")
output_dir.mkdir(parents=True, exist_ok=True)

def load_all_data():
    """Load data for both capped and uncapped scenarios"""
    data = {
        'capped': {'losses': {}, 'profits': {}},
        'uncapped': {'losses': {}, 'profits': {}}
    }
    
    for cap_status in ['capped', 'no_cap']:
        for s in scenarios:
            # Load losses
            loss_file = Path(f"evaluation/{s}-highestLosses_{cap_status}.csv")
            if loss_file.exists():
                loss_df = pd.read_csv(loss_file)
                if 'absolute_losses [t]' in loss_df.columns:
                    key = 'capped' if cap_status == 'capped' else 'uncapped'
                    data[key]['losses'][s] = loss_df
            
            # Load profits 
            profit_file = Path(f"evaluation/{s}-highestProfits_{cap_status}.csv")
            if profit_file.exists():
                profit_df = pd.read_csv(profit_file)
                if 'absolute_losses [t]' in profit_df.columns:
                    key = 'capped' if cap_status == 'capped' else 'uncapped'
                    data[key]['profits'][s] = profit_df
    
    return data

def create_single_plot(data, plot_type, cap_status):
    """Create and save individual plot"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if data:
        values = {s: df['absolute_losses [t]'].sum() for s, df in data.items()}
        if plot_type == 'profits':
            values = {s: df['absolute_losses [t]'].abs().sum() for s, df in data.items()}
        
        group_total = sum(v for k, v in values.items() if k != "ALL")
        values["GROUP"] = group_total
        
        color = 'red' if plot_type == 'losses' else 'green'
        sns.barplot(x=list(values.keys()), y=list(values.values()), color=color, ax=ax)
        
        ax.set_title(f"{plot_type.capitalize()} (Cap: {'ON' if cap_status == 'capped' else 'OFF'})")
        ax.set_ylabel("Tonnes")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir/f"{plot_type}_{cap_status}.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_comparison_plot(loss_data, profit_data, cap_status):
    """Create and save comparison plot for one cap status"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Losses plot
    if loss_data:
        losses = {s: df['absolute_losses [t]'].sum() for s, df in loss_data.items()}
        group_loss = sum(v for k, v in losses.items() if k != "ALL")
        losses["GROUP"] = group_loss
        
        sns.barplot(x=list(losses.keys()), y=list(losses.values()), color='red', ax=ax1)
        ax1.set_title(f"Losses (Cap: {'ON' if cap_status == 'capped' else 'OFF'})")
        ax1.set_ylabel("Tonnes")
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax1.tick_params(axis='x', rotation=45)
    
    # Profits plot
    if profit_data:
        profits = {s: df['absolute_losses [t]'].abs().sum() for s, df in profit_data.items()}
        group_profit = sum(v for k, v in profits.items() if k != "ALL")
        profits["GROUP"] = group_profit
        
        sns.barplot(x=list(profits.keys()), y=list(profits.values()), color='green', ax=ax2)
        ax2.set_title(f"Profits (Cap: {'ON' if cap_status == 'capped' else 'OFF'})")
        ax2.set_ylabel("Tonnes")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir/f"comparison_{cap_status}.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_all_plots(data):
    """Create and save all required plots"""
    # Create individual plots
    create_single_plot(data['capped']['losses'], 'losses', 'capped')
    create_single_plot(data['capped']['profits'], 'profits', 'capped')
    create_single_plot(data['uncapped']['losses'], 'losses', 'uncapped')
    create_single_plot(data['uncapped']['profits'], 'profits', 'uncapped')
    
    # Create comparison plots
    create_comparison_plot(data['capped']['losses'], data['capped']['profits'], 'capped')
    create_comparison_plot(data['uncapped']['losses'], data['uncapped']['profits'], 'uncapped')

if __name__ == "__main__":
    all_data = load_all_data()
    
    if any(all_data['capped'].values()) or any(all_data['uncapped'].values()):
        create_all_plots(all_data)
        print("Successfully created plots:")
        print("Individual plots:")
        print("- losses_capped.png")
        print("- profits_capped.png") 
        print("- losses_uncapped.png")
        print("- profits_uncapped.png")
        print("\nComparison plots:")
        print("- comparison_capped.png")
        print("- comparison_uncapped.png")
    else:
        print("No data available for visualization")