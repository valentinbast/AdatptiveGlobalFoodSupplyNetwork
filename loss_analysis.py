import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from matplotlib.ticker import FuncFormatter

# Configuration
production_cap = False  # Change this to True/False as needed
scenario = ["ALL", "HOA", "RUS", "URU", "PAK"]
current_dir = Path.cwd()
loss_calculation_folder = current_dir / "evaluation"
output_dir = loss_calculation_folder / "plots"

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

def verify_files_exist():
    """Check if expected files exist"""
    print("\n Verifying input files:")
    for s in scenario:
        for capped in [True, False]:
            cap_status = "capped" if capped else "no_cap"
            filename = loss_calculation_folder / f"{s}-highestLosses_{cap_status}.csv"
            print(f"{'found' if filename.exists() else 'not found'} {filename.name}")

def load_and_validate_data():
    """Load data with validation for both losses and profits"""
    data = {'losses': {}, 'profits': {}}
    
    for s in scenario:
        cap_status = "capped" if production_cap else "no_cap"
        
        # Load losses data
        loss_file = loss_calculation_folder / f"{s}-highestLosses_{cap_status}.csv"
        if loss_file.exists():
            try:
                loss_df = pd.read_csv(loss_file)
                if 'absolute_losses [t]' in loss_df.columns:
                    print(f" Loaded losses: {loss_file.name} ({len(loss_df)} records)")
                    data['losses'][s] = loss_df
            except Exception as e:
                print(f" Error loading {loss_file.name}: {str(e)}")
        
        # Load profits data
        profit_file = loss_calculation_folder / f"{s}-highestProfits_{cap_status}.csv"
        if profit_file.exists():
            try:
                profit_df = pd.read_csv(profit_file)
                if 'absolute_losses [t]' in profit_df.columns:
                    print(f" Loaded profits: {profit_file.name} ({len(profit_df)} records)")
                    data['profits'][s] = profit_df
            except Exception as e:
                print(f" Error loading {profit_file.name}: {str(e)}")
    
    return data

def analyze_and_visualize(data_dict):
    """Process and visualize both losses and profits"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    # ===== LOSSES VISUALIZATION =====
    if data_dict['losses']:
        totals_loss = {
            s: df['absolute_losses [t]'].sum() 
            for s, df in data_dict['losses'].items()
        }
        group_total_loss = sum(v for k, v in totals_loss.items() if k != "ALL")
        totals_loss["SUM(PAK+URU+RUS+HOA)"] = group_total_loss
        sorted_loss = dict(sorted(totals_loss.items(), key=lambda x: x[1], reverse=True))
        
        sns.barplot(
            x=list(sorted_loss.keys()),
            y=list(sorted_loss.values()),
            palette="Reds_r",
            edgecolor="black",
            ax=ax1
        )
        
        ax1.set_title(
            f"Total Absolute Losses\n(Production Cap: {'ON' if production_cap else 'OFF'})",
            pad=15
        )
        ax1.set_ylabel("Total Losses [t]")
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax1.tick_params(axis='x', rotation=45)
        
        for p in ax1.patches:
            ax1.annotate(
                f"{p.get_height():,.0f}",
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points'
            )
    else:
        ax1.text(0.5, 0.5, 'No Loss Data Available', ha='center', va='center')
    
    # ===== PROFITS VISUALIZATION =====
    if data_dict['profits']:
        totals_profit = {
            s: df['absolute_losses [t]'].abs().sum()  
            for s, df in data_dict['profits'].items()
        }
        group_total_profit = sum(v for k, v in totals_profit.items() if k != "ALL")
        totals_profit["SUM(PAK+URU+RUS+HOA)"] = group_total_profit
        sorted_profit = dict(sorted(totals_profit.items(), key=lambda x: x[1], reverse=True))
        
        sns.barplot(
            x=list(sorted_profit.keys()),
            y=list(sorted_profit.values()),
            palette="Greens_r",
            edgecolor="black",
            ax=ax2
        )
        
        ax2.set_title(
            f"Total Absolute Profits\n(Production Cap: {'ON' if production_cap else 'OFF'})",
            pad=15
        )
        ax2.set_ylabel("Total Profits [t]")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax2.tick_params(axis='x', rotation=45)
        
        for p in ax2.patches:
            ax2.annotate(
                f"{p.get_height():,.0f}",
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points'
            )
    else:
        ax2.text(0.5, 0.5, 'No Profit Data Available', ha='center', va='center')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"losses_profits_comparison_{'capped' if production_cap else 'uncapped'}.jpg"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n Saved visualization to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    print(f"\nWorking Directory: {current_dir}")
    print(f"Production Cap: {'ON' if production_cap else 'OFF'}")
    
    verify_files_exist()
    data_dict = load_and_validate_data()
    
    if data_dict['losses'] or data_dict['profits']:  # Fixed typo: 'losses' instead of 'losses'
        analyze_and_visualize(data_dict)
    else:
        print("Error: No valid data available for visualization")