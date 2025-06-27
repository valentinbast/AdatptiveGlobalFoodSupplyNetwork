import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from matplotlib.ticker import FuncFormatter

# Configuration
production_cap = True  # Change this to True/False as needed
scenario = ["ALL", "HOA", "RUS", "URU", "PAK"]
current_dir = Path.cwd()
loss_calculation_folder = current_dir / "evaluation"
output_dir = loss_calculation_folder / "plots"

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

def verify_files_exist():
    """Check if expected files exist"""
    print("\n🔍 Verifying input files:")
    for s in scenario:
        for capped in [True, False]:
            cap_status = "capped" if capped else "no_cap"
            filename = loss_calculation_folder / f"{s}-highestLosses_{cap_status}.csv"
            print(f"{'✅' if filename.exists() else '❌'} {filename.name}")

def load_and_validate_data():
    """Load data with validation"""
    df = {}
    for s in scenario:
        cap_status = "capped" if production_cap else "no_cap"
        filename = loss_calculation_folder / f"{s}-highestLosses_{cap_status}.csv"
        
        if not filename.exists():
            print(f"⚠️ File not found: {filename.name}")
            continue
            
        try:
            temp_df = pd.read_csv(filename)
            # Validate we have loss data
            if 'absolute_losses [t]' not in temp_df.columns:
                raise ValueError("Missing absolute_losses column")
                
            print(f"✅ Loaded {filename.name} ({len(temp_df)} records)")
            df[s] = temp_df
            
        except Exception as e:
            print(f"❌ Error loading {filename.name}: {str(e)}")
    
    return df

def analyze_and_visualize(data_dict):
    """Process and visualize the data"""
    # Calculate total losses per scenario
    totals = {
        s: df['absolute_losses [t]'].sum() 
        for s, df in data_dict.items()
        if 'absolute_losses [t]' in df.columns
    }
    
    # Add group total (excluding ALL)
    group_total = sum(v for k, v in totals.items() if k != "ALL")
    totals["SUM(PAK+URU+RUS+HOA)"] = group_total
    
    # Sort by value
    sorted_totals = dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))
    
    # Visualization
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        x=list(sorted_totals.keys()),
        y=list(sorted_totals.values()),
        palette="rocket",
        edgecolor="black"
    )
    
    # Formatting
    ax.set_title(
        f"Total Absolute Losses from highestLosses Files\n"
        f"(Production Cap: {'ON' if production_cap else 'OFF'})",
        pad=20
    )
    ax.set_ylabel("Total Losses [t]")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():,.0f}",
            (p.get_x() + p.get_width()/2, p.get_height()),
            ha='center', va='center',
            xytext=(0, 5),
            textcoords='offset points'
        )
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"highestLosses_totals_{'capped' if production_cap else 'uncapped'}.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved visualization to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    print(f"\nWorking Directory: {current_dir}")
    print(f"Production Cap: {'ON' if production_cap else 'OFF'}")
    
    # First verify files exist
    verify_files_exist()
    
    # Load and process data
    data_dict = load_and_validate_data()
    
    if data_dict:
        analyze_and_visualize(data_dict)
    else:
        print("⛔ No valid data available for visualization")