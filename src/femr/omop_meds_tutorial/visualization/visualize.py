import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Set publication-quality defaults
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# ==========================================
# Data Preparation
# ==========================================

# --- TRANSFORMER BACKBONE DATA ---
data_mimic_trans = {
    'Title': '(1) MIMIC Patients\n(Transformer Backbone)',
    'Tasks': ['Bilirubin', 'Creatinine', 'Platelets','PaO₂'],
    'Data': {
        'Best Baseline vs NTP': [2.99, 6.74,-1.01, -1.85],
        'Unmark vs NTP': [5.35, 11.85,  -0.15,-1.66],
        'Mark vs NTP': [7.34, 11.85, 24.25,0.56],
    }
}

data_cumc_trans = {
    'Title': '(2) CUMC Patients\n(Transformer Backbone)',
    'Tasks': ['Bilirubin', 'Creatinine', 'Platelets', 'SpO₂'],
    'Data': {
        'Best Baseline vs NTP': [30.32, 11.83, -7.46, 2.45],
        'Unmark vs NTP': [17.42, -7.89, -10.96, 1.96],
        'Mark vs NTP': [43.87, 25.75, 53.07, 2.45],
    }
}

# --- MAMBA BACKBONE DATA ---
data_mimic_mamba = {
    'Title': '(1) MIMIC Patients\n(Mamba Backbone)',
    'Tasks': ['Bilirubin', 'Creatinine', 'Platelets','PaO₂'],
    'Data': {
        'Best Baseline vs NTP': [-0.65, -4.65, -8.79,-2.60],
        'Unmark vs NTP': [10.71, 2.81, -0.58,-1.94],
        'Mark vs NTP': [10.46, 6.38, 24.17,1.08],
    }
}

data_cumc_mamba = {
    'Title': '(2) CUMC Patients\n(Mamba Backbone)',
    'Tasks': ['Bilirubin', 'Creatinine', 'Platelets', 'SpO₂'],
    'Data': {
        'Best Baseline vs NTP': [33.77, 13.95, -4.09, 3.34],
        'Unmark vs NTP': [36.42, 28.37, 41.36, 4.94],
        'Mark vs NTP': [67.55, 37.59, 110.91, 5.32],
    }
}

# ==========================================
# Professional color scheme (same as reference)
# ==========================================
COLORS = {
    'Mark vs NTP': '#2ca02c',            # Green
    'Unmark vs NTP': '#1f77b4',          # Blue  
    'Best Baseline vs NTP': '#d62728',   # Red
}
MARKERS = {
    'Mark vs NTP': 'o',
    'Unmark vs NTP': 's',
    'Best Baseline vs NTP': '^',
}
LINESTYLES = {
    'Mark vs NTP': '-',
    'Unmark vs NTP': '--',
    'Best Baseline vs NTP': ':',
}

# ==========================================
# Improved Radar Chart Function
# ==========================================
def make_radar_chart(ax, data_dict, show_annotations=True):
    """
    Creates a clean, publication-ready radar chart.
    """
    tasks = data_dict['Tasks']
    datasets = data_dict['Data']
    N = len(tasks)
    
    # Compute angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Find data range for scaling
    all_values = [val for sublist in datasets.values() for val in sublist]
    max_val = max(all_values)
    min_val = min(all_values)
    
    # Set floor and ceiling
    floor = min(-5, min_val * 1.1) if min_val < 0 else -5
    ceiling = max_val * 1.1
    
    # Configure polar plot
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)       # Clockwise
    
    # Remove default tick labels
    ax.set_xticklabels([])
    
    # Add task labels CLOSE to the chart (just slightly outside)
    for i, (angle, task) in enumerate(zip(angles[:-1], tasks)):
        angle_deg = np.degrees(angle) % 360
        
        # Calculate label position - slightly different padding for top label
        if 350 < angle_deg or angle_deg < 10:  # Top (Bilirubin)
            label_r = ceiling * 1.12  # Slightly more padding for top
            ha, va = 'center', 'bottom'
        elif 170 < angle_deg < 190:  # Bottom
            label_r = ceiling * 1.08
            ha, va = 'center', 'top'
        elif angle_deg < 180:  # Right side
            label_r = ceiling * 1.08
            ha, va = 'left', 'center'
        else:  # Left side
            label_r = ceiling * 1.08
            ha, va = 'right', 'center'
        
        ax.text(angle, label_r, task,
                ha=ha, va=va,
                fontsize=11, fontweight='bold',
                color='black')
    
    # Set up radial grid
    if ceiling > 100:
        step = 25
    elif ceiling > 50:
        step = 20
    else:
        step = 10
    
    # Create grid values
    grid_values = list(range(0, int(ceiling) + step, step))
    if floor < 0:
        neg_step = 5 if abs(floor) < 15 else 10
        neg_grids = list(range(int(np.floor(floor / neg_step) * neg_step), 0, neg_step))
        grid_values = neg_grids + grid_values
    
    grid_values = [g for g in grid_values if floor <= g <= ceiling]
    
    # Style the radial axis
    ax.set_ylim(floor, ceiling)
    ax.set_yticks(grid_values)
    ax.set_yticklabels([f'{int(g)}%' for g in grid_values], 
                       fontsize=9, color='grey')
    
    # Style the grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='grey', linewidth=0.8)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3, color='grey', linewidth=0.8)
    
    # Highlight the 0% reference line
    zero_line = [0] * (N + 1)
    ax.plot(angles, zero_line, linestyle='-', linewidth=1.5, 
            color='black', alpha=0.6, zorder=5)
    
    # Plot each model's data
    for model_name, values in datasets.items():
        values_closed = values + values[:1]
        color = COLORS[model_name]
        marker = MARKERS[model_name]
        linestyle = LINESTYLES[model_name]
        
        # Plot line with smaller markers
        ax.plot(angles, values_closed,
                color=color, linewidth=2, linestyle=linestyle,
                marker=marker, markersize=7,  # Smaller markers
                label=model_name, zorder=10)
        
        # Fill area with light color
        ax.fill(angles, values_closed, color=color, alpha=0.1, zorder=2)
    
    # Add annotations only for Mark vs NTP (smaller, less intrusive)
    if show_annotations:
        mark_values = datasets['Mark vs NTP']
        for i, (angle, val) in enumerate(zip(angles[:-1], mark_values)):
            angle_deg = np.degrees(angle) % 360
            
            # Position annotation - adjust based on position to avoid label overlap
            offset = ceiling * 0.15
            
            # For bottom position (Platelets in CUMC), put annotation inside
            if 170 < angle_deg < 190:  # Bottom
                if val > 0:
                    ann_r = val - offset  # Put inside for bottom
                else:
                    ann_r = val - offset
            else:
                if val >= 0:
                    ann_r = val + offset
                else:
                    ann_r = val - offset
            
            # Format annotation
            sign = '+' if val > 0 else ''
            annotation = f'{sign}{val:.1f}%'
            
            # Color based on sign
            ann_color = '#2ca02c' if val > 0 else '#d62728'
            
            ax.annotate(annotation,
                       xy=(angle, val),
                       xytext=(angle, ann_r),
                       fontsize=9, fontweight='bold',
                       color=ann_color,
                       ha='center', va='center',
                       zorder=20)
    
    # Set title with more padding to avoid overlap with Bilirubin
    ax.set_title(data_dict['Title'], fontsize=13, fontweight='bold', 
                 pad=35, color='black')
    
    # Style the spine
    ax.spines['polar'].set_color('#CCCCCC')
    ax.spines['polar'].set_linewidth(1)
    
    return ax

def create_figure(data_left, data_right, main_title, filename):
    """Creates a figure with two radar subplots."""
    fig = plt.figure(figsize=(14, 7))
    
    # Add main title
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=1.02)
    
    # Create subplots with more space for the charts
    ax1 = fig.add_subplot(121, polar=True)
    ax2 = fig.add_subplot(122, polar=True)
    
    # Create radar charts
    make_radar_chart(ax1, data_left)
    make_radar_chart(ax2, data_right)
    
    # Create custom legend (same style as reference)
    legend_elements = [
        Line2D([0], [0], color=COLORS['Mark vs NTP'], linewidth=2,
               linestyle=LINESTYLES['Mark vs NTP'],
               marker=MARKERS['Mark vs NTP'], markersize=7,
               label='Mark vs NTP'),
        Line2D([0], [0], color=COLORS['Unmark vs NTP'], linewidth=2,
               linestyle=LINESTYLES['Unmark vs NTP'],
               marker=MARKERS['Unmark vs NTP'], markersize=7,
               label='Unmark vs NTP'),
        Line2D([0], [0], color=COLORS['Best Baseline vs NTP'], linewidth=2,
               linestyle=LINESTYLES['Best Baseline vs NTP'],
               marker=MARKERS['Best Baseline vs NTP'], markersize=7,
               label='Best Baseline vs NTP'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', 
               bbox_to_anchor=(0.5, -0.05), ncol=3, 
               frameon=False, fontsize=12)
    
    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight', dpi=300, 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {filename}")

# ==========================================
# Generate Figures
# ==========================================

# Figure 1: Transformer Backbone
create_figure(
    data_mimic_trans, 
    data_cumc_trans,
    "Relative Improvement over Transformer-NTP Baseline (%)",
    "transformer_radar.png"
)

# Figure 2: Mamba Backbone  
create_figure(
    data_mimic_mamba,
    data_cumc_mamba,
    "Relative Improvement over Mamba-NTP Baseline (%)",
    "mamba_radar.png"
)

print("\nAll figures generated successfully!")