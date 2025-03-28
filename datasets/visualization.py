import pandas as pd
import matplotlib.pyplot as plt

# # Read the CSV data
# df = pd.read_csv('./Evaluations/system_metrics_preprocessed.csv')

# # Convert timestamp to datetime
# df['timestamp'] = pd.to_datetime(df['timestamp'])

# # Create the plot
# plt.figure(figsize=(12, 6))

# # Plot chrony_system_time_offset
# plt.plot(df['timestamp'], df['chrony_system_time_offset'], label='Chrony System Time Offset')

# # Plot chrony_last_offset
# plt.plot(df['timestamp'], -1 * df['chrony_last_offset'], label='Chrony Last Offset')

# plt.title('Chrony Time Offsets Over Time')
# plt.xlabel('Timestamp')
# plt.ylabel('Offset (seconds)')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()

# # Add legend
# plt.legend()

# # Format y-axis to scientific notation
# plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# # Add horizontal line at y=0 for reference
# plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

# # Show the plot
# plt.show()

from bokeh.plotting import figure, show
from bokeh.models import HoverTool

# Load your dataset
df_uc = pd.read_csv('./datasets/unsynced_uc.csv')
df_local = pd.read_csv('./datasets/unsynced.csv')
df_tacc = pd.read_csv('./datasets/synced_tacc.csv')

# Convert timestamp to datetime
df_uc['timestamp'] = pd.to_datetime(df_uc['system_time'])
df_local['timestamp'] = pd.to_datetime(df_local['system_time'])
df_tacc['timestamp'] = pd.to_datetime(df_tacc['system_time'])

global_start = max(df_uc['timestamp'].min(), df_local['timestamp'].min(), df_tacc['timestamp'].min())
global_end = min(df_uc['timestamp'].max(), df_local['timestamp'].max(), df_tacc['timestamp'].max())

df_uc['relative_time'] = (df_uc['timestamp'] - df_uc['timestamp'].iloc[0]).dt.total_seconds()
df_local['relative_time'] = (df_local['timestamp'] - df_local['timestamp'].iloc[0]).dt.total_seconds()
df_tacc['relative_time'] = (df_tacc['timestamp'] - df_tacc['timestamp'].iloc[0]).dt.total_seconds()

# Create the plot
p = figure(title='Chrony Time Offsets Over Time', x_axis_label='Timestamp', y_axis_label='Offset (seconds)')

# Add lines
p.line(df_uc['relative_time'], df_uc['chrony_system_time_offset'], legend_label='Time Offset (UC)')
p.line(df_local['relative_time'], df_local['chrony_system_time_offset'], legend_label='Time Offset (Local)')
p.line(df_tacc['relative_time'], df_tacc['chrony_system_time_offset'], legend_label='Time Offset (TACC)')

# Add hover tool
hover = HoverTool(tooltips=[
    ("Timestamp", "@x{%F %T}"),
    ("Offset", "@y")
], formatters={"x": "datetime"})

p.add_tools(hover)

# Show the plot
show(p)