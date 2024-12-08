import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import os

# Load the CSV file
file_path = os.path.join(os.getcwd(), "data_store_pkl_files/air_pollution.pkl")
data = pd.read_pickle(file_path)

# Pivot the data based on 'parameter' column
pivoted_data = data.pivot_table(index='date', columns='parameter', values='value')

# Keep only the 'pm25' column
pivoted_data = pivoted_data[['pm25']]

# Reset the index to ensure 'date' remains available
pivoted_data.reset_index(inplace=True)

# Convert 'date' to datetime and set it as the index
pivoted_data['date'] = pd.to_datetime(pivoted_data['date'])
pivoted_data.set_index('date', inplace=True)

# Sort data by date
pivoted_data = pivoted_data.sort_index()

# Interpolate missing values (if any) to ensure smooth seasonality analysis
pivoted_data['pm25'] = pivoted_data['pm25'].interpolate()

# Smooth the data using rolling means for different seasonality levels
daily_smoothed_data = pivoted_data['pm25'].rolling(window=24).mean()  # Daily rolling mean
weekly_smoothed_data = pivoted_data['pm25'].rolling(window=168).mean()  # Weekly rolling mean
monthly_smoothed_data = pivoted_data['pm25'].rolling(window=720).mean()  # Monthly rolling mean
quarterly_smoothed_data = pivoted_data['pm25'].rolling(window=2160).mean()  # Quarterly rolling mean

import os

def decompose_and_plot(data, period, title_suffix, save_dir="plots"):
    # Create the directory to save plots if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    stl = STL(data.dropna(), period=period, robust=True)
    result = stl.fit()
    
    # Plot STL decomposition
    fig = result.plot()
    fig.suptitle(f'STL Decomposition ({title_suffix})', y=1.02)
    # stl_plot_path = os.path.join(save_dir, f"stl_decomposition_{title_suffix}.png")
    fig.savefig(f"stl_decomposition_{title_suffix}.png")  # Save the STL decomposition plot
    # plt.show()
    
    # Plot seasonal component
    plt.figure(figsize=(10, 6))
    plt.plot(result.seasonal, label='Seasonal Component', color='green')
    plt.title(f'Seasonal Component ({title_suffix})')
    plt.xlabel('Date')
    plt.ylabel('Seasonal Effect')
    plt.legend()
    plt.grid()
    
    # seasonal_plot_path = os.path.join(save_dir, f"seasonal_component_{title_suffix}.png")
    plt.savefig(f"seasonal_component_{title_suffix}.png")  # Save the seasonal component plot
    # plt.show()
    
    # print(f"Plots saved: {stl_plot_path}, {seasonal_plot_path}")


# Daily Seasonality (period=24)
decompose_and_plot(daily_smoothed_data, period=24, title_suffix="Daily")

# Weekly Seasonality (period=168)
decompose_and_plot(weekly_smoothed_data, period=168, title_suffix="Weekly")

# Monthly Seasonality (period=720)
decompose_and_plot(monthly_smoothed_data, period=720, title_suffix="Monthly")

# Quarterly Seasonality (period=2160)
decompose_and_plot(quarterly_smoothed_data, period=2160, title_suffix="Quarterly")
