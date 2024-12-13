from data_loader import load_stock_data
from data_analysis import calculate_moving_average
from data_visualization import plot_stock_data

FILE_PATH = "stock_data.csv"

def main():
    #load the data
    stock_data = load_stock_data(FILE_PATH)

    #Calculate moving average
    stock_data["Moving Average"] = calculate_moving_average(stock_data["Moving Average"], window=5)

    #Visualize the data
    plot_stock_data(stock_data)

if __name__ == "__main__":
    main()
