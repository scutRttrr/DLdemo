

from chapter6.general_functions import *
from chapter6.h1_functions import  *
from chapter6.h2_functions import *

def main():
    data_path = '../Data/portfolio_data.csv'
    T = 10
    k = 1
    # %%
    ticker_list = ['AXP', 'CI', 'GD', 'HON', 'MMM',
                   'MO', 'MRK', 'PFE', 'PG', 'SO',
                   'NEM', 'NSC', 'GE', 'AEP', 'AAPL',
                   'ABT', 'SNA', 'PTC', 'BAC', 'NKE']

    train_data, val_data, test_data = read_data('../Data/portfolio_data.csv')
    # %%
    train_x = data_classification(train_data[[tic + '_Return' for tic in ticker_list]], k, T)
    train_y = prepare_y(train_data[[tic + '_Return' for tic in ticker_list]], k, T)

    val_x = data_classification(val_data[[tic + '_Return' for tic in ticker_list]], k, T)
    val_y = prepare_y(val_data[[tic + '_Return' for tic in ticker_list]], k, T)

    test_x = data_classification(test_data[[tic + '_Return' for tic in ticker_list]], k, T)
    test_y = prepare_y(test_data[[tic + '_Return' for tic in ticker_list]], k, T)

    print(train_x.shape, train_y.shape)
    print(val_x.shape, val_y.shape)
    print(test_x.shape, test_y.shape)

    train_loader = torch.utils.data.DataLoader(dataset=MyDataset(train_x, train_y), batch_size=512, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=MyDataset(val_x, val_y), batch_size=512, shuffle=False)

    # %%
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MLP(T, len(ticker_list), y_dim=len(ticker_list))
    model.to(device)
    model, y_pred = train_model(model, train_loader, val_loader, test_x, savepath='./model/best_mlp', epochs=1000,
                                lr=0.001, patience=50)

    port_ret = np.array(test_y) * np.array(y_pred)
    port_ret = np.sum(port_ret, axis=1)
    port_cumret = np.cumsum(port_ret)
    # %%
    print(report_metrics(port_ret))

    # %%
    plt.figure(figsize=(10, 6))
    plt.plot(port_cumret, label="MLP")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.show()

# 检查是否作为独立脚本运行
if __name__ == "__main__":
    main()