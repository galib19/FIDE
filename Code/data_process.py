def process_data(ori_data):
    # Normalize the data
    scaler = StandardScaler().fit(ori_data)
    ori_data = scaler.transform(ori_data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - time_series_seq_len):
      _x = ori_data[i:i + time_series_seq_len]
      temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.arange(len(temp_data))
    data = []
    for i in range(len(temp_data)):
      data.append(temp_data[idx[i]])
    return torch.from_numpy(np.array(data))

import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

def fit_AR_model(data, true_coeffs, order =2):

    time_series_data = np.array(data.cpu()).reshape(-1,seq_len)

    print("True AR Coefficients:", true_coeffs)

    # Estimate AR coefficients from the generated data
    estimated_ar_coefficients = []
    for series in time_series_data:
        model = AutoReg(series, lags=order)
        result = model.fit()
        estimated_ar_coefficients.append(result.params[1:])

    # Convert the estimated coefficients to a numpy array
    estimated_ar_coefficients = np.array(estimated_ar_coefficients)

    # Print the estimated AR coefficients (mean across all samples)
    print("Estimated AR Coefficients:", np.mean(estimated_ar_coefficients, axis=0))

    mse = np.mean((true_coeffs - np.mean(estimated_ar_coefficients, axis=0))**2)
    mae = np.mean(np.abs(true_coeffs - np.mean(estimated_ar_coefficients, axis=0)))
    print("Mean squared error of the estimated coefficients, MSE:", mse)
    print("Mean absolute error of the estimated coefficients, MAE:", mae)