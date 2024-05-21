from typing import List, Callable
import math
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
from scipy.fft import rfft, rfftfreq, irfft

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from pylab import rcParams
rcParams['figure.figsize'] = 24, 10
plt.style.use('default')
plt.rcParams.update({'font.size': 14})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#setting parameters
time_series_seq_len = 30
seq_len = time_series_seq_len
n_seq = 1
is_fit_AR = False
data_folder_path = "/content/drive/MyDrive/PhD/Diffusion/Data/"
data_path = data_folder_path+ 'temperature_raw.csv'
#processing data
data = pd.read_csv(data_path, parse_dates=['Date'], error_bad_lines=False)
data.columns = ['date', 'temp']
no_digit = data[~data.temp.str[0].str.isdigit()]
data = data[~data.index.isin(no_digit.index)]
data['temp'] = data['temp'].astype(float)

data.set_index('date', inplace=True)

# Extract year and month from the date
data['year'] = data.index.year
data['month'] = data.index.month

# Calculate monthly mean and standard deviation
monthly_stats = data.groupby(['year', 'month'])['temp'].agg(['mean', 'std']).reset_index()

# Merge the monthly statistics back to the original data
data = pd.merge(data, monthly_stats, on=['year', 'month'], how='left', suffixes=('', '_monthly'))

# Standardize the temperature based on monthly mean and standard deviation
data['temp_standardized'] = (data['temp'] - data['mean']) / data['std']

# Drop unnecessary columns
data.drop(['year', 'month', 'mean', 'std'], axis=1, inplace=True)

real_data = np.array(data["temp_standardized"]).reshape(-1,1)
print(f"Real data shape: {real_data.shape}")

real_data = process_data(real_data).to(device)
t = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, seq_len, 1).expand(real_data.shape[0], seq_len, 1).to(device)
real_data = real_data.to(dtype=torch.float32)
print(f"Real Data: Mean: {torch.mean(real_data)} Std:{torch.std(real_data)}")
if is_fit_AR: fit_AR_model(real_data, true_coeffs=[0.5], order =1)

block_maxima_real_data_value, block_maxima_real_data_pos = torch.max(real_data, dim=1)
block_maxima_real_data_value = block_maxima_real_data_value.reshape(-1,1,1)
block_maxima_real_data_pos = block_maxima_real_data_pos.reshape(-1,1,1)
block_maxima_real_data = block_maxima_real_data_value #torch.cat((block_maxima_real_data_value, block_maxima_real_data_pos), dim=1)

print(f"Shape of real data, time steps (t), block maxima after processing: {real_data.shape,  t.shape, block_maxima_real_data.shape}")
num_samples = block_maxima_real_data.shape[0]
print(f"Number of samples: {num_samples}")

#Fitting GEV and Metrics

block_maxima_real_data_value = block_maxima_real_data_value.cpu().numpy().reshape(-1)
block_maxima_real_data_pos = block_maxima_real_data_pos.cpu().numpy().reshape(-1)
bm_samples_gev, gev_model= fitting_gev_and_sampling(block_maxima_real_data_value, num_samples)
plot_kde(block_maxima_real_data_value, bm_samples_gev, x_axis_label = "Max Value", title = "KDE Density Plot of Max Values (Real vs GEV Fitted)")
KS_Test(block_maxima_real_data_value, bm_samples_gev)
CMD(block_maxima_real_data_value, bm_samples_gev)
KL_JS_divergence(block_maxima_real_data_value, bm_samples_gev)
CRPS(block_maxima_real_data_value, bm_samples_gev)

# frequency enhancement

if not isinstance(real_data, np.ndarray):
    real_data = real_data.cpu().numpy()
real_data_fft = rfft(real_data, axis=1)
real_data_freq = rfftfreq(real_data.shape[1])
print(real_data_freq.shape)

c = 1.1

percentage_of_freq_enhanced = 20
top_freq_enhanced = int((real_data_fft.shape[1] * percentage_of_freq_enhanced) / 100)

high_freq_enhanced_fft_result = real_data_fft.copy()
top_indices = np.argsort(real_data_freq)[-top_freq_enhanced:]

# Iterate over all datapoints along the second dimension
for i in range(real_data_fft.shape[0]):
    high_freq_enhanced_fft_result[i, :, 0][top_indices] *= c

# print(real_data_fft[0], high_freq_enhanced_fft_result[0], real_data_fft[-1], high_freq_enhanced_fft_result[-1])

print(high_freq_enhanced_fft_result.shape)
real_data = irfft(high_freq_enhanced_fft_result.reshape(-1, real_data_freq.shape[0]))
real_data  =  torch.from_numpy(real_data.reshape(real_data.shape[0], real_data.shape[1], 1)).to(device)

#training

is_regularizer = True

batch_size = 2000
n_epochs = 400

#model initialization

model = TransformerModel(dim=1, hidden_dim=64, max_i=diffusion_steps).to(device)
optim = torch.optim.Adam(model.parameters())

training_loss_history = np.array([])

train_loader = DataLoader(TensorDataset(real_data, t, block_maxima_real_data), batch_size=batch_size, shuffle=False, drop_last=True)
for epoch in tqdm(range(n_epochs)):
    for i, (input, time, bm) in enumerate(train_loader):
      optim.zero_grad()
      loss, ddpm_loss, reg_loss = get_loss(input, time, bm)
      loss.backward()
      optim.step()
    if epoch<25 or epoch%20==0: print(f"Iteration: {epoch} --- Loss: {loss.cpu().item()}, DDPM Loss: {ddpm_loss.cpu().item()}, Regularizer Loss: {reg_loss.cpu().item()}")
    # if epoch<25 or epoch%20==0: print(f"Iteration: {epoch} --- Loss: {loss.cpu().item()}")
    training_loss_history = np.append(training_loss_history, loss.item())
    # break
plot_losses(training_loss_history)

#after training: sampling and evaluation with the corresponding files
