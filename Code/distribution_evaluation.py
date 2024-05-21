import general_utilities

plot_kde(block_maxima_real_data_value, bm_samples_ddpm_value, x_axis_label = "Max Value", title = "KDE Density Plot of Max Values (Real vs DDPM Generated)")
KS_Test(block_maxima_real_data_value, bm_samples_ddpm_value)
CMD(block_maxima_real_data_value, bm_samples_ddpm_value)
KL_JS_divergence(block_maxima_real_data_value, bm_samples_ddpm_value)
CRPS(block_maxima_real_data_value, bm_samples_ddpm_value)

plot_kde(real_data.cpu().numpy().reshape(-1,1), bm_samples_ddpm, x_axis_label = "All Values", title = "KDE Density Plot of All Values (Real vs Generated (DDPM))")
KS_Test(real_data.cpu().numpy().reshape(-1), bm_samples_ddpm)
CMD(real_data.cpu().numpy().reshape(-1), bm_samples_ddpm)
KL_JS_divergence(real_data.cpu().numpy().reshape(-1), bm_samples_ddpm)
CRPS(real_data.cpu().numpy().reshape(-1), bm_samples_ddpm)
if is_fit_AR: fit_AR_model(torch.tensor(bm_samples_ddpm.reshape(-1,seq_len)).to(device),true_coeffs=[0.5], order =1)