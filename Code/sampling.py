@torch.no_grad()
def sample(t, bm_sample):
    cov = get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)

    x = L @ torch.randn_like(t)

    for diff_step in reversed(range(0, diffusion_steps)):
        alpha = alphas[diff_step]
        beta = betas[diff_step]

        z = L @ torch.randn_like(t)

        i = torch.Tensor([diff_step]).expand_as(x[...,:1]).to(device)
        pred_noise = model(x, t, i, bm_sample)

        x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() + beta.sqrt() * z
    return x
# bm_samples_conditional = np.vstack((bm_samples_gev, bm_samples_pos_uniform)).T
bm_samples_conditional = torch.tensor(bm_samples_gev, dtype=torch.float32).to(device)
bm_samples_conditional = bm_samples_conditional.reshape(-1,1,1)
t_grid = torch.linspace(0, 30, 30).view(1, -1, 1).to(device) # Note that we can use different sequence length here without any issues
samples_ddpm = sample(t_grid.repeat(num_samples, 1, 1), bm_samples_conditional)
for i in range(10):
  plt.plot(t_grid.squeeze().detach().cpu().numpy(), samples_ddpm[i].squeeze().detach().cpu().numpy(), color='C0', alpha=1 / (i + 1))
plt.title('10 new realizations')
plt.xlabel('t')
plt.ylabel('x')
plt.show()
# bm_samples_conditional =
bm_samples_ddpm_value, bm_samples_ddpm_pos = torch.max(samples_ddpm, dim=1)
bm_samples_ddpm_value = bm_samples_ddpm_value.cpu().numpy().reshape(-1)
bm_samples_conditional = bm_samples_gev#bm_samples_conditional.cpu().numpy().reshape(-1)
bm_samples_ddpm = samples_ddpm.cpu().numpy().reshape(-1)
print(f"Shape (block_maxima_real_data_value, bm_samples_conditional, samples_ddpm): {block_maxima_real_data_value.shape, bm_samples_conditional.shape, samples_ddpm.shape}")