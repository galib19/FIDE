
def get_betas(steps):
    beta_start, beta_end = 1e-4, 0.2
    diffusion_ind = torch.linspace(0, 1, steps).to(device)
    return beta_start * (1 - diffusion_ind) + beta_end * diffusion_ind

diffusion_steps = 100
betas = get_betas(diffusion_steps)
alphas = torch.cumprod(1 - betas, dim=0)

gp_sigma = 0.05

def get_gp_covariance(t):
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5 # for numerical stability
    return torch.exp(-torch.square(s / gp_sigma)) + diag

def add_noise(x, t, i):
    """
    x: Clean data sample, shape [B, S, D]
    t: Times of observations, shape [B, S, 1]
    i: Diffusion step, shape [B, S, 1]
    """
    noise_gaussian = torch.randn_like(x)

    cov = get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)
    noise = L @ noise_gaussian

    alpha = alphas[i.long()].to(x)
    x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise

    return x_noisy, noise


def get_loss(x, t, bm):
    i = torch.randint(0, diffusion_steps, size=(x.shape[0],))
    i = i.view(-1, 1, 1).expand_as(x[...,:1]).to(x)

    x_noisy, noise = add_noise(x, t, i)
    pred_noise = model(x_noisy, t, i, bm)
    ddpm_loss = torch.sqrt(torch.mean((pred_noise - noise)**2))

    if is_regularizer:
        lambda_1 = linear_decay(i[:,0,:].reshape(-1,1))
        pred_0 = x_noisy - pred_noise
        bm_pred, _ = torch.max(pred_0, dim=1)
        reg_loss = np.mean(gev_model.logpdf(bm_pred.detach().cpu().numpy()))
        reg_loss = -0.05*torch.tensor(reg_loss, dtype=torch.float32).to(device)

        loss = ddpm_loss + reg_loss
    else:
        loss = ddpm_loss
    return loss, ddpm_loss, reg_loss

# Plotting functions
def plot_losses(train_history, ylim_low=0, ylim_high=0.05):
    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plotting the first subplot (linear scale)
    x_values = np.arange(len(train_history)) * 10
    axes[0].plot(x_values, train_history, label="Training loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Plotting the second subplot (logarithmic scale)
    axes[1].plot(x_values, train_history, label="Training loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss (log scale)")
    axes[1].set_yscale("log")  # Set y-axis to logarithmic scale
    axes[1].legend()

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Show the plots
    plt.show()

# Example usage:
# train_history = np.random.rand(50) * 0.05  # Replace this with your actual training loss history
# plot_losses(train_history)

def linear_decaying_weight(input_tensor):
    # Assuming input_tensor is a torch tensor of shape [10000, 1]

    # Calculate the maximum and minimum values of the input tensor
    max_value = torch.max(input_tensor)
    min_value = torch.min(input_tensor)

    # Ensure the input tensor is in the range [0, 1]
    normalized_input = (input_tensor - min_value) / (max_value - min_value)

    # Calculate linear decay weights
    weights = 1 - normalized_input

    return weights

def linear_decay(input_tensor):
    # Set the decay starting point and ending point
    start_index = 0
    end_index = int(0.66*diffusion_steps)

    # Create an output tensor with the same size as the input tensor
    output_tensor = torch.zeros_like(input_tensor)

    # Apply linear decay to the values based on their indices
    for i in range(input_tensor.size(0)):
        index_value = input_tensor[i, 0].item()

        if index_value == start_index:
            output_tensor[i, 0] = 1.0
        elif start_index < index_value < end_index:
            # Linear decay function: f(x) = 1 - (x - start_index) / (end_index - start_index)
            output_tensor[i, 0] = 1.0 - (index_value - start_index) / (end_index - start_index)
        else:
            output_tensor[i, 0] = 0.0

    return output_tensor

# Example usage:
# input_values = i[:,0,:].reshape(-1,1)   # Assuming your input values range from 0 to 99
# weights = linear_decaying_weight(input_values)

