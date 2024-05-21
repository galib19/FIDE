# FIDE

## Data
All data files can be found in the **Data** folder.

## Code
All code files are available in the **Code** folder (*todo: Example notebook will be uploaded*).

### Code Files
- `data_process.py`: Contains data processing steps.
- `distribution_evaluation.py`: Runs all distribution evaluation functions (defined in `general_utilities`).
- `general_utilities.py`: Contains general functions and distribution evaluation functions.
- `evaluation_predictive_discriminative_score_all_values.py`: Contains evaluations of predictive and discriminative scores (according to TimeGAN paper) for all time step values.
- `evaluation_predicitive_score_block_maxima.py`: Contains evaluations of predictive scores (according to TimeGAN paper) for only block maxima.
- `main_temperature_example.py`: Contains the main program with example temperature data.
- `model.py`: Contains the conditional diffusion-based model.
- `sampling.py`: Contains the sampling procedure for generating samples.
- `train_utilities.py`: Contains functions for training, including plotting and metrics calculations.
- `TSNE_evaluation.py`: Measures and plots t-SNE of the real and generated data.
