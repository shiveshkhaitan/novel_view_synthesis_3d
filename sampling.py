import jax
import numpy as np
import optax

import cv2

from torch.utils import data

from flax.training import train_state, checkpoints
from flax.core import freeze

from dataset.data_loader import SceneClassDataset

from model.xunet import XUNet

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype = np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.9999)

betas = cosine_beta_schedule(1000)
alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.pad(alphas_cumprod[:-1], (1, 0), 'constant', constant_values=(1))
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_log_variance_clipped = np.log(posterior_variance.clip(min =1e-20))
posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

def predict_start_from_noise(x_t, t, noise):
    return (sqrt_recip_alphas_cumprod[t] * x_t - sqrt_recipm1_alphas_cumprod[t] * noise)

def q_posterior(x_start, x_t, t):
    posterior_mean = (
        posterior_mean_coef1[t] * x_start +
        posterior_mean_coef2[t] * x_t
    )
    pos_var = posterior_variance[t]
    pos_log_var_clipped = posterior_log_variance_clipped[t]
    return posterior_mean, pos_var, pos_log_var_clipped

img_sidelength = 64

ds = SceneClassDataset(root_dir='./cars_train_val',
                       max_num_instances=-1,
                       max_observations_per_instance=50,
                       img_sidelength=img_sidelength,
                       specific_observation_idcs=None,
                       samples_per_instance=1)

diffusion_model = XUNet()

batch_size = 1

def cycle(dl):
    while True:
        for data in dl:
            yield data

def logsnr_schedule_cosine(t, *, logsnr_min=-20., logsnr_max=20.):
    b = np.arctan(np.exp(-.5 * logsnr_max))
    a = np.arctan(np.exp(-.5 * logsnr_min)) - b
    return -2. * np.log(np.tan(a * t + b))


def create_sample_data(batch_size, img_sidelength):
    sample = dict()
    sample['x'] = np.random.random((batch_size, img_sidelength, img_sidelength, 3))
    sample['z'] = np.random.random((batch_size, img_sidelength, img_sidelength, 3))
    sample['logsnr'] = np.random.random((batch_size))
    sample['R1'] = np.random.random((batch_size, 3, 3))
    sample['t1'] = np.random.random((batch_size, 3))
    sample['R2'] = np.random.random((batch_size, 3, 3))
    sample['t2'] = np.random.random((batch_size, 3))
    sample['K'] = np.random.random((batch_size, 3, 3))
    sample['noise'] = np.random.random((batch_size, img_sidelength, img_sidelength, 3))
    return sample

dl = cycle(data.DataLoader(ds,
                           batch_size = batch_size,
                           shuffle=True,
                           drop_last=True,
                           collate_fn=ds.collate_fn,
                           pin_memory=True))

sample = create_sample_data(batch_size, img_sidelength)
params = diffusion_model.init({'params' : jax.random.PRNGKey(0), 'dropout' : jax.random.PRNGKey(1)},
                              sample,
                              cond_mask=np.zeros((batch_size)), train=True)['params']

train_state = train_state.TrainState.create(apply_fn=diffusion_model.apply, params=params, tx=optax.adam(1e-3))

loaded_model_state = checkpoints.restore_checkpoint(
                                                     ckpt_dir='checkpoints',   # Folder with the checkpoints
                                                     target=train_state,   # (optional) matching object to rebuild state in
                                                     prefix='model0'  # Checkpoint file name prefix
                                                   )
if loaded_model_state is train_state:
    raise FileNotFoundError(f"Checkpoint does not exist")

params = freeze(loaded_model_state.params)

while True:
    data = next(dl)[0]

    def apply_model(state, data):
        def t_schedule_cosine(logsnr, *, logsnr_min=-20., logsnr_max=20.):
            b = np.arctan(np.exp(-.5 * logsnr_max))
            a = np.arctan(np.exp(-.5 * logsnr_min)) - b
            return (((np.arctan(np.exp(logsnr / -2)) - b) / a) * 1000).astype(int)

        data['z'] = np.random.randn(*data['x'].shape)
        data['logsnr'] = np.ones(*data['logsnr'].shape) * -20

        for time_step in range(999, -1, -1):

            def p_mean_variance():
                output1 = state.apply_fn({'params': params}, data, cond_mask=np.ones(data['x'].shape[0]), train=False)
                output2 = state.apply_fn({'params': params}, data, cond_mask=np.zeros(data['x'].shape[0]), train=False)
                w = 3
                output = (1 + w) * output1  - w * output2

                x_recon = predict_start_from_noise(data['z'], t=time_step, noise = output)
                x_recon = np.clip(x_recon, -1., 1.)

                model_mean, pos_var, pos_log_var = q_posterior(x_start=x_recon, x_t=data['z'], t=time_step)
                return model_mean, pos_var, pos_log_var

            def p_sample():
                b = data['z'].shape[0]
                model_mean, _, model_log_variance = p_mean_variance()
                noise = np.random.randn(*data['z'].shape)
                # no noise when t == 0
                nonzero_mask = np.array(1.0 - (time_step == 0)).reshape(b, *((1,) * (len(data['z']) - 1)))
                return model_mean + nonzero_mask * np.exp(0.5 * model_log_variance) * noise

            data['z'] = p_sample()
            data['logsnr'] = logsnr_schedule_cosine(time_step / 1000.0)

        cv2.imshow('output', np.array(data['z'][0] / 2.0 + 0.5))
        cv2.waitKey(0)

    model_input = dict()
    model_input['x'] = data['x'].numpy()
    model_input['z'] = data['z'].numpy()
    model_input['logsnr'] = np.array(data['logsnr'])
    model_input['R1'] = data['R1'].numpy()
    model_input['t1'] = data['t1'].numpy()
    model_input['R2'] = data['R2'].numpy()
    model_input['t2'] = data['t2'].numpy()
    model_input['K'] = data['K'].numpy()
    model_input['noise'] = data['noise'].numpy()

    apply_model(loaded_model_state, model_input)
