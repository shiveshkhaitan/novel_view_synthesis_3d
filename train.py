from torch.utils import data
from pathlib import Path

import functools
import jax
import jax.numpy as jnp
import optax
import numpy as np

from flax import jax_utils
from flax.training import train_state, checkpoints

from dataset.data_loader import SceneClassDataset
from model.xunet import XUNet

from tqdm import tqdm

def cycle(dl):
    while True:
        for data in dl:
            yield data

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

@functools.partial(jax.pmap, static_broadcasted_argnums=(2, 3, 4))
def create_train_state(rng, rng_dropout, learning_rate, train_batch_size, img_sidelength):
    """Creates initial `TrainState`."""
    diffusion_model = XUNet()
    sample = create_sample_data(train_batch_size, img_sidelength)
    params = diffusion_model.init({'params' : rng, 'dropout' : rng_dropout},
                                  sample,
                                  cond_mask=np.zeros((train_batch_size)), train=True)['params']

    opt = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=diffusion_model.apply, params=params, tx=opt)
    return state

@functools.partial(jax.pmap, axis_name='ensemble')
def apply_model(state, batch_x, batch_z, batch_logsnr, batch_R1, batch_t1, batch_R2, batch_t2, batch_K, batch_noise):
    """Computes gradients, loss and accuracy for a single batch."""
    print(batch_x.shape)
    batch = dict(x=batch_x,
                 z=batch_z,
                 logsnr=batch_logsnr,
                 R1=batch_R1,
                 t1=batch_t1,
                 R2=batch_R2,
                 t2=batch_t2,
                 K=batch_K)

    def loss_fn(params):
        output = state.apply_fn({'params': params}, batch,
                                cond_mask=np.where(np.random.random((batch_x.shape[0])) > 0.1, 1, 0),
                                train=True,
                                rngs={'dropout': jax.random.PRNGKey(0)})
        loss = jnp.mean(jnp.linalg.norm(output - jnp.array(batch_noise)))
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)
    return loss, grads

@jax.pmap
def update_model(state, grads):
  return state.apply_gradients(grads=grads)

class Trainer(object):
    def __init__(
        self,
        folder,
        *,
        train_batch_size = 2,
        train_lr = 1e-4,
        train_num_steps = 100000,
        save_every = 1000,
        img_sidelength = 64,
        results_folder = './results',
    ):
        super().__init__()

        self.save_every = save_every

        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps

        self.img_sidelength = img_sidelength

        self.ds = SceneClassDataset(root_dir=folder,
                                    max_num_instances=-1,
                                    max_observations_per_instance=50,
                                    img_sidelength=self.img_sidelength,
                                    specific_observation_idcs=None,
                                    samples_per_instance=1)

        assert len(self.ds) > 0

        self.dl = cycle(data.DataLoader(self.ds,
                                        batch_size = train_batch_size,
                                        shuffle=True, 
                                        drop_last=True,
                                        collate_fn=self.ds.collate_fn,
                                        pin_memory=True))

        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        rng = jax.random.PRNGKey(0)
        rng, dropout_rng = jax.random.split(rng)
        self.train_state = create_train_state(jax.random.split(rng, jax.device_count()),
                                              jax.random.split(dropout_rng, jax.device_count()),
                                              train_lr,
                                              train_batch_size,
                                              img_sidelength)
    def train(self):

        while self.step < self.train_num_steps:
            data = next(self.dl)[0]

            model_input_x = jax_utils.replicate(data['x'].numpy())
            model_input_z = jax_utils.replicate(data['z'].numpy())
            model_input_logsnr = jax_utils.replicate(np.array(data['logsnr']))
            model_input_R1 = jax_utils.replicate(data['R1'].numpy())
            model_input_t1 = jax_utils.replicate(data['t1'].numpy())
            model_input_R2 = jax_utils.replicate(data['R2'].numpy())
            model_input_t2 = jax_utils.replicate(data['t2'].numpy())
            model_input_K = jax_utils.replicate(data['K'].numpy())
            model_input_noise = jax_utils.replicate(data['noise'].numpy())

            loss, grads = apply_model(self.train_state,
                                      model_input_x,
                                      model_input_z,
                                      model_input_logsnr,
                                      model_input_R1,
                                      model_input_t1,
                                      model_input_R2,
                                      model_input_t2,
                                      model_input_K,
                                      model_input_noise)

            self.train_state = update_model(self.train_state, grads)

            loss = jax_utils.unreplicate(loss)

            print(f'{self.step}: {loss.item()}')

            if self.step % self.save_every == 0:

                checkpoints.save_checkpoint(
                    ckpt_dir='checkpoints/',  # Folder to save checkpoint in
                    target=self.train_state.params,  # What to save. To only save parameters, use model_state.params
                    step=self.step,  # Training step or other metric to save best model on
                    prefix='model',  # Checkpoint file name prefix
                    overwrite=True   # Overwrite existing checkpoint files
                )

            self.step += 1

        print('training completed')


if __name__ == "__main__":
    trainer = Trainer('cars_train_val')
    trainer.train()