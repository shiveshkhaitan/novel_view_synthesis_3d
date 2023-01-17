from torch.utils import data
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import numpy as np

from flax.training import train_state, checkpoints

from data_loader import SceneClassDataset
from model import XUNet

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

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 128,
        train_lr = 1e-4,
        train_num_steps = 100000,
        save_every = 1000,
        img_sidelength = 64,
        results_folder = './results',
    ):
        super().__init__()
        self.model = diffusion_model

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

        sample = create_sample_data(train_batch_size, self.img_sidelength)
        params = diffusion_model.init({'params' : jax.random.PRNGKey(0), 'dropout' : jax.random.PRNGKey(1)},
                                      sample,
                                      cond_mask=np.zeros((train_batch_size)), train=True)['params']

        self.opt = optax.adam(train_lr)
        self.train_state = train_state.TrainState.create(apply_fn=diffusion_model.apply, params=params, tx=self.opt)

        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)


    def train(self):

        while self.step < self.train_num_steps:
            data = next(self.dl)[0]

            @jax.jit
            def apply_model(state, data):
              """Computes gradients, loss and accuracy for a single batch."""
              def loss_fn(params):
                output = state.apply_fn({'params': params},
                                        data,
                                        cond_mask=np.where(np.random.random((data['x'].shape[0])) > 0.1, 1, 0),
                                        train=True,
                                        rngs={'dropout': jax.random.PRNGKey(0)})
                loss = jnp.mean(jnp.linalg.norm(output - jnp.array(data['noise'])))
                return loss

              grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
              loss, grads = grad_fn(state.params)
              return loss, grads

            @jax.jit
            def update_model(state, grads):
              return state.apply_gradients(grads=grads)

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

            loss, grads = apply_model(self.train_state, model_input)
            self.train_state = update_model(self.train_state, grads)

            print(f'{self.step}: {loss.item()}')

            if self.step % self.save_every == 0:

                checkpoints.save_checkpoint(ckpt_dir='checkpoints/',  # Folder to save checkpoint in
                                            target=self.train_state.params,  # What to save. To only save parameters, use model_state.params
                                            step=self.step,  # Training step or other metric to save best model on
                                            prefix='model',  # Checkpoint file name prefix
                                            overwrite=True   # Overwrite existing checkpoint files
                                           )

            self.step += 1

        print('training completed')


if __name__ == "__main__":
    diffusion_model = XUNet()
    trainer = Trainer(diffusion_model, 'cars_train_val')
    trainer.train()