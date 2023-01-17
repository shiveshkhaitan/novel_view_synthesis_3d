import os
import torch
import numpy as np
from glob import glob
import data_util
import util


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class SceneInstanceDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 instance_dir,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 img_sidelength=None,
                 num_images=-1):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = instance_dir

        color_dir = os.path.join(instance_dir, "rgb")
        pose_dir = os.path.join(instance_dir, "pose")
        param_dir = os.path.join(instance_dir, "params")

        if not os.path.isdir(color_dir):
            print("Error! root dir %s is wrong" % instance_dir)
            return

        self.has_params = os.path.isdir(param_dir)
        self.color_paths = sorted(data_util.glob_imgs(color_dir))
        self.pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))

        if self.has_params:
            self.param_paths = sorted(glob(os.path.join(param_dir, "*.txt")))
        else:
            self.param_paths = []

        if specific_observation_idcs is not None:
            self.color_paths = pick(self.color_paths, specific_observation_idcs)
            self.pose_paths = pick(self.pose_paths, specific_observation_idcs)
            self.param_paths = pick(self.param_paths, specific_observation_idcs)
        elif num_images != -1:
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_paths = pick(self.color_paths, idcs)
            self.pose_paths = pick(self.pose_paths, idcs)
            self.param_paths = pick(self.param_paths, idcs)

        betas = cosine_beta_schedule(1000)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.pose_paths)

    def __getitem__(self, idx):
        intrinsics, _, _, _ = util.parse_intrinsics(os.path.join(self.instance_dir, "intrinsics.txt"),
                                                                  trgt_sidelength=self.img_sidelength)
        intrinsics = torch.Tensor(intrinsics).float()

        rgb = data_util.load_rgb(self.color_paths[idx], sidelength=self.img_sidelength)
        pose = data_util.load_pose(self.pose_paths[idx])

        idx2 = np.random.randint(len(self.pose_paths))
        rgb2 = data_util.load_rgb(self.color_paths[idx2], sidelength=self.img_sidelength)
        pose2 = data_util.load_pose(self.pose_paths[idx2])

        noise = np.random.randn(*rgb2.transpose(1, 2, 0).shape)

        def logsnr_schedule_cosine(t, *, logsnr_min=-20., logsnr_max=20.):
            b = np.arctan(np.exp(-.5 * logsnr_max))
            a = np.arctan(np.exp(-.5 * logsnr_min)) - b
            return -2. * np.log(np.tan(a * t + b))

        time_step = np.random.randint(0, 1000)
        z = self.sqrt_alphas_cumprod[time_step] * rgb2.transpose(1, 2, 0) + self.sqrt_one_minus_alphas_cumprod[time_step] * noise

        sample = {
            "x": torch.from_numpy(rgb.transpose(1, 2, 0)).float(),
            "z": z,
            "R1": torch.from_numpy(pose[:3, :3]).float(),
            "R2": torch.from_numpy(pose2[:3, :3]).float(),
            "t1": torch.from_numpy(pose[:3, -1]).float(),
            "t2": torch.from_numpy(pose2[:3, -1]).float(),
            "K": intrinsics[:3, :3],
            "logsnr": logsnr_schedule_cosine(time_step / 1000.0),
            "noise": torch.from_numpy(noise)
        }
        return sample


class SceneClassDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 root_dir,
                 img_sidelength=None,
                 max_num_instances=-1,
                 max_observations_per_instance=-1,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 samples_per_instance=2):

        self.samples_per_instance = samples_per_instance
        self.instance_dirs = sorted(glob(os.path.join(root_dir, "*/")))

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        if max_num_instances != -1:
            self.instance_dirs = self.instance_dirs[:max_num_instances]

        self.all_instances = [SceneInstanceDataset(instance_idx=idx,
                                                   instance_dir=dir,
                                                   specific_observation_idcs=specific_observation_idcs,
                                                   img_sidelength=img_sidelength,
                                                   num_images=max_observations_per_instance)
                              for idx, dir in enumerate(self.instance_dirs)]

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend( [bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                   ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)

    def __getitem__(self, idx):
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        for i in range(self.samples_per_instance - 1):
            observations.append(self.all_instances[obj_idx][np.random.randint(len(self.all_instances[obj_idx]))])

        ground_truth = [{'noise':ray_bundle['noise']} for ray_bundle in observations]

        return observations, ground_truth