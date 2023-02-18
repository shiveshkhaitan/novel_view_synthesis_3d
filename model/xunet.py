import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp
import visu3d as v3d

from typing import Optional

nonlinearity = nn.swish

def out_init_scale():
	return nn.initializers.variance_scaling(0., 'fan_in', 'truncated_normal')

def nearest_neighbor_upsample(h):
	B, F, H, W, C = h.shape
	h = h.reshape(B, F, H, 1, W, 1, C)
	h = jnp.broadcast_to(h, (B, F, H, 2, W, 2, C))
	return h.reshape(B, F, H * 2, W * 2, C)

def avgpool_downsample(h, k=2):
	return nn.avg_pool(h, (1, k, k), (1, k, k))

def posenc_ddpm(timesteps, emb_ch: int, max_time=1000., dtype=jnp.float32):
	"""Positional encodings for noise levels, following DDPM."""
	# 1000 is the magic number from DDPM. With different timesteps, we
	# normalize by the number of steps but still multiply by 1000.
	timesteps *= (1000. / max_time)
	half_dim = emb_ch // 2
	# 10000 is the magic number from transformers.
	emb = onp.log(10000) / (half_dim - 1)
	emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
	emb = emb.reshape(*([1] * (timesteps.ndim - 1)), emb.shape[-1])
	emb = timesteps.astype(dtype)[..., None] * emb
	emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
	return emb

def posenc_nerf(x, min_deg=0, max_deg=15):
	"""Concatenate x and its positional encodings, following NeRF."""
	if min_deg == max_deg:
		return x
	scales = jnp.array([2**i for i in range(min_deg, max_deg)])
	xb = jnp.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
	emb = jnp.sin(jnp.concatenate([xb, xb + onp.pi / 2.], axis=-1))
	return jnp.concatenate([x, emb], axis=-1)

class GroupNorm(nn.Module):
	"""Group normalization, applied over frames."""
	@nn.compact
	def __call__(self, h):
		B, F, H, W, C = h.shape
		h = nn.GroupNorm(num_groups=32)(h.reshape(B,2, H, W, C))
		return h.reshape(B, 2, H, W, C)

class FiLM(nn.Module):
	"""Feature-wise linear modulation."""
	features: int
	@nn.compact
	def __call__(self, h, emb):
		emb = nn.Dense(2 * self.features)(nonlinearity(emb))
		scale, shift = jnp.split(emb, 2, axis=-1)
		return h * (1. + scale) + shift

class ResnetBlock(nn.Module):
	"""BigGAN-style residual block, applied over frames."""
	features: Optional[int] = None
	dropout: float = 0.
	resample: Optional[str] = None

	@nn.compact
	def __call__(self, h_in, emb, *, train: bool):
		B, _, _, _, C = h_in.shape
		features = C if self.features is None else self.features
		h = nonlinearity(GroupNorm()(h_in))
		if self.resample is not None:
			updown = {
			'up': nearest_neighbor_upsample,
			'down': avgpool_downsample,
			}[self.resample]
			h = updown(h)
			h_in = updown(h_in)
		h = nn.Conv(features, kernel_size=(1, 3, 3), strides=(1, 1, 1))(h)
		h = FiLM(features=features)(GroupNorm()(h), emb)
		h = nonlinearity(h)
		h = nn.Dropout(rate=self.dropout)(h, deterministic=not train)
		h = nn.Conv(
					features,
					kernel_size=(1, 3, 3),
					strides=(1, 1, 1),
					kernel_init=out_init_scale())(h)
		if C != features:
			h_in = nn.Dense(features)(h_in)
		return (h + h_in) / onp.sqrt(2)

class AttnLayer(nn.Module):
	attn_heads: int = 4
	@nn.compact
	def __call__(self, *, q, kv):
		C = q.shape[-1]
		head_dim = C // self.attn_heads
		q = nn.DenseGeneral((self.attn_heads, head_dim))(q)
		k = nn.DenseGeneral((self.attn_heads, head_dim))(kv)
		v = nn.DenseGeneral((self.attn_heads, head_dim))(kv)
		return nn.dot_product_attention(q, k, v)

class AttnBlock(nn.Module):
	attn_type: str
	attn_heads: int = 4
	@nn.compact
	def __call__(self, h_in):
		B, F, H, W, C = h_in.shape
		h = GroupNorm()(h_in)
		h0 = h[:, 0].reshape(B, H * W, C)
		h1 = h[:, 1].reshape(B, H * W, C)
		attn_layer = AttnLayer(attn_heads=self.attn_heads)
		if self.attn_type == 'self':
			h0 = attn_layer(q=h0, kv=h0)
			h1 = attn_layer(q=h1, kv=h1)
		elif self.attn_type == 'cross':
			original_h0 = h0
			h0 = attn_layer(q=h0, kv=h1)
			h1 = attn_layer(q=h1, kv=original_h0)
		else:
			raise NotImplementedError(self.attn_type)
		h = jnp.stack([h0, h1], axis=1)
		h = h.reshape(B, F, H, W, -1)
		# h = nn.DenseGeneral(C, axis=(-2, -1), kernel_init=out_init_scale())(h)
		return (h + h_in) / onp.sqrt(2)

class XUNetBlock(nn.Module):
	features: int
	use_attn: bool = False
	attn_heads: int = 4
	dropout: float = 0.
	@nn.compact
	def __call__(self, x, emb, *, train: bool):
		h = ResnetBlock(features=self.features, dropout=self.dropout)(x, emb, train=train)
		if self.use_attn:
			h = AttnBlock(attn_type='self', attn_heads=self.attn_heads)(h)
			h = AttnBlock(attn_type='cross', attn_heads=self.attn_heads)(h)
		return h

class ConditioningProcessor(nn.Module):
	"""Process conditioning inputs into embeddings."""
	emb_ch: int
	num_resolutions: int
	use_pos_emb: bool = True
	use_ref_pose_emb: bool = True

	@nn.compact
	def __call__(self, batch, cond_mask):
		B, H, W, C = batch['x'].shape
		# Log signal-to-noise-ratio embedding.
		logsnr = jnp.clip(batch['logsnr'], -20., 20.)
		logsnr = 2. * jnp.arctan(jnp.exp(-logsnr / 2.)) / onp.pi
		logsnr_emb = posenc_ddpm(logsnr, emb_ch=self.emb_ch, max_time=1.)
		logsnr_emb = nn.Dense(self.emb_ch)(logsnr_emb)
		logsnr_emb = nn.Dense(self.emb_ch)(nonlinearity(logsnr_emb))
		# Pose embeddings.
		world_from_cam1 = v3d.Transform(R=batch['R1'], t=batch['t1'])
		cam_spec1 = v3d.PinholeCamera(resolution=(H, W), K=batch['K'])
		rays1 = v3d.Camera(spec=cam_spec1, world_from_cam=world_from_cam1).rays()
		pose_emb_pos1 = posenc_nerf(rays1.pos, min_deg=0, max_deg=15)
		pose_emb_dir1 = posenc_nerf(rays1.dir, min_deg=0, max_deg=8)
		pose_emb1 = jnp.concatenate([pose_emb_pos1, pose_emb_dir1], axis=-1)

		world_from_cam2 = v3d.Transform(R=batch['R2'], t=batch['t2'])
		cam_spec2 = v3d.PinholeCamera(resolution=(H, W), K=batch['K'])
		rays2 = v3d.Camera(spec=cam_spec2, world_from_cam=world_from_cam2).rays()
		pose_emb_pos2 = posenc_nerf(rays2.pos, min_deg=0, max_deg=15)
		pose_emb_dir2 = posenc_nerf(rays2.dir, min_deg=0, max_deg=8)
		pose_emb2 = jnp.concatenate([pose_emb_pos2, pose_emb_dir2], axis=-1)

		pose_emb = jnp.concatenate([jnp.expand_dims(pose_emb1, axis=1), jnp.expand_dims(pose_emb2, axis=1)], axis=1)
		# Enable classifier-free guidance over poses.
		D = pose_emb.shape[-1]
		assert cond_mask.shape == (B,)

		cond_mask = cond_mask[:, None, None, None, None]
		pose_emb = jnp.where(cond_mask, pose_emb, jnp.zeros_like(pose_emb))
		# Learnable position embeddings over (H, W) of frames (optional).
		if self.use_pos_emb:
			pos_emb = self.param('pos_emb', nn.initializers.normal(stddev=1. / onp.sqrt(D)), (H, W, D), pose_emb.dtype)
			pose_emb += pos_emb[None, None]

		# Binary embedding to let the model distinguish frames (optional).
		if self.use_ref_pose_emb:
			first_emb = self.param('ref_pose_emb_first', nn.initializers.normal(stddev=1. / onp.sqrt(D)), (D,),
						pose_emb.dtype)[None, None, None, None]
			other_emb = self.param(
					'ref_pose_emb_other',
					nn.initializers.normal(stddev=1. / onp.sqrt(D)),
											(D,),
											pose_emb.dtype)[None, None, None, None]
			pose_emb += jnp.concatenate([first_emb, other_emb], axis=1)
			# Downsample ray embeddings for each UNet resolution.

		pose_embs = []
		for i_level in range(self.num_resolutions):
			pose_embs.append(nn.Conv(
									features=self.emb_ch,
									kernel_size=(1, 3, 3),
									strides=(1, 2 ** i_level, 2 ** i_level))(pose_emb))
		return logsnr_emb, pose_embs

class XUNet(nn.Module):
	"""Our proposed XUNet architecture."""
	ch: int = 32
	ch_mult = (1, 2,)
	emb_ch: int = 32
	num_res_blocks: int = 2
	attn_resolutions = (8, 16, 32)
	attn_heads: int = 4
	dropout: float = 0.1
	use_pos_emb: bool = False
	use_ref_pose_emb: bool = False

	@nn.compact
	def __call__(self, batch: dict, *, cond_mask: jnp.ndarray, train: bool):
		B, H, W, C = batch['x'].shape
		num_resolutions = len(self.ch_mult)
		logsnr_emb, pose_embs = ConditioningProcessor(
			emb_ch=self.emb_ch,
			num_resolutions=num_resolutions,
			use_pos_emb=self.use_pos_emb,
			use_ref_pose_emb=self.use_ref_pose_emb)(batch, cond_mask)
		del cond_mask

		h = jnp.stack([batch['x'], batch['z']], axis=1)
		h = nn.Conv(self.ch, kernel_size=(1, 3, 3), strides=(1, 1, 1))(h)
		# Downsampling.
		hs = [h]
		for i_level in range(num_resolutions):
			emb = jnp.expand_dims(logsnr_emb[..., None, None, :], axis=1) + pose_embs[i_level]
			for i_block in range(self.num_res_blocks):
				use_attn = h.shape[2] in self.attn_resolutions
				h = XUNetBlock(
					features=self.ch * self.ch_mult[i_level],
					dropout=self.dropout,
					attn_heads=self.attn_heads,
					use_attn=use_attn)(h, emb, train=train)
				hs.append(h)

			if i_level != num_resolutions - 1:
				emb = jnp.expand_dims(logsnr_emb[..., None, None, :], axis=1) + pose_embs[i_level + 1]
				h = ResnetBlock(dropout=self.dropout, resample='down')(h, emb, train=train)
				hs.append(h)

		# Middle.
		emb = jnp.expand_dims(logsnr_emb[..., None, None, :], axis=1) + pose_embs[-1]
		use_attn = h.shape[2] in self.attn_resolutions
		h = XUNetBlock(
			features=self.ch * self.ch_mult[i_level],
			dropout=self.dropout,
			attn_heads=self.attn_heads,
			use_attn=use_attn)(h, emb, train=train)
		# Upsampling.
		for i_level in reversed(range(num_resolutions)):
			emb = jnp.expand_dims(logsnr_emb[..., None, None, :], axis=1) + pose_embs[i_level]

			for i_block in range(self.num_res_blocks + 1):
				use_attn = hs[-1].shape[2] in self.attn_resolutions
				h = jnp.concatenate([h, hs.pop()], axis=-1)
				h = XUNetBlock(
					features=self.ch * self.ch_mult[i_level],
					dropout=self.dropout,
					attn_heads=self.attn_heads,
					use_attn=use_attn)(h, emb, train=train)

			if i_level != 0:
				emb = jnp.expand_dims(logsnr_emb[..., None, None, :], axis=1) + pose_embs[i_level - 1]
				h = ResnetBlock(dropout=self.dropout, resample='up')(h, emb, train=train)
			# End.

		assert not hs
		h = nonlinearity(GroupNorm()(h))
		return nn.Conv(
			C,
			kernel_size=(1, 3, 3),
			strides=(1, 1, 1),
			kernel_init=out_init_scale())(h)[:, 1]
