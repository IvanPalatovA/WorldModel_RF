import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from reportlab.lib.utils import ImageReader

import gymnasium as gym
import minigrid  # registers envs
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import imageio

from transformers import CLIPModel, CLIPProcessor, logging as hf_logging

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int):
    env = gym.make(env_id, render_mode="rgb_array")
    # Provide RGB image observation directly (uint8 HxWx3)
    env = RGBImgObsWrapper(env)   # adds 'image' in dict
    env = ImgObsWrapper(env)      # makes observation just the image array
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def obs_to_tensor(obs, device: torch.device) -> torch.Tensor:
    """
    obs: np.ndarray HxWx3 uint8 OR torch.Tensor HxWx3 (or CHW) -> returns 1,C,64,64 float32 on device
    """
    if isinstance(obs, torch.Tensor):
        x = obs
        if x.dtype != torch.float32:
            x = x.float()
        # If uint8-like range
        if x.max() > 1.0:
            x = x / 255.0
        # HWC -> CHW if needed
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)  # 1,C,H,W
        x = x.to(device)
    else:
        # numpy
        x = torch.from_numpy(obs).to(device).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W

    if x.shape[-2:] != (64, 64):
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
    return x


def batch_obs_to_tensor(obs_batch, device: torch.device) -> torch.Tensor:
    """
    obs_batch: np.ndarray (B,H,W,3) uint8 OR torch.Tensor (B,H,W,3) -> returns B,C,64,64 float32 on device
    """
    if isinstance(obs_batch, torch.Tensor):
        x = obs_batch
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        # BHWC -> BCHW
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.to(device)
    else:
        x = torch.from_numpy(obs_batch).to(device).float() / 255.0
        x = x.permute(0, 3, 1, 2)  # B,C,H,W

    if x.shape[-2:] != (64, 64):
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
    return x


def save_gif(frames: List[np.ndarray], path: str, fps: int = 10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


# =========================
# Dataset: sequences for RSSM training
# =========================

class TransitionSeqDataset(Dataset):
    """
    Sequences of length seq_len that DO NOT cross episode terminals (done=True inside the window).
    This stabilizes RSSM training for imagined rollouts.
    """

    def __init__(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, seq_len: int):
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.dones = dones.astype(np.bool_)
        self.seq_len = seq_len
        self.N = obs.shape[0]
        if self.N <= seq_len + 1:
            raise ValueError("Not enough data for sequences. Collect more transitions or reduce seq_len.")

        # Valid start indices such that dones[s:s+seq_len-1] has no terminal inside.
        # We allow done at the last element of the window (it is still a valid target).
        valid = []
        max_start = self.N - seq_len
        for s in range(max_start):
            window_dones = self.dones[s : s + seq_len - 1]
            if not window_dones.any():
                valid.append(s)

        if len(valid) == 0:
            raise ValueError("No valid sequences without terminals. Collect more data or reduce seq_len.")

        self.valid_starts = np.array(valid, dtype=np.int64)

    def __len__(self):
        return int(self.valid_starts.shape[0])

    def __getitem__(self, idx):
        s = int(self.valid_starts[idx])
        e = s + self.seq_len
        return (
            self.obs[s:e],
            self.actions[s:e],
            self.rewards[s:e],
            self.dones[s:e],
        )


# =========================
# RSSM components (Dreamer/PlaNet style)
# =========================

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128 * 6 * 6, embed_dim)

    def forward(self, x):
        y = self.net(x)
        return self.fc(y)


class ConvDecoder(nn.Module):
    def __init__(self, state_dim: int, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(state_dim, 128 * 6 * 6)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 6, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, s):
        y = self.fc(s).view(-1, 128, 6, 6)
        x = self.net(y)
        if x.shape[-1] != 64:
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        return x


class RSSM(nn.Module):
    """
    Minimal RSSM:
      deterministic GRU state h_t
      stochastic latent z_t (diag Gaussian)
      prior p(z_t|h_t), posterior q(z_t|h_t, enc(o_t))
      decoder recon(o_t|h_t,z_t)
      reward head and continue head (for WM-planning baseline objective)
    """
    def __init__(self, action_dim: int, h_dim=256, z_dim=32, embed_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.encoder = ConvEncoder(3, embed_dim)
        self.gru = nn.GRUCell(input_size=z_dim + action_dim, hidden_size=h_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, 256), nn.ReLU(),
            nn.Linear(256, 2 * z_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(h_dim + embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 2 * z_dim),
        )

        state_dim = h_dim + z_dim
        self.decoder = ConvDecoder(state_dim, out_channels=3)

        self.reward_head = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.continue_head = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def init_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    def _dist_params(self, net_out: torch.Tensor):
        mean, logstd = torch.chunk(net_out, 2, dim=-1)
        logstd = torch.clamp(logstd, -5, 2)
        std = torch.exp(logstd)
        return mean, std

    def _sample(self, mean: torch.Tensor, std: torch.Tensor):
        eps = torch.randn_like(mean)
        return mean + std * eps

    def prior(self, h: torch.Tensor):
        mean, std = self._dist_params(self.prior_net(h))
        return mean, std

    def posterior(self, h: torch.Tensor, embed: torch.Tensor):
        mean, std = self._dist_params(self.post_net(torch.cat([h, embed], dim=-1)))
        return mean, std

    def observe_step(self, h: torch.Tensor, z_prev: torch.Tensor, a_prev_oh: torch.Tensor, obs: torch.Tensor):
        h = self.gru(torch.cat([z_prev, a_prev_oh], dim=-1), h)
        embed = self.encoder(obs)
        post_mean, post_std = self.posterior(h, embed)
        z = self._sample(post_mean, post_std)
        prior_mean, prior_std = self.prior(h)
        return h, z, (prior_mean, prior_std, post_mean, post_std)

    def imagine_step(self, h: torch.Tensor, z_prev: torch.Tensor, a_prev_oh: torch.Tensor):
        h = self.gru(torch.cat([z_prev, a_prev_oh], dim=-1), h)
        prior_mean, prior_std = self.prior(h)
        z = self._sample(prior_mean, prior_std)
        return h, z, (prior_mean, prior_std)

    def decode(self, h: torch.Tensor, z: torch.Tensor):
        s = torch.cat([h, z], dim=-1)
        return self.decoder(s)

    def predict_reward_continue(self, h: torch.Tensor, z: torch.Tensor):
        s = torch.cat([h, z], dim=-1)
        r = self.reward_head(s).squeeze(-1)
        c_logits = self.continue_head(s).squeeze(-1)  # ← логиты, не вероятности
        return r, c_logits

    @staticmethod
    def kl_diag_gauss(p_mean, p_std, q_mean, q_std):
        var_ratio = (q_std / p_std) ** 2
        t1 = ((q_mean - p_mean) / p_std) ** 2
        kl = 0.5 * (var_ratio + t1 - 1 - torch.log(var_ratio + 1e-8))
        return kl.sum(dim=-1)


# =========================
# VLM-based scorer (CLIP)
# =========================

class CLIPScorer:
    def __init__(self, device: torch.device, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = device
        # In Colab, explicit token=False prevents noisy secret lookup warnings if HF_TOKEN is not set.
        hf_token = os.getenv("HF_TOKEN") or False

        # Compatibility across transformers versions: some use token=..., some use use_auth_token=...
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False, token=hf_token)
        except TypeError:
            if hf_token:
                self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False, use_auth_token=hf_token)
            else:
                self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)

        prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            try:
                self.model = CLIPModel.from_pretrained(model_name, token=hf_token).to(device)
            except TypeError:
                if hf_token:
                    self.model = CLIPModel.from_pretrained(model_name, use_auth_token=hf_token).to(device)
                else:
                    self.model = CLIPModel.from_pretrained(model_name).to(device)
        finally:
            hf_logging.set_verbosity(prev_verbosity)
        self.model.eval()

    @torch.no_grad()
    def score_images(self, images: List[np.ndarray], text: str, batch_size: int = 64) -> torch.Tensor:
        if len(images) == 0:
            return torch.empty(0, dtype=torch.float32)
        scores = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            pil_images = [Image.fromarray(img) for img in chunk]
            # One shared text prompt for all imagined frames -> logits_per_image shape [B, 1].
            inputs = self.processor(text=[text], images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[:, 0]  # [B]
            scores.append(torch.tanh(logits / 10.0).detach().cpu())
        return torch.cat(scores, dim=0)


# =========================
# Data collection
# =========================

@dataclass
class CollectConfig:
    env_id: str = "MiniGrid-DoorKey-6x6-v0"
    seed: int = 0
    steps: int = 80_000
    max_ep_len: int = 256


def collect_random_data(cfg: CollectConfig) -> Dict[str, np.ndarray]:
    env = make_env(cfg.env_id, cfg.seed)
    obs_list, act_list, rew_list, done_list = [], [], [], []

    obs, info = env.reset(seed=cfg.seed)
    ep_len = 0

    for _ in tqdm(range(cfg.steps), desc="Collect random data"):
        a = env.action_space.sample()
        next_obs, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        obs_list.append(obs)
        act_list.append(a)
        rew_list.append(float(r))
        done_list.append(bool(done))

        obs = next_obs
        ep_len += 1
        if done or ep_len >= cfg.max_ep_len:
            obs, info = env.reset()
            ep_len = 0

    env.close()
    return {
        "obs": np.stack(obs_list).astype(np.uint8),
        "actions": np.array(act_list, dtype=np.int64),
        "rewards": np.array(rew_list, dtype=np.float32),
        "dones": np.array(done_list, dtype=np.bool_),
    }


# =========================
# Train RSSM
# =========================

@dataclass
class TrainConfig:
    seq_len: int = 32
    batch_size: int = 32
    epochs: int = 12
    lr: float = 3e-4
    kl_beta: float = 1.0
    recon_scale: float = 1.0
    reward_scale: float = 1.0
    continue_scale: float = 1.0


def one_hot_actions(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    return F.one_hot(actions.long(), num_classes=action_dim).float()


def train_rssm(
    rssm: RSSM,
    data: Dict[str, np.ndarray],
    cfg: TrainConfig,
    device: torch.device,
    use_amp: bool = False,
    early_stop_patience: int = 2,
    min_delta: float = 1e-3
) -> RSSM:
    ds = TransitionSeqDataset(data["obs"], data["actions"], data["rewards"], data["dones"], seq_len=cfg.seq_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)

    opt = torch.optim.Adam(rssm.parameters(), lr=cfg.lr)

    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device_type, enabled=(use_amp and device.type == "cuda"))

    rssm.train()

    best = float("inf")
    bad_epochs = 0

    for ep in range(cfg.epochs):
        losses = []
        for obs_seq, act_seq, rew_seq, done_seq in tqdm(dl, desc=f"Train epoch {ep+1}/{cfg.epochs}", leave=False):
            B, L = obs_seq.shape[0], obs_seq.shape[1]
            H, W = obs_seq.shape[2], obs_seq.shape[3]

            obs_t = batch_obs_to_tensor(
                obs_seq.reshape(B * L, H, W, 3),
                device
            ).reshape(B, L, 3, 64, 64)

            act_t = act_seq.to(device).long()     # [B,L]
            rew_t = rew_seq.to(device).float()    # [B,L]
            done_t = done_seq.to(device).float()  # [B,L]
            cont_t = 1.0 - done_t                 # [B,L]

            h, z = rssm.init_state(B, device)
            a_prev = torch.zeros(B, dtype=torch.long, device=device)
            a_prev_oh = one_hot_actions(a_prev, rssm.action_dim)

            recon_loss = 0.0
            reward_loss = 0.0
            cont_loss = 0.0
            kl_loss = 0.0

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(amp_device_type, enabled=(use_amp and device.type == "cuda")):
                for t in range(L):
                    h, z, (pr_m, pr_s, po_m, po_s) = rssm.observe_step(h, z, a_prev_oh, obs_t[:, t])

                    recon = rssm.decode(h, z)
                    r_pred, c_pred = rssm.predict_reward_continue(h, z)

                    recon_loss = recon_loss + F.mse_loss(recon, obs_t[:, t], reduction="mean")
                    reward_loss = reward_loss + F.mse_loss(r_pred, rew_t[:, t], reduction="mean")
                    cont_loss = cont_loss + F.binary_cross_entropy_with_logits(c_pred, cont_t[:, t], reduction="mean")
                    kl_loss = kl_loss + RSSM.kl_diag_gauss(pr_m, pr_s, po_m, po_s).mean()

                    a_prev = act_t[:, t]
                    a_prev_oh = one_hot_actions(a_prev, rssm.action_dim)

                loss = (
                    cfg.recon_scale * recon_loss +
                    cfg.reward_scale * reward_loss +
                    cfg.continue_scale * cont_loss +
                    cfg.kl_beta * kl_loss
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
            scaler.step(opt)
            scaler.update()

            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        print(f"[RSSM] epoch {ep+1}: loss={mean_loss:.4f}")

        # Early stopping
        if mean_loss < best - min_delta:
            best = mean_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                print(f"[RSSM] Early stopping at epoch {ep+1} (best={best:.4f})")
                break

    rssm.eval()
    return rssm


# =========================
# Planning (MPC random shooting)
# =========================

@dataclass
class PlanConfig:
    horizon: int = 15
    num_candidates: int = 256
    gamma: float = 0.99
    vlm_batch_size: int = 64
    vlm_score_stride: int = 2

@torch.no_grad()
def mpc_action(
    rssm: RSSM,
    h: torch.Tensor,
    z: torch.Tensor,
    action_dim: int,
    plan_cfg: PlanConfig,
    device: torch.device,
    mode: str,
    scorer: Optional[CLIPScorer] = None,
    goal_text: str = "",
) -> int:
    """
    mode:
      - "wm_reward": planner objective uses world-model predicted reward
      - "wm_vlm": planner objective uses CLIP score on imagined frames (future frames)

    NOTE: h,z are assumed to be the current posterior belief state already updated with the real obs.
    """
    assert mode in ["wm_reward", "wm_vlm"]
    if mode == "wm_vlm":
        assert scorer is not None and goal_text

    H = plan_cfg.horizon
    K = plan_cfg.num_candidates
    action_seqs = torch.randint(low=0, high=action_dim, size=(K, H), device=device)
    objectives = torch.zeros(K, device=device)

    all_images: List[np.ndarray] = []
    img_index_map: List[Tuple[int, int]] = []

    for k in range(K):
        hk = h.clone()
        zk = z.clone()
        G = 0.0
        discount = 1.0

        for t in range(H):
            a_t = action_seqs[k, t].view(1)
            a_t_oh = one_hot_actions(a_t, action_dim)
            hk, zk, _ = rssm.imagine_step(hk, zk, a_t_oh)

            if mode == "wm_reward":
                r_pred, _ = rssm.predict_reward_continue(hk, zk)
                G += discount * float(r_pred.item())
            else:
                # Subsample imagined frames for VLM scoring to keep MPC responsive.
                if (t % plan_cfg.vlm_score_stride == 0) or (t == H - 1):
                    recon = rssm.decode(hk, zk)
                    img = (recon.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
                    all_images.append(img)
                    img_index_map.append((k, t))

            discount *= plan_cfg.gamma

        if mode == "wm_reward":
            objectives[k] = G

    if mode == "wm_vlm":
        scores = scorer.score_images(all_images, goal_text, batch_size=plan_cfg.vlm_batch_size).numpy().reshape(-1)
        for idx, (k, t) in enumerate(img_index_map):
            objectives[k] += (plan_cfg.gamma ** t) * float(scores[idx])

    best_k = int(torch.argmax(objectives).item())
    return int(action_seqs[best_k, 0].item())


# =========================
# Evaluation
# =========================

@dataclass
class EvalConfig:
    env_id: str = "MiniGrid-DoorKey-6x6-v0"
    seeds: List[int] = None
    episodes_per_seed: int = 5
    max_ep_len: int = 256
    save_gifs: bool = True
    out_dir: str = "outputs_doorkey"
    goal_text: str = "agent opened the door and reached the goal"


def run_episode(env, policy_fn, max_ep_len: int, reset_seed: Optional[int] = None) -> Tuple[float, bool, List[np.ndarray]]:
    frames = []
    obs, info = env.reset(seed=reset_seed)
    ep_return = 0.0
    prev_action = 0

    if hasattr(policy_fn, "reset"):
        policy_fn.reset()

    for _ in range(max_ep_len):
        frames.append(env.render())
        a = policy_fn(obs, prev_action)
        obs, r, terminated, truncated, info = env.step(a)
        prev_action = a
        ep_return += float(r)
        if terminated or truncated:
            break

    success = ep_return > 0.0
    return ep_return, success, frames

def make_random_policy(env):
    def _pi(_obs, _prev_a):
        return env.action_space.sample()
    return _pi

def make_wm_reward_policy(rssm: RSSM, env, plan_cfg: PlanConfig, device: torch.device):
    class _Policy:
        def __init__(self):
            self.h, self.z = rssm.init_state(1, device)

        def reset(self):
            self.h, self.z = rssm.init_state(1, device)

        @torch.no_grad()
        def __call__(self, obs: np.ndarray, prev_a: int) -> int:
            x = obs_to_tensor(obs, device)
            a0 = torch.tensor([prev_a], dtype=torch.long, device=device)
            a0_oh = one_hot_actions(a0, env.action_space.n)
            self.h, self.z, _ = rssm.observe_step(self.h, self.z, a0_oh, x)

            return mpc_action(
                rssm=rssm,
                h=self.h,
                z=self.z,
                action_dim=env.action_space.n,
                plan_cfg=plan_cfg,
                device=device,
                mode="wm_reward",
            )

    return _Policy()

def make_wm_vlm_policy(
    rssm: RSSM,
    scorer: CLIPScorer,
    env,
    plan_cfg: PlanConfig,
    device: torch.device,
    goal_text: str
):
    class _Policy:
        def __init__(self):
            self.h, self.z = rssm.init_state(1, device)

        def reset(self):
            self.h, self.z = rssm.init_state(1, device)

        @torch.no_grad()
        def __call__(self, obs: np.ndarray, prev_a: int) -> int:
            x = obs_to_tensor(obs, device)
            a0 = torch.tensor([prev_a], dtype=torch.long, device=device)
            a0_oh = one_hot_actions(a0, env.action_space.n)
            self.h, self.z, _ = rssm.observe_step(self.h, self.z, a0_oh, x)

            return mpc_action(
                rssm=rssm,
                h=self.h,
                z=self.z,
                action_dim=env.action_space.n,
                plan_cfg=plan_cfg,
                device=device,
                mode="wm_vlm",
                scorer=scorer,
                goal_text=goal_text,
            )

    return _Policy()

# ПОСЛЕ (ЗАМЕНА ФУНКЦИИ evaluate_all)
def evaluate_all(
    rssm: RSSM,
    scorer: CLIPScorer,
    device: torch.device,
    plan_cfg: PlanConfig,
    eval_cfg: EvalConfig
) -> Dict[str, Dict[str, float]]:

    if eval_cfg.seeds is None:
        eval_cfg.seeds = [0, 1, 2]

    methods = ["random", "wm_reward", "wm_vlm"]
    stats = {m: {"returns": [], "success": []} for m in methods}
    total_runs = len(eval_cfg.seeds) * eval_cfg.episodes_per_seed * len(methods)
    pbar = tqdm(total=total_runs, desc="Evaluate policies")

    for seed in eval_cfg.seeds:
        env = make_env(eval_cfg.env_id, seed)

        # политики (RSSM-belief state) создаём 1 раз на env и reset() на каждый эпизод
        wm_reward_pi = make_wm_reward_policy(rssm, env, plan_cfg, device)
        wm_vlm_pi = make_wm_vlm_policy(rssm, scorer, env, plan_cfg, device, goal_text=eval_cfg.goal_text)

        for ep in range(eval_cfg.episodes_per_seed):
            ep_seed = int(seed * 10_000 + ep)

            # Random baseline
            random_pi = make_random_policy(env)
            R, S, frames = run_episode(env, random_pi, eval_cfg.max_ep_len, reset_seed=ep_seed)
            stats["random"]["returns"].append(R)
            stats["random"]["success"].append(S)
            if eval_cfg.save_gifs:
                save_gif(frames, os.path.join(eval_cfg.out_dir, "gifs", f"random_seed{seed}_ep{ep}.gif"))
            pbar.update(1)

            # World-model planning WITHOUT VLM (objective: predicted reward)
            R, S, frames = run_episode(env, wm_reward_pi, eval_cfg.max_ep_len, reset_seed=ep_seed)
            stats["wm_reward"]["returns"].append(R)
            stats["wm_reward"]["success"].append(S)
            if eval_cfg.save_gifs:
                save_gif(frames, os.path.join(eval_cfg.out_dir, "gifs", f"wm_reward_seed{seed}_ep{ep}.gif"))
            pbar.update(1)

            # World-model planning + VLM scorer
            R, S, frames = run_episode(env, wm_vlm_pi, eval_cfg.max_ep_len, reset_seed=ep_seed)
            stats["wm_vlm"]["returns"].append(R)
            stats["wm_vlm"]["success"].append(S)
            if eval_cfg.save_gifs:
                save_gif(frames, os.path.join(eval_cfg.out_dir, "gifs", f"wm_vlm_seed{seed}_ep{ep}.gif"))
            pbar.update(1)

        env.close()
    pbar.close()

    results = {}
    for m in methods:
        returns = np.array(stats[m]["returns"], dtype=np.float32)
        success = np.array(stats[m]["success"], dtype=np.float32)
        results[m] = {
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "success_rate": float(success.mean()),
            "num_episodes": int(len(returns)),
        }

    return results


# =========================
# PDF report
# =========================

def first_frame_from_gif(gif_path: str) -> Image.Image:
    rdr = imageio.get_reader(gif_path)
    frame0 = rdr.get_data(0)
    rdr.close()
    return Image.fromarray(frame0)

def write_pdf_report(results: Dict[str, Dict[str, float]], eval_cfg: EvalConfig, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    c = canvas.Canvas(out_path, pagesize=A4)
    w, h = A4
    y = h - 60

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Demo: RSSM world model + MPC + VLM-based scorer (CLIP)")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Environment: {eval_cfg.env_id}")
    y -= 16
    c.drawString(50, y, f"Goal text: {eval_cfg.goal_text}")
    y -= 16
    c.drawString(50, y, f"Seeds: {eval_cfg.seeds}, episodes/seed: {eval_cfg.episodes_per_seed}")
    y -= 26

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Method")
    c.drawString(250, y, "Mean Return")
    c.drawString(360, y, "Std")
    c.drawString(430, y, "Success Rate")
    y -= 14
    c.line(50, y, 540, y)
    y -= 14

    c.setFont("Helvetica", 11)
    order = ["random", "wm_reward", "wm_vlm"]
    names = {
        "random": "Random",
        "wm_reward": "World-model MPC (no VLM, predicted reward objective)",
        "wm_vlm": "World-model MPC + VLM scorer objective",
    }
    for k in order:
        r = results[k]
        c.drawString(50, y, names[k])
        c.drawString(250, y, f"{r['mean_return']:.3f}")
        c.drawString(360, y, f"{r['std_return']:.3f}")
        c.drawString(450, y, f"{r['success_rate']:.3f}")
        y -= 18
        if y < 130:
            c.showPage()
            y = h - 60

    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Visuals (sample frames from GIFs)")
    y -= 10

    # pick one example per method (seed0_ep0)
    samples = {
        "Random": os.path.join(eval_cfg.out_dir, "gifs", f"random_seed{eval_cfg.seeds[0]}_ep0.gif"),
        "WM (no VLM)": os.path.join(eval_cfg.out_dir, "gifs", f"wm_reward_seed{eval_cfg.seeds[0]}_ep0.gif"),
        "WM + VLM": os.path.join(eval_cfg.out_dir, "gifs", f"wm_vlm_seed{eval_cfg.seeds[0]}_ep0.gif"),
    }

    x0 = 50
    img_w = 160
    img_h = 160
    gap = 20

    y -= 180
    for i, (title, gif_path) in enumerate(samples.items()):
        if os.path.exists(gif_path):
            img = first_frame_from_gif(gif_path)
            img_reader = ImageReader(img)
            xi = x0 + i * (img_w + gap)
            c.setFont("Helvetica", 10)
            c.drawString(xi, y + img_h + 8, title)
            c.drawImage(img_reader, xi, y, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
        else:
            c.drawString(x0 + i * (img_w + gap), y + img_h / 2, f"Missing: {os.path.basename(gif_path)}")
    y -= 18

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Failure modes observed (common):")
    y -= 14
    c.setFont("Helvetica", 11)
    c.drawString(60, y, "- Compounding model error: imagined frames drift away from real environment.")
    y -= 14
    c.drawString(60, y, "- CLIP misalignment: VLM may not understand MiniGrid style well; noisy scores.")
    y -= 14
    c.drawString(60, y, "- Planning budget limits: too small horizon/candidates can miss solutions.")
    y -= 18

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Future work ideas:")
    y -= 14
    c.setFont("Helvetica", 11)
    c.drawString(60, y, "- Use CEM instead of random shooting for better action search.")
    y -= 14
    c.drawString(60, y, "- Improve RSSM training: longer sequences, multi-step latent losses, bigger model.")
    y -= 14
    c.drawString(60, y, "- Try a different VLM (SigLIP/BLIP) or finetune a scorer on MiniGrid captions.")
    y -= 14
    c.drawString(60, y, "- Add uncertainty-aware planning (penalize high uncertainty regions).")

    c.save()


# =========================
# Main
# =========================

def main():
    # Device selection: MPS (Apple), then CUDA, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    # Fast settings
    use_amp = (device.type == "cuda")
    use_compile = (device.type == "cuda")

    # Auto-tune to avoid absurd runtimes on CPU
    if device.type == "cpu":
        steps = 20_000
        epochs = 3
        num_candidates = 48
        horizon = 10
        episodes_per_seed = 2
        batch_size = 16
        seq_len = 24
    elif device.type == "mps":
        steps = 35_000
        epochs = 3
        num_candidates = 96
        horizon = 12
        episodes_per_seed = 3
        batch_size = 32
        seq_len = 32
    else:  # cuda
        steps = 50_000
        epochs = 4
        # Keep planning budget moderate: wm_vlm mode is expensive because CLIP scores imagined futures.
        num_candidates = 96
        horizon = 12
        episodes_per_seed = 4
        batch_size = 64
        seq_len = 32

    collect_cfg = CollectConfig(
        env_id="MiniGrid-DoorKey-6x6-v0",
        seed=312,
        steps=steps,
        max_ep_len=256
    )

    train_cfg = TrainConfig(
        seq_len=seq_len,
        batch_size=batch_size,
        epochs=epochs,
        lr=3e-4,
        kl_beta=1.0,
        recon_scale=1.0,
        reward_scale=1.0,
        continue_scale=1.0,
    )

    plan_cfg = PlanConfig(
        horizon=horizon,
        num_candidates=num_candidates,
        gamma=0.99
    )

    eval_cfg = EvalConfig(
        env_id=collect_cfg.env_id,
        seeds=[77, 56, 99],
        episodes_per_seed=episodes_per_seed,
        max_ep_len=256,
        save_gifs=True,
        out_dir="outputs_doorkey",
        goal_text="agent has the key, opened the door, and reached the goal"
    )

    os.makedirs(eval_cfg.out_dir, exist_ok=True)

    # 1) Collect random data
    set_seed(collect_cfg.seed)
    data = collect_random_data(collect_cfg)

    # action_dim
    env_tmp = make_env(collect_cfg.env_id, collect_cfg.seed)
    action_dim = env_tmp.action_space.n
    env_tmp.close()

    # 2) Train RSSM world model
    rssm = RSSM(action_dim=action_dim, h_dim=256, z_dim=32, embed_dim=256).to(device)

    if use_compile:
        try:
            rssm = torch.compile(rssm)
            print("Using torch.compile for RSSM")
        except Exception as e:
            print("torch.compile failed, continue without it:", e)

    rssm = train_rssm(rssm, data, train_cfg, device, use_amp=use_amp)

    # 3) VLM scorer
    scorer = CLIPScorer(device=device)

    # 4) Evaluate baselines
    results = evaluate_all(rssm, scorer, device, plan_cfg, eval_cfg)

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(k, v)

    # 5) Save PDF report
    report_path = os.path.join(eval_cfg.out_dir, "report.pdf")
    write_pdf_report(results, eval_cfg, report_path)
    print(f"\nSaved report: {report_path}")
    print(f"Saved gifs:   {os.path.join(eval_cfg.out_dir, 'gifs')}")


if __name__ == "__main__":
    main()
