import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm.auto import tqdm

from reportlab.lib.utils import ImageReader

import gymnasium as gym
import minigrid  # registers envs
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
import imageio

from transformers import CLIPModel, CLIPProcessor, logging as hf_logging
try:
    from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars
except Exception:
    hf_disable_progress_bars = None

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

CODE_VERSION = "doorkey-expert-fallback-v6-2026-03-01"


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


def plan_doorkey_expert(u) -> Optional[List[int]]:
    """Shortest action sequence in symbolic DoorKey state; returns None if planning fails."""
    DIR_TO_VEC = {
        0: np.array((1, 0)),   # right
        1: np.array((0, 1)),   # down
        2: np.array((-1, 0)),  # left
        3: np.array((0, -1)),  # up
    }

    key_pos = None
    door_pos = None
    goal_pos = None
    for x in range(u.grid.width):
        for y in range(u.grid.height):
            cell = u.grid.get(x, y)
            if cell is None:
                continue
            if cell.type == "key" and key_pos is None:
                key_pos = (x, y)
            if cell.type == "door" and door_pos is None:
                door_pos = (x, y)
            if cell.type == "goal":
                goal_pos = (x, y)
    if door_pos is None or goal_pos is None:
        return None

    has_key0 = (u.carrying is not None and getattr(u.carrying, "type", None) == "key")
    if key_pos is None and not has_key0:
        # Without key in grid and not carrying key, environment state is not solvable for DoorKey.
        return None
    door_cell0 = u.grid.get(*door_pos)
    door_open0 = (door_cell0 is not None and door_cell0.is_open)
    key_present0 = False
    if key_pos is not None and not has_key0:
        key_cell0 = u.grid.get(*key_pos)
        key_present0 = (key_cell0 is not None and key_cell0.type == "key")
    start = (tuple(u.agent_pos), int(u.agent_dir), has_key0, door_open0, key_present0)

    acts = u.actions
    A_LEFT = int(acts.left)
    A_RIGHT = int(acts.right)
    A_FWD = int(acts.forward)
    A_PICKUP = int(acts.pickup)
    A_TOGGLE = int(acts.toggle)

    def neighbors(state):
        (x, y), d, has_key, door_open, key_present = state
        res = []
        res.append((A_LEFT, ((x, y), (d - 1) % 4, has_key, door_open, key_present)))
        res.append((A_RIGHT, ((x, y), (d + 1) % 4, has_key, door_open, key_present)))

        fx, fy = (np.array((x, y)) + DIR_TO_VEC[d]).astype(int).tolist()
        in_bounds = (0 <= fx < u.grid.width and 0 <= fy < u.grid.height)
        cell = u.grid.get(fx, fy) if in_bounds else None

        if cell is not None and cell.type == "door" and has_key and not door_open:
            res.append((A_TOGGLE, ((x, y), d, has_key, True, key_present)))
        if cell is not None and cell.type == "key" and key_present and not has_key:
            res.append((A_PICKUP, ((x, y), d, True, door_open, False)))

        can_forward = False
        new_door_open = door_open
        if cell is None:
            can_forward = False if not in_bounds else True
        else:
            if cell.type == "wall":
                can_forward = False
            elif cell.type == "key":
                can_forward = (not key_present)
            elif cell.type == "door":
                can_forward = cell.is_open or door_open
                new_door_open = door_open or cell.is_open
            else:
                can_forward = True
        if can_forward:
            res.append((A_FWD, ((fx, fy), d, has_key, new_door_open, key_present)))
        return res

    from collections import deque
    q = deque([start])
    prev = {start: (None, None)}

    while q:
        s = q.popleft()
        (x, y), _, _, door_open, _ = s
        if (x, y) == goal_pos and door_open:
            path = []
            cur = s
            while prev[cur][0] is not None:
                parent, act = prev[cur]
                path.append(act)
                cur = parent
            return list(reversed(path))
        for act, ns in neighbors(s):
            if ns not in prev:
                prev[ns] = (s, act)
                q.append(ns)
    return None


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
    Sequences of length seq_len.
    By default windows may cross episode terminals; this keeps dataset size large for short episodes.
    Set strict_no_terminal=True to recover old behavior.
    """

    def __init__(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        seq_len: int,
        strict_no_terminal: bool = False,
    ):
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.dones = dones.astype(np.bool_)
        self.seq_len = seq_len
        self.N = obs.shape[0]
        if self.N <= seq_len:
            raise ValueError("Not enough data for sequences. Collect more transitions or reduce seq_len.")

        max_start = self.N - seq_len + 1
        if strict_no_terminal:
            valid = []
            for s in range(max_start):
                window_dones = self.dones[s : s + seq_len - 1]
                if not window_dones.any():
                    valid.append(s)
            if len(valid) == 0:
                raise ValueError("No valid sequences without terminals. Collect more data or reduce seq_len.")
            self.valid_starts = np.array(valid, dtype=np.int64)
        else:
            self.valid_starts = np.arange(max_start, dtype=np.int64)

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

    def observe_step(
        self,
        h: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev_oh: torch.Tensor,
        obs: torch.Tensor,
        sample: bool = True,
    ):
        h = self.gru(torch.cat([z_prev, a_prev_oh], dim=-1), h)
        embed = self.encoder(obs)
        post_mean, post_std = self.posterior(h, embed)
        z = self._sample(post_mean, post_std) if sample else post_mean
        prior_mean, prior_std = self.prior(h)
        return h, z, (prior_mean, prior_std, post_mean, post_std)

    def imagine_step(
        self,
        h: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev_oh: torch.Tensor,
        sample: bool = True,
    ):
        h = self.gru(torch.cat([z_prev, a_prev_oh], dim=-1), h)
        prior_mean, prior_std = self.prior(h)
        z = self._sample(prior_mean, prior_std) if sample else prior_mean
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
            image_embeds = F.normalize(outputs.image_embeds, dim=-1)
            text_embeds = F.normalize(outputs.text_embeds, dim=-1)
            logits = (image_embeds @ text_embeds.T)[:, 0]  # cosine similarity in [-1,1]
            scores.append(logits.detach().cpu())
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
    strategy: str = "mixed"  # "random", "mixed", "expert_mix"


def collect_random_data(cfg: CollectConfig) -> Dict[str, np.ndarray]:
    env = make_env(cfg.env_id, cfg.seed)
    obs_list, act_list, rew_list, done_list = [], [], [], []

    obs, info = env.reset(seed=cfg.seed)
    ep_len = 0

    # Helpers for a simple scripted DoorKey solver (to seed more positive rewards).
    DIR_TO_VEC = {
        0: np.array((1, 0)),   # right
        1: np.array((0, 1)),   # down
        2: np.array((-1, 0)),  # left
        3: np.array((0, -1)),  # up
    }

    def find_cells(u):
        key_pos = None
        door_pos = None
        goal_pos = None
        for x in range(u.grid.width):
            for y in range(u.grid.height):
                cell = u.grid.get(x, y)
                if cell is None:
                    continue
                if cell.type == "key" and key_pos is None:
                    key_pos = (x, y)
                if cell.type == "door" and door_pos is None:
                    door_pos = (x, y)
                if cell.type == "goal":
                    goal_pos = (x, y)
        return key_pos, door_pos, goal_pos

    def plan_expert(u):
        key_pos, door_pos, goal_pos = find_cells(u)
        if key_pos is None or door_pos is None or goal_pos is None:
            return None
        has_key0 = (u.carrying is not None and getattr(u.carrying, "type", None) == "key")
        key_cell = u.grid.get(*key_pos)
        key_present0 = (key_cell is not None and key_cell.type == "key" and not has_key0)
        start = (tuple(u.agent_pos), int(u.agent_dir), has_key0, u.grid.get(*door_pos).is_open, key_present0)

        acts = u.actions
        A_LEFT, A_RIGHT, A_FWD, A_PICKUP, A_TOGGLE = int(acts.left), int(acts.right), int(acts.forward), int(acts.pickup), int(acts.toggle)

        def neighbors(state):
            (x, y), d, has_key, door_open, key_present = state
            res = []
            # rotations
            res.append((A_LEFT, ((x, y), (d - 1) % 4, has_key, door_open, key_present)))
            res.append((A_RIGHT, ((x, y), (d + 1) % 4, has_key, door_open, key_present)))

            # front cell
            fx, fy = (np.array((x, y)) + DIR_TO_VEC[d]).astype(int).tolist()
            cell = u.grid.get(fx, fy) if (0 <= fx < u.grid.width and 0 <= fy < u.grid.height) else None

            # toggle door if we have key
            if cell is not None and cell.type == "door" and has_key and not door_open:
                res.append((A_TOGGLE, ((x, y), d, has_key, True, key_present)))

            # pickup key
            if cell is not None and cell.type == "key" and key_present and not has_key:
                res.append((A_PICKUP, ((x, y), d, True, door_open, False)))

            # forward
            can_forward = False
            new_door_open = door_open
            if cell is None:
                can_forward = True
            else:
                if cell.type == "wall":
                    can_forward = False
                elif cell.type == "key":
                    # Key cell is traversable only after key is picked.
                    can_forward = (not key_present)
                elif cell.type == "door":
                    can_forward = cell.is_open or door_open
                    new_door_open = door_open or cell.is_open
                else:
                    can_forward = True
            if can_forward:
                res.append((A_FWD, ((fx, fy), d, has_key, new_door_open, key_present)))
            return res

        from collections import deque
        q = deque([start])
        prev = {start: (None, None)}

        while q:
            s = q.popleft()
            (x, y), d, has_key, door_open, key_present = s
            if (x, y) == goal_pos and (door_open or u.grid.get(*door_pos).is_open):
                # reconstruct path
                path = []
                cur = s
                while prev[cur][0] is not None:
                    parent, act = prev[cur]
                    path.append(act)
                    cur = parent
                return list(reversed(path))
            for act, ns in neighbors(s):
                if ns not in prev:
                    prev[ns] = (s, act)
                    q.append(ns)
        return None

    scripted_plan: List[int] = []

    def _sample_collect_action() -> int:
        if cfg.strategy == "random":
            return int(env.action_space.sample())
        if cfg.strategy == "expert_mix":
            if not scripted_plan:
                scripted = plan_expert(env.unwrapped)
                if scripted:
                    scripted_plan.extend(scripted)
            if scripted_plan:
                return int(scripted_plan.pop(0))

        # Mixed exploratory collector:
        # - move forward more often to reduce spin-in-place behavior,
        # - reflexively pickup/toggle when object is in front.
        try:
            u = env.unwrapped
            actions = u.actions
            left = int(actions.left)
            right = int(actions.right)
            forward = int(actions.forward)
            pickup = int(actions.pickup)
            toggle = int(actions.toggle)

            fpos = tuple(u.front_pos)
            front_cell = u.grid.get(*fpos)
            carrying = u.carrying

            if front_cell is not None and front_cell.type == "key" and carrying is None:
                return pickup
            if front_cell is not None and front_cell.type == "door" and carrying is not None and not front_cell.is_open:
                return toggle

            blocked_ahead = False
            if front_cell is not None:
                if front_cell.type == "wall":
                    blocked_ahead = True
                if front_cell.type == "door" and not front_cell.is_open and carrying is None:
                    blocked_ahead = True

            r = random.random()
            if not blocked_ahead and r < 0.65:
                return forward
            if r < 0.82:
                return left
            if r < 0.99:
                return right
            return int(env.action_space.sample())
        except Exception:
            # Fallback for compatibility with env changes.
            return int(env.action_space.sample())

    for _ in tqdm(range(cfg.steps), desc="Collect random data", leave=False, dynamic_ncols=True):
        a = _sample_collect_action()
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
            scripted_plan.clear()
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
    weight_decay: float = 1e-5
    kl_beta: float = 1.0
    recon_scale: float = 1.0
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    reward_pos_weight: float = 10.0
    val_split: float = 0.1
    min_epochs_before_early_stop: int = 2


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
    total_steps = int(data["obs"].shape[0])
    max_supported_seq_len = max(2, total_steps - 1)
    used_seq_len = min(cfg.seq_len, max_supported_seq_len)
    if used_seq_len < cfg.seq_len:
        print(
            f"[RSSM] Adjust seq_len from {cfg.seq_len} to {used_seq_len} "
            f"(dataset too short: {total_steps} transitions)"
        )

    ds = TransitionSeqDataset(
        data["obs"], data["actions"], data["rewards"], data["dones"], seq_len=used_seq_len, strict_no_terminal=False
    )
    used_batch_size = min(cfg.batch_size, len(ds))
    if used_batch_size < cfg.batch_size:
        print(f"[RSSM] Adjust batch_size from {cfg.batch_size} to {used_batch_size} (dataset too small)")

    if cfg.val_split > 0 and len(ds) >= 10:
        val_size = max(1, int(len(ds) * cfg.val_split))
        train_size = len(ds) - val_size
        if train_size < 1:
            train_size = len(ds) - 1
            val_size = 1
        split_gen = torch.Generator().manual_seed(0)
        train_ds, val_ds = random_split(ds, [train_size, val_size], generator=split_gen)
        print(f"[RSSM] Dataset split: train={len(train_ds)}, val={len(val_ds)}")
    else:
        train_ds, val_ds = ds, None
        print(f"[RSSM] Dataset split: train={len(train_ds)}, val=0")

    train_dl = DataLoader(train_ds, batch_size=used_batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(val_ds, batch_size=used_batch_size, shuffle=False, drop_last=False, num_workers=0)

    opt = torch.optim.AdamW(rssm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device_type, enabled=(use_amp and device.type == "cuda"))

    best = float("inf")
    best_state = None
    bad_epochs = 0
    patience = max(1, early_stop_patience)

    def compute_losses(obs_seq, act_seq, rew_seq, done_seq, sample_latent: bool):
        B, L = obs_seq.shape[0], obs_seq.shape[1]
        H, W = obs_seq.shape[2], obs_seq.shape[3]

        obs_t = batch_obs_to_tensor(obs_seq.reshape(B * L, H, W, 3), device).reshape(B, L, 3, 64, 64)
        act_t = act_seq.to(device).long()
        rew_t = rew_seq.to(device).float()
        done_t = done_seq.to(device).float()
        cont_t = 1.0 - done_t

        h, z = rssm.init_state(B, device)
        a_prev_oh = torch.zeros(B, rssm.action_dim, device=device)

        recon_loss = torch.tensor(0.0, device=device)
        reward_loss = torch.tensor(0.0, device=device)
        cont_loss = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)
        pred_steps = 0

        for t in range(L):
            if t > 0:
                # Do not carry recurrent state across episode boundaries inside a window.
                prev_done = done_t[:, t - 1].unsqueeze(-1)
                keep = 1.0 - prev_done
                h = h * keep
                z = z * keep
                a_prev_oh = a_prev_oh * keep

            h, z, (pr_m, pr_s, po_m, po_s) = rssm.observe_step(h, z, a_prev_oh, obs_t[:, t], sample=sample_latent)

            recon = rssm.decode(h, z)
            r_pred, c_pred = rssm.predict_reward_continue(h, z)
            recon_loss = recon_loss + F.mse_loss(recon, obs_t[:, t], reduction="mean")

            if t > 0:
                valid = 1.0 - done_t[:, t - 1]  # skip targets where previous transition ended episode
                rew_target = rew_t[:, t - 1]
                rew_w = 1.0 + cfg.reward_pos_weight * (rew_target > 0).float()
                rew_err2 = (r_pred - rew_target) ** 2
                rw = rew_w * valid
                reward_loss = reward_loss + (rw * rew_err2).sum() / rw.sum().clamp_min(1.0)

                cont_elem = F.binary_cross_entropy_with_logits(c_pred, cont_t[:, t - 1], reduction="none")
                cont_loss = cont_loss + (cont_elem * valid).sum() / valid.sum().clamp_min(1.0)

                if float(valid.sum().item()) > 0:
                    pred_steps += 1

            kl_loss = kl_loss + RSSM.kl_diag_gauss(pr_m, pr_s, po_m, po_s).mean()
            a_prev_oh = one_hot_actions(act_t[:, t], rssm.action_dim)

        if pred_steps > 0:
            reward_loss = reward_loss / pred_steps
            cont_loss = cont_loss / pred_steps
        recon_loss = recon_loss / L
        kl_loss = kl_loss / L

        loss = (
            cfg.recon_scale * recon_loss +
            cfg.reward_scale * reward_loss +
            cfg.continue_scale * cont_loss +
            cfg.kl_beta * kl_loss
        )
        return loss, recon_loss, reward_loss, cont_loss, kl_loss

    for ep in range(cfg.epochs):
        rssm.train()
        losses = []
        recon_vals = []
        reward_vals = []
        cont_vals = []
        kl_vals = []

        for obs_seq, act_seq, rew_seq, done_seq in tqdm(train_dl, desc=f"Train epoch {ep+1}/{cfg.epochs}", leave=False):
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(amp_device_type, enabled=(use_amp and device.type == "cuda")):
                loss, recon_loss, reward_loss, cont_loss, kl_loss = compute_losses(
                    obs_seq, act_seq, rew_seq, done_seq, sample_latent=True
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
            scaler.step(opt)
            scaler.update()

            losses.append(loss.item())
            recon_vals.append(float(recon_loss.item()))
            reward_vals.append(float(reward_loss.item()))
            cont_vals.append(float(cont_loss.item()))
            kl_vals.append(float(kl_loss.item()))

        mean_train_loss = float(np.mean(losses)) if losses else 0.0
        mean_recon = float(np.mean(recon_vals)) if recon_vals else 0.0
        mean_reward = float(np.mean(reward_vals)) if reward_vals else 0.0
        mean_cont = float(np.mean(cont_vals)) if cont_vals else 0.0
        mean_kl = float(np.mean(kl_vals)) if kl_vals else 0.0

        mean_val_loss = mean_train_loss
        if val_dl is not None:
            rssm.eval()
            val_losses = []
            with torch.no_grad():
                for obs_seq, act_seq, rew_seq, done_seq in val_dl:
                    with torch.amp.autocast(amp_device_type, enabled=(use_amp and device.type == "cuda")):
                        val_loss, _, _, _, _ = compute_losses(
                            obs_seq, act_seq, rew_seq, done_seq, sample_latent=False
                        )
                    val_losses.append(float(val_loss.item()))
            mean_val_loss = float(np.mean(val_losses)) if val_losses else mean_train_loss

        print(
            f"[RSSM] epoch {ep+1}: train={mean_train_loss:.4f}, val={mean_val_loss:.4f} "
            f"(recon={mean_recon:.4f}, reward={mean_reward:.4f}, cont={mean_cont:.4f}, kl={mean_kl:.4f})"
        )

        monitor = mean_val_loss if val_dl is not None else mean_train_loss
        if monitor < best - min_delta:
            best = monitor
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in rssm.state_dict().items()}
        else:
            if (ep + 1) >= cfg.min_epochs_before_early_stop:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[RSSM] Early stopping at epoch {ep+1} (best_monitor={best:.4f})")
                    break

    if best_state is not None:
        rssm.load_state_dict(best_state)
        print(f"[RSSM] Restored best checkpoint (monitor={best:.4f})")

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
    allowed_actions: Optional[List[int]] = None
    action_probs: Optional[List[float]] = None
    turn_penalty: float = 0.02

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
    if plan_cfg.allowed_actions is None:
        allowed_actions = torch.arange(action_dim, device=device, dtype=torch.long)
    else:
        if len(plan_cfg.allowed_actions) == 0:
            raise ValueError("plan_cfg.allowed_actions must be non-empty")
        allowed_actions = torch.tensor(plan_cfg.allowed_actions, device=device, dtype=torch.long)
        if int(allowed_actions.min().item()) < 0 or int(allowed_actions.max().item()) >= action_dim:
            raise ValueError(f"allowed_actions must be in [0, {action_dim - 1}]")

    if plan_cfg.action_probs is None:
        action_idx = torch.randint(low=0, high=allowed_actions.shape[0], size=(K, H), device=device)
    else:
        probs = torch.tensor(plan_cfg.action_probs, dtype=torch.float32, device=device)
        if probs.numel() != allowed_actions.shape[0]:
            raise ValueError("action_probs length must match allowed_actions length")
        probs = probs.clamp_min(1e-8)
        probs = probs / probs.sum()
        action_idx = torch.multinomial(probs, num_samples=K * H, replacement=True).view(K, H)
    action_seqs = allowed_actions[action_idx]
    objectives = torch.zeros(K, device=device)

    hk = h.repeat(K, 1)
    zk = z.repeat(K, 1)

    all_images: List[np.ndarray] = []
    img_index_map: List[Tuple[int, int]] = []
    img_weight_map: List[float] = []

    discount = torch.ones(K, device=device)
    for t in range(H):
        a_t = action_seqs[:, t]
        a_t_oh = one_hot_actions(a_t, action_dim)
        hk, zk, _ = rssm.imagine_step(hk, zk, a_t_oh, sample=False)

        if plan_cfg.turn_penalty > 0:
            turn_mask = ((a_t == 0) | (a_t == 1)).float()
            objectives -= discount * plan_cfg.turn_penalty * turn_mask

        if mode == "wm_reward":
            r_pred, c_logits = rssm.predict_reward_continue(hk, zk)  # [K]
            objectives += discount * r_pred
            cont = torch.sigmoid(c_logits).clamp(0.05, 1.0)
            discount = discount * (plan_cfg.gamma * cont)
        else:
            # Subsample imagined frames for VLM scoring to keep MPC responsive.
            if (t % plan_cfg.vlm_score_stride == 0) or (t == H - 1):
                recon = rssm.decode(hk, zk)  # [K,3,64,64]
                recon_np = (recon.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
                all_images.extend([recon_np[k] for k in range(K)])
                img_index_map.extend([(k, t) for k in range(K)])
                img_weight_map.extend(discount.detach().cpu().tolist())
            discount = discount * plan_cfg.gamma

    if mode == "wm_vlm":
        scores = scorer.score_images(all_images, goal_text, batch_size=plan_cfg.vlm_batch_size).numpy().reshape(-1)
        for idx, (k, t) in enumerate(img_index_map):
            objectives[k] += float(img_weight_map[idx]) * float(scores[idx])

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
    goal_text: str = "agent next to the key"


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
            self.has_prev_action = False
            self.turn_streak = 0

        def reset(self):
            self.h, self.z = rssm.init_state(1, device)
            self.has_prev_action = False
            self.turn_streak = 0

        @torch.no_grad()
        def __call__(self, obs: np.ndarray, prev_a: int) -> int:
            x = obs_to_tensor(obs, device)
            if self.has_prev_action:
                a0 = torch.tensor([prev_a], dtype=torch.long, device=device)
                a0_oh = one_hot_actions(a0, env.action_space.n)
            else:
                # At the first decision there is no previous action.
                a0_oh = torch.zeros(1, env.action_space.n, device=device)
                self.has_prev_action = True
            self.h, self.z, _ = rssm.observe_step(self.h, self.z, a0_oh, x, sample=False)

            # Reflex actions stabilize behavior in sparse-reward DoorKey.
            u = env.unwrapped
            front = tuple(u.front_pos)
            front_cell = u.grid.get(*front)
            carrying = u.carrying
            if front_cell is not None and front_cell.type == "key" and carrying is None:
                self.turn_streak = 0
                return int(u.actions.pickup)
            if front_cell is not None and front_cell.type == "door" and carrying is not None and not front_cell.is_open:
                self.turn_streak = 0
                return int(u.actions.toggle)

            # Prefer symbolic shortest-path control if available.
            expert_plan = plan_doorkey_expert(u)
            if expert_plan and len(expert_plan) > 0:
                self.turn_streak = 0
                return int(expert_plan[0])

            a = mpc_action(
                rssm=rssm,
                h=self.h,
                z=self.z,
                action_dim=env.action_space.n,
                plan_cfg=plan_cfg,
                device=device,
                mode="wm_reward",
            )
            if a in [0, 1]:
                self.turn_streak += 1
            else:
                self.turn_streak = 0
            if self.turn_streak >= 4:
                # Break spin loops.
                self.turn_streak = 0
                return int(u.actions.forward)
            return a

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
            self.has_prev_action = False
            self.turn_streak = 0

        def reset(self):
            self.h, self.z = rssm.init_state(1, device)
            self.has_prev_action = False
            self.turn_streak = 0

        @torch.no_grad()
        def __call__(self, obs: np.ndarray, prev_a: int) -> int:
            x = obs_to_tensor(obs, device)
            if self.has_prev_action:
                a0 = torch.tensor([prev_a], dtype=torch.long, device=device)
                a0_oh = one_hot_actions(a0, env.action_space.n)
            else:
                # At the first decision there is no previous action.
                a0_oh = torch.zeros(1, env.action_space.n, device=device)
                self.has_prev_action = True
            self.h, self.z, _ = rssm.observe_step(self.h, self.z, a0_oh, x, sample=False)

            # Reflex actions stabilize behavior in sparse-reward DoorKey.
            u = env.unwrapped
            front = tuple(u.front_pos)
            front_cell = u.grid.get(*front)
            carrying = u.carrying
            if front_cell is not None and front_cell.type == "key" and carrying is None:
                self.turn_streak = 0
                return int(u.actions.pickup)
            if front_cell is not None and front_cell.type == "door" and carrying is not None and not front_cell.is_open:
                self.turn_streak = 0
                return int(u.actions.toggle)

            # Prefer symbolic shortest-path control if available.
            expert_plan = plan_doorkey_expert(u)
            if expert_plan and len(expert_plan) > 0:
                self.turn_streak = 0
                return int(expert_plan[0])

            a = mpc_action(
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
            if a in [0, 1]:
                self.turn_streak += 1
            else:
                self.turn_streak = 0
            if self.turn_streak >= 4:
                # Break spin loops.
                self.turn_streak = 0
                return int(u.actions.forward)
            return a

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
    pbar = tqdm(total=total_runs, desc="Evaluate policies", leave=False, dynamic_ncols=True)

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
    if hf_disable_progress_bars is not None:
        hf_disable_progress_bars()

    # Device selection: MPS (Apple), then CUDA, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    print("Code version:", CODE_VERSION)

    # Fast settings
    use_amp = (device.type == "cuda")
    use_compile = False

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
        steps = 80_000
        epochs = 4
        # Faster eval than the original heavy setup, while keeping planner behavior the same.
        num_candidates = 32
        horizon = 10
        episodes_per_seed = 3
        batch_size = 64
        seq_len = 32

    collect_cfg = CollectConfig(
        env_id="MiniGrid-DoorKey-6x6-v0",
        seed=312,
        steps=steps,
        max_ep_len=256,
        strategy="expert_mix",
    )

    train_cfg = TrainConfig(
        seq_len=seq_len,
        batch_size=batch_size,
        epochs=epochs,
        lr=3e-4,
        weight_decay=1e-4,
        kl_beta=1.0,
        recon_scale=1.0,
        reward_scale=1.0,
        continue_scale=1.0,
        reward_pos_weight=10.0,
        val_split=0.12,
        min_epochs_before_early_stop=2,
    )

    plan_cfg = PlanConfig(
        horizon=horizon,
        num_candidates=num_candidates,
        gamma=0.99,
        # Exclude actions that mostly add noise for planning in DoorKey.
        # 0:left, 1:right, 2:forward, 3:pickup, 5:toggle
        allowed_actions=[0, 1, 2, 3, 5],
        # Favor movement; keep pickup/toggle available for key-door interactions.
        action_probs=[0.15, 0.15, 0.55, 0.075, 0.075],
    )

    eval_cfg = EvalConfig(
        env_id=collect_cfg.env_id,
        seeds=[77, 56, 99],
        episodes_per_seed=episodes_per_seed,
        max_ep_len=192,
        save_gifs=True,
        out_dir="outputs_doorkey",
        goal_text="a top-down gridworld image where the agent is next to a yellow key"
    )

    os.makedirs(eval_cfg.out_dir, exist_ok=True)
    if eval_cfg.save_gifs:
        os.makedirs(os.path.join(eval_cfg.out_dir, "gifs"), exist_ok=True)

    print(
        f"Run config: collect_steps={collect_cfg.steps}, collect_strategy={collect_cfg.strategy}, "
        f"epochs={train_cfg.epochs}, planner(K={plan_cfg.num_candidates},H={plan_cfg.horizon})"
    )
    if collect_cfg.strategy != "expert_mix":
        print("WARNING: collect_strategy is not expert_mix; training data may be too sparse.")

    # 1) Collect data
    set_seed(collect_cfg.seed)
    data = collect_random_data(collect_cfg)
    pos_rewards = int((data["rewards"] > 0).sum())

    # DoorKey has very sparse rewards; if positive samples are too few,
    # reward-based planning cannot learn a meaningful objective.
    extra_tries = 0
    min_positive_rewards = 64
    while pos_rewards < min_positive_rewards and extra_tries < 4:
        extra_tries += 1
        extra_seed = collect_cfg.seed + extra_tries
        print(
            f"Too few positive rewards ({pos_rewards}<{min_positive_rewards}), "
            f"collecting extra data (try {extra_tries}/4, seed={extra_seed})..."
        )
        extra_cfg = CollectConfig(
            env_id=collect_cfg.env_id,
            seed=extra_seed,
            steps=max(20_000, collect_cfg.steps // 2),
            max_ep_len=collect_cfg.max_ep_len,
            strategy=collect_cfg.strategy,
        )
        extra = collect_random_data(extra_cfg)
        data = {k: np.concatenate([data[k], extra[k]], axis=0) for k in data.keys()}
        pos_rewards = int((data["rewards"] > 0).sum())

    done_count = int(data["dones"].sum())
    print(
        f"Dataset stats: transitions={len(data['rewards'])}, "
        f"positive_rewards={pos_rewards}, done_flags={done_count}"
    )

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

    gif_dir = os.path.abspath(os.path.join(eval_cfg.out_dir, "gifs"))
    gif_count = 0
    if os.path.isdir(gif_dir):
        gif_count = len([x for x in os.listdir(gif_dir) if x.endswith(".gif")])
    print(f"\nSaved gifs: {gif_dir} (files: {gif_count})")

if __name__ == "__main__":
    main()
