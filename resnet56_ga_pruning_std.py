"""
ResNet-56 Adaptive GA Pruning — CIFAR-10/100
============================================
Adapted from VGG16 GA pruning framework.

Key differences vs VGG16:
  1. ResNet-56 has residual (skip) connections → coupled layer groups
  2. Physical pruning uses torch_pruning DependencyGraph (handles coupling automatically)
  3. FilterPruner uses named_modules hooks instead of model.features iteration
  4. _estimate_metrics accounts for residual block structure
  5. Projection shortcuts at stage transitions are NOT pruned (shape constraint)

Usage (single command):
  python resnet56_ga_pruning.py --dataset CIFAR10 --target_macs 0.30 --resume auto

Time estimates (RTX 4090):
  - Baseline train  : ~2h  (skip if checkpoint exists)
  - Taylor ranking  : ~90s (1 epoch, CIFAR-10, batch=128)
  - NSGA-II search  : ~20s (pop=50, gen=30, pure numpy)
  - Fine-tuning KD  : ~3h  (150 epochs, CIFAR-10)
  - CIFAR-100       : ~4h  (fine-tuning only, ranking ~100s)
  Total new run     : ~5-6h (CIFAR-10), ~6-7h (CIFAR-100)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
import copy
import time
import json
import random
import argparse
import datetime
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

try:
    import torch_pruning as tp
    HAS_TP = True
except ImportError:
    HAS_TP = False
    print("⚠️  torch_pruning not found — physical pruning will use manual method.")


# ==============================================================================
# CONFIG
# ==============================================================================
class Config:
    device          = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_class   = 'CIFAR10'   # 'CIFAR10' or 'CIFAR100'
    batch_size      = 128
    num_workers     = 4
    use_cuda        = torch.cuda.is_available()
    seed            = 42

    # GA / NSGA-II
    pop_size        = 50
    generations     = 30
    min_ratio       = 0.10        # Hard floor per layer

    # Fine-tuning KD
    kd_epochs       = 150
    kd_lr           = 0.01
    kd_temp         = 4.0
    kd_alpha        = 0.9

    # Paths
    checkpoint_dir  = './checkpoint_resnet56'
    log_dir         = './logs_resnet56'
    figure_dir      = './figures_resnet56'


# ==============================================================================
# RESNET-56 MODEL
# ==============================================================================
class BasicBlock(nn.Module):
    """ResNet BasicBlock with optional projection shortcut."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # Projection shortcut — NOT pruned (shape constraint)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet56(nn.Module):
    """ResNet-56 for CIFAR (n=9 blocks per stage, 3 stages)."""

    def __init__(self, num_classes=10):
        super().__init__()
        block = BasicBlock
        num_blocks = [9, 9, 9]

        self.in_planes = 16
        self.conv1  = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def build_resnet56(num_classes=10):
    return ResNet56(num_classes=num_classes)


# ==============================================================================
# COUPLED LAYER GROUPS
# Projection shortcuts (shortcut.0) are excluded from pruning.
# Within each BasicBlock: conv1 output must equal conv2 input.
# Between consecutive blocks in the same stage: block[i].conv2 out == block[i+1].conv1 in.
# ==============================================================================
def get_coupled_groups(model: ResNet56) -> List[List[str]]:
    """
    Returns list of groups where each group contains layer names
    that must be pruned to the SAME output channel count.

    For ResNet-56 BasicBlock (no bottleneck):
      - Each block has conv1 and conv2.
      - conv2 output feeds into next block's conv1 input (same stage).
      - At stage boundary, first block's conv1 input = previous stage's last conv2 output.
      - Projection shortcut conv must match conv2 output — so we skip projection.

    Coupling rule: conv2 of block[i] and conv1 of block[i+1] (same stage)
    must output/input the same channel count → they form a coupled group.
    """
    groups = []
    for stage_name in ['layer1', 'layer2', 'layer3']:
        stage = getattr(model, stage_name)
        n_blocks = len(stage)
        for i in range(n_blocks - 1):
            # conv2 output of block[i] == conv1 input of block[i+1]
            # These two must be pruned consistently
            name_a = f"{stage_name}.{i}.conv2"
            name_b = f"{stage_name}.{i+1}.conv1"
            groups.append([name_a, name_b])
    return groups


def get_prunable_layers(model: ResNet56) -> List[Tuple[str, nn.Conv2d]]:
    """
    Returns (name, module) for all Conv2d layers that CAN be pruned.
    Projection shortcuts (shortcut.0) are excluded.
    """
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if 'shortcut' not in name:
                prunable.append((name, module))
    return prunable


# ==============================================================================
# DATA LOADERS
# ==============================================================================
def get_data_loaders(config: Config):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std =[0.2023, 0.1994, 0.2010]
    )
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset_fn = (torchvision.datasets.CIFAR10 if config.dataset_class == 'CIFAR10'
                  else torchvision.datasets.CIFAR100)

    trainset = dataset_fn('./data', train=True,  download=True, transform=transform_train)
    testset  = dataset_fn('./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset,  batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    return train_loader, test_loader


# ==============================================================================
# FILTER PRUNER — Taylor importance, architecture-agnostic via hooks
# ==============================================================================
class FilterPrunerResNet:
    """
    Computes Taylor importance for any architecture using forward hooks.
    Works on arbitrary Conv2d layers, not model.features iteration.
    Skips projection shortcuts automatically.
    """

    def __init__(self, model: nn.Module, config: Config,
                 prunable_names: List[str]):
        self.model          = model
        self.config         = config
        self.prunable_names = set(prunable_names)
        self.reset()

    def reset(self):
        self.filter_ranks       = {}   # name → accumulated Taylor score tensor
        self._activations       = {}   # name → activation tensor (for hook)
        self._hooks             = []

    def _register_hooks(self):
        """Register forward hooks on all prunable Conv2d layers."""
        self._activations = {}
        for name, module in self.model.named_modules():
            if name in self.prunable_names and isinstance(module, nn.Conv2d):
                # Capture name in closure
                def make_hook(n):
                    def hook(mod, inp, out):
                        # Save activation; register backward hook on output
                        out.retain_grad()
                        self._activations[n] = out
                    return hook
                h = module.register_forward_hook(make_hook(name))
                self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def accumulate_ranks(self, inputs, targets, criterion):
        """
        One forward+backward pass to accumulate Taylor scores.
        Call this once per batch during the ranking epoch.
        """
        self._register_hooks()
        self.model.zero_grad()

        outputs = self.model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()

        # Compute Taylor importance = |activation * gradient| averaged over batch, H, W
        for name, act in self._activations.items():
            if act.grad is None:
                continue
            # act shape: (B, C, H, W)
            taylor = (act * act.grad).abs().mean(dim=(0, 2, 3)).detach().cpu()
            if name not in self.filter_ranks:
                self.filter_ranks[name] = torch.zeros(act.size(1))
            self.filter_ranks[name] += taylor

        self._remove_hooks()

    def normalize_ranks(self):
        """L2-normalize per layer."""
        for name in self.filter_ranks:
            v    = self.filter_ranks[name]
            norm = torch.sqrt((v * v).sum())
            if norm > 0:
                self.filter_ranks[name] = v / norm


# ==============================================================================
# GA PRUNER FOR RESNET-56
# ==============================================================================
class GAPrunerResNet:
    """
    GA-based pruner adapted for ResNet-56.
    Handles coupled layer groups and projection shortcuts.
    """

    def __init__(self, model: ResNet56, config: Config):
        self.model   = model
        self.config  = config
        self.device  = config.device

        self.prunable_layers = get_prunable_layers(model)   # (name, module) list
        self.coupled_groups  = get_coupled_groups(model)    # [[nameA, nameB], ...]
        self.n_layers        = len(self.prunable_layers)

        # name → index in prunable_layers
        self.name_to_idx = {name: i for i, (name, _) in enumerate(self.prunable_layers)}

        # Precompute coupled index groups
        self.coupled_idx_groups = []
        for grp in self.coupled_groups:
            idx_grp = [self.name_to_idx[n] for n in grp if n in self.name_to_idx]
            if len(idx_grp) > 1:
                self.coupled_idx_groups.append(idx_grp)

        # Layer info for MACs estimation
        self._extract_layer_info()

        # To be filled by build_ranking_tables
        self.ranking_tables = {}
        self.layer_stats    = {}

        # Base metrics
        dummy = torch.randn(1, 3, 32, 32).to(self.device)
        self.model.eval()
        if HAS_TP:
            self.base_macs, self.base_params = tp.utils.count_ops_and_params(model, dummy)
        else:
            self.base_macs   = self._rough_macs()
            self.base_params = sum(p.numel() for p in model.parameters())
        print(f"Base MACs: {self.base_macs/1e9:.4f}G | Base Params: {self.base_params/1e6:.4f}M")
        print(f"Prunable Conv2d layers: {self.n_layers}")
        print(f"Coupled groups: {len(self.coupled_idx_groups)}")

    def _rough_macs(self):
        """Fallback MACs estimate if torch_pruning unavailable."""
        total = 0
        for _, module in self.prunable_layers:
            k = module.kernel_size[0]
            total += k * k * module.in_channels * module.out_channels * 16 * 16
        return total

    def _extract_layer_info(self):
        """Extract spatial dims H, W, K, Cin, Cout via hooks for fast estimation."""
        self.layer_info = {}
        hooks = []
        dummy = torch.randn(1, 3, 32, 32).to(self.device)

        def make_hook(name):
            def hook(module, inp, out):
                x = inp[0]
                self.layer_info[name] = {
                    'h'   : x.shape[2],
                    'w'   : x.shape[3],
                    'k'   : module.kernel_size[0],
                    'cin' : module.in_channels,
                    'cout': module.out_channels,
                }
            return hook

        for name, module in self.prunable_layers:
            hooks.append(module.register_forward_hook(make_hook(name)))

        self.model.eval()
        with torch.no_grad():
            self.model(dummy)
        for h in hooks:
            h.remove()

    def build_ranking_tables(self, train_loader, criterion,
                              n_batches: Optional[int] = None):
        """
        Run one ranking epoch (or n_batches batches) to build Taylor tables.
        n_batches=None → full epoch (~390 batches for CIFAR-10, ~90s on RTX 4090)
        n_batches=50   → fast mode (~12s, slightly less accurate)
        """
        prunable_names = [name for name, _ in self.prunable_layers]
        fp = FilterPrunerResNet(self.model, self.config, prunable_names)

        print(f"\n[Step 1] Building ranking tables "
              f"({'full epoch' if n_batches is None else f'{n_batches} batches'})...")
        t0 = time.time()

        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if n_batches is not None and batch_idx >= n_batches:
                break
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)
            fp.accumulate_ranks(inputs, targets, criterion)
            if batch_idx % 50 == 0:
                print(f"   Ranking batch {batch_idx}/{len(train_loader) if n_batches is None else n_batches}")

        fp.normalize_ranks()
        print(f"   Ranking done in {time.time()-t0:.1f}s")

        # Build tables
        self.ranking_tables = {'min': {}, 'avg': {}}
        self.layer_stats    = {}

        for i, (name, _) in enumerate(self.prunable_layers):
            if name not in fp.filter_ranks:
                # Layer never activated (edge case) — fill with uniform
                cout = self.layer_info[name]['cout']
                ranks = np.ones(cout) / cout
            else:
                ranks = fp.filter_ranks[name].numpy()

            max_score  = np.max(ranks) + 1e-8
            norm_scores = ranks / max_score

            self.layer_stats[i] = {
                'raw' : ranks,
                'norm': norm_scores,
                'max' : max_score,
                'mean': np.mean(ranks),
                'name': name,
            }
            self.ranking_tables['min'][i] = np.sort(norm_scores)

        # Global sorted reference for Median strategy
        all_norm = np.concatenate([v['norm'] for v in self.layer_stats.values()])
        global_sorted = np.sort(all_norm)
        for i in self.layer_stats:
            self.ranking_tables['avg'][i] = global_sorted

        print(f"✅ Ranking tables built for {len(self.layer_stats)} layers.")

    # ------------------------------------------------------------------
    # Chromosome decode with coupling enforcement
    # ------------------------------------------------------------------
    def _decode_chromosome(self, chrom):
        """
        Decode chromosome → (channels, strategies).
        Enforces:
          1. Hard floor: min 10% retention, rounded to multiples of 8
          2. Coupled groups: all layers in a group get min(channels) of the group
        """
        channels   = []
        strategies = []

        for i, (name, module) in enumerate(self.prunable_layers):
            r    = chrom[i * 2]
            s    = chrom[i * 2 + 1]
            orig = module.out_channels

            n_keep = int(orig * r)
            n_keep = max(int(orig * self.config.min_ratio), n_keep)
            n_keep = max(8, round(n_keep / 8) * 8)
            n_keep = min(n_keep, orig)

            channels.append(n_keep)
            strategies.append('min' if s < 0.5 else 'avg')

        # Enforce coupled groups: take minimum to keep shapes consistent
        for grp in self.coupled_idx_groups:
            min_ch = min(channels[i] for i in grp)
            for i in grp:
                channels[i] = min_ch

        return channels, strategies

    # ------------------------------------------------------------------
    # Fast MACs/Params estimation
    # ------------------------------------------------------------------
    def _estimate_metrics(self, channels: List[int]):
        total_macs   = 0
        total_params = 0

        # Build name→channel map for output channels
        out_ch = {name: channels[i] for i, (name, _) in enumerate(self.prunable_layers)}

        for i, (name, module) in enumerate(self.prunable_layers):
            info  = self.layer_info[name]
            c_out = channels[i]

            # Input channels: for layer 0 (stem conv1) it's 3;
            # for others, it's the output of the previous *connected* layer.
            # Simple approximation: use module's original in_channels scaled by ratio
            orig_out = module.out_channels
            orig_in  = module.in_channels
            ratio_in = 1.0  # conservative — shortcut paths keep original in_channels
            c_in     = max(8, int(orig_in * (channels[i-1] / self.prunable_layers[i-1][1].out_channels))
                           if i > 0 else orig_in)

            macs         = info['h'] * info['w'] * info['k'] * info['k'] * c_in * c_out
            params       = info['k'] * info['k'] * c_in * c_out
            total_macs  += macs
            total_params += params

        return total_macs, total_params

    # ------------------------------------------------------------------
    # Proxy fitness
    # ------------------------------------------------------------------
    def proxy_capacity_score(self, channels: List[int],
                              strategies: List[str]) -> float:
        """Layer-wise normalized Taylor proxy (used by NSGA-II)."""
        total_score = 0.0
        global_max  = max(v['max'] for v in self.layer_stats.values()) + 1e-8

        for i, keep in enumerate(channels):
            stats = self.layer_stats[i]
            raw   = stats['raw']

            sorted_idx = np.argsort(raw)[::-1]
            kept_idx   = sorted_idx[:keep]
            preserved  = raw[kept_idx].sum()
            total      = raw.sum() + 1e-8
            layer_score = preserved / total

            if strategies[i] == 'avg':
                layer_score *= 1.05
            else:
                layer_score *= 0.95

            attenuation = stats['max'] / global_max
            if attenuation < 0.25 and keep / len(raw) < 0.4:
                layer_score *= 0.8

            total_score += layer_score

        return total_score / len(channels)

    def _calc_fitness_so(self, chrom, target_macs_ratio: float) -> float:
        """Single-objective fitness (GA-SO baseline)."""
        channels, strategies = self._decode_chromosome(chrom)

        est_macs, est_params = self._estimate_metrics(channels)
        macs_ratio   = est_macs   / self.base_macs
        params_ratio = est_params / self.base_params

        # Taylor proxy
        total_score = 0.0
        total_max   = 0.0
        for i, n_keep in enumerate(channels):
            strat = strategies[i]
            if strat == 'min':
                ranks      = self.ranking_tables['min'][i]
                preserved  = np.sum(ranks[-n_keep:])
                layer_total = np.sum(ranks)
            else:
                global_ranks = self.ranking_tables['avg'][i]
                start        = int((len(global_ranks) - n_keep) / 2)
                preserved    = np.sum(global_ranks[start:start + n_keep])
                layer_total  = np.sum(global_ranks)
            total_score += preserved
            total_max   += layer_total
        base_fitness = total_score / (total_max + 1e-8)

        # Structural safeguard
        structure_penalty = 0.0
        global_max_score  = max(v['max'] for v in self.layer_stats.values())
        for i, c in enumerate(channels):
            orig        = self.prunable_layers[i][1].out_channels
            ratio       = c / orig
            attenuation = self.layer_stats[i]['max'] / (global_max_score + 1e-8)
            if attenuation < 0.2 and ratio < 0.4:
                structure_penalty += 3.0 * ((0.4 - ratio) ** 2)
            if ratio < 0.1:
                return -1e6

        # MACs penalty — symmetric: penalize both over AND under pruning
        delta = macs_ratio - target_macs_ratio
        if delta > 0:
            # Over budget: heavy penalty
            macs_penalty = 10.0 * delta + 20.0 * (delta ** 2)
        elif delta < -0.10:
            # Pruned more than 10% below target: mild penalty
            macs_penalty = 5.0 * (-delta - 0.10)
        else:
            macs_penalty = 0.0

        return base_fitness - macs_penalty - structure_penalty - 0.1 * params_ratio


# ==============================================================================
# NSGA-II PRUNER (same logic as original, uses GAPrunerResNet)
# ==============================================================================
@dataclass
class NSGASolution:
    chrom   : list
    acc     : float = None
    macs    : float = None
    params  : float = None
    rank    : int   = None
    crowding: float = 0.0


class NSGAPrunerResNet:
    def __init__(self, ga_pruner: GAPrunerResNet,
                 pop_size=50, generations=30, min_ratio=0.1):
        self.ga      = ga_pruner
        self.pop_size    = pop_size
        self.generations = generations
        self.min_ratio   = min_ratio
        self.n_layers    = ga_pruner.n_layers
        self.population  = []

    def random_chrom(self):
        chrom = []
        for _ in range(self.n_layers):
            chrom.append(random.uniform(self.min_ratio, 1.0))
            chrom.append(random.random())
        return chrom

    def init_population(self):
        self.population = [NSGASolution(self.random_chrom())
                           for _ in range(self.pop_size)]

    def evaluate(self, sol: NSGASolution):
        channels, strategies = self.ga._decode_chromosome(sol.chrom)
        macs, params         = self.ga._estimate_metrics(channels)
        proxy_acc            = self.ga.proxy_capacity_score(channels, strategies)
        sol.acc    = proxy_acc
        sol.macs   = macs
        sol.params = params

    def dominates(self, a, b):
        return ((a.acc >= b.acc and a.macs <= b.macs and a.params <= b.params) and
                (a.acc > b.acc  or  a.macs < b.macs  or  a.params < b.params))

    def fast_nondominated_sort(self):
        fronts = [[]]
        for p in self.population:
            p.dom_count = 0
            p.dom_set   = []
            for q in self.population:
                if self.dominates(p, q):   p.dom_set.append(q)
                elif self.dominates(q, p): p.dom_count += 1
            if p.dom_count == 0:
                p.rank = 0
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in p.dom_set:
                    q.dom_count -= 1
                    if q.dom_count == 0:
                        q.rank = i + 1
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        return fronts[:-1]

    def crowding_distance(self, front):
        if not front: return
        for s in front: s.crowding = 0
        for key in ['acc', 'macs', 'params']:
            front.sort(key=lambda x: getattr(x, key))
            front[0].crowding = front[-1].crowding = float('inf')
            mn = getattr(front[0],  key)
            mx = getattr(front[-1], key)
            if mx == mn: continue
            for i in range(1, len(front) - 1):
                front[i].crowding += (getattr(front[i+1], key) -
                                      getattr(front[i-1], key)) / (mx - mn)

    def select_next_population(self, fronts):
        new_pop = []
        for f in fronts:
            self.crowding_distance(f)
            if len(new_pop) + len(f) <= self.pop_size:
                new_pop.extend(f)
            else:
                f.sort(key=lambda x: -x.crowding)
                new_pop.extend(f[:self.pop_size - len(new_pop)])
                break
        self.population = new_pop

    def crossover(self, p1, p2):
        cut = random.randint(1, len(p1.chrom) - 2)
        return (NSGASolution(p1.chrom[:cut] + p2.chrom[cut:]),
                NSGASolution(p2.chrom[:cut] + p1.chrom[cut:]))

    def mutate(self, sol, prob=0.1):
        for i in range(len(sol.chrom)):
            if random.random() < prob:
                if i % 2 == 0:
                    sol.chrom[i] = random.uniform(self.min_ratio, 1.0)
                else:
                    sol.chrom[i] = random.random()

    def run(self, log_path: str = None):
        print(f"\n[Step 2] NSGA-II Search (pop={self.pop_size}, gen={self.generations})...")
        t0 = time.time()
        self.init_population()

        for g in range(self.generations):
            for s in self.population:
                self.evaluate(s)
            fronts = self.fast_nondominated_sort()
            self.select_next_population(fronts)

            offspring = []
            while len(offspring) < self.pop_size:
                p1, p2 = random.sample(self.population, 2)
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)
                offspring.extend([c1, c2])
            self.population = offspring[:self.pop_size]

            if (g + 1) % 10 == 0:
                print(f"   Gen {g+1}/{self.generations}")

        # Final eval
        for s in self.population:
            self.evaluate(s)

        elapsed = time.time() - t0
        print(f"   NSGA-II done in {elapsed:.1f}s")
        return self.get_pareto_front()

    def get_pareto_front(self):
        fronts = self.fast_nondominated_sort()
        return fronts[0]

    def select_knee(self, pareto):
        macs   = np.array([s.macs   for s in pareto])
        accs   = np.array([s.acc    for s in pareto])
        params = np.array([s.params for s in pareto])

        def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)

        m_n = norm(macs)
        a_n = norm(accs.max() - accs)   # minimize negative acc
        p_n = norm(params)
        dist = np.sqrt(m_n**2 + a_n**2 + p_n**2)
        return pareto[int(np.argmin(dist))]


# ==============================================================================
# PHYSICAL PRUNING
# ==============================================================================
def _get_tp_version():
    """Return torch_pruning major version as int."""
    try:
        ver = tp.__version__.split('.')[0]
        return int(ver)
    except Exception:
        return 0


def physical_prune_resnet56(model: ResNet56,
                             ga_pruner: GAPrunerResNet,
                             knee_sol: NSGASolution,
                             config: Config) -> ResNet56:
    """
    Apply physical pruning using torch_pruning DependencyGraph.
    Supports both torch_pruning v0.x (get_pruning_plan) and
    v1.x+ (get_pruning_group / importance-based pruner).
    """
    print("\n[Step 3] Physical Pruning...")

    if not HAS_TP:
        raise RuntimeError("torch_pruning required for physical pruning of ResNet.")

    tp_ver = _get_tp_version()
    print(f"   torch_pruning version: {tp.__version__} (API v{tp_ver})")

    channels, strategies = ga_pruner._decode_chromosome(knee_sol.chrom)
    pruned_model = copy.deepcopy(model).to(config.device)
    pruned_model.eval()

    dummy = torch.randn(1, 3, 32, 32).to(config.device)

    if tp_ver >= 1:
        # ── New API (torch_pruning >= 1.0) ───────────────────────────────
        # Build importance scores from our Taylor rankings
        # Use MagnitudeImportance as a wrapper — we override with our scores
        # via custom pruner approach

        # Collect all pruning decisions first, then apply via grouped pruner
        # to respect dependency (coupled layers handled automatically)

        # Step A: build importance dict {module: keep_idxs}
        importance_dict = {}
        module_map = dict(pruned_model.named_modules())

        for i, (name, _) in enumerate(ga_pruner.prunable_layers):
            actual_mod = module_map.get(name)
            if actual_mod is None:
                continue

            n_keep     = channels[i]
            orig       = actual_mod.out_channels
            raw_scores = ga_pruner.layer_stats[i]['raw']
            sorted_idx = np.argsort(raw_scores)  # weak → strong

            if strategies[i] == 'min':
                keep_idx = np.argsort(raw_scores)[::-1][:n_keep]  # top-k
            else:
                global_ref = ga_pruner.ranking_tables['avg'][i]
                g_low      = np.percentile(global_ref, 25)
                g_high     = np.percentile(global_ref, 75)
                inner      = np.where(
                    (raw_scores >= g_low) & (raw_scores <= g_high)
                )[0]
                if len(inner) >= n_keep:
                    keep_idx = inner[:n_keep]
                else:
                    keep_idx = np.argsort(raw_scores)[::-1][:n_keep]

            prune_idx = [j for j in range(orig) if j not in set(keep_idx.tolist())]
            if prune_idx:
                importance_dict[actual_mod] = prune_idx

        # Step B: apply pruning group by group
        DG = tp.DependencyGraph().build_dependency(
            pruned_model, example_inputs=dummy
        )

        pruned_count = 0
        for actual_mod, prune_idx in importance_dict.items():
            try:
                # get_pruning_group is the v1.x API
                group = DG.get_pruning_group(
                    actual_mod,
                    tp.prune_conv_out_channels,
                    idxs=prune_idx
                )
                if DG.check_pruning_group(group):
                    group.prune()
                    pruned_count += 1
            except AttributeError:
                # Fallback: try get_pruning_plan (v0.x)
                try:
                    plan = DG.get_pruning_plan(
                        actual_mod, tp.prune_conv_out_channels, idxs=prune_idx
                    )
                    plan.exec()
                    pruned_count += 1
                except Exception as e2:
                    pass
            except Exception as e:
                pass

        print(f"   ✅ Pruned {pruned_count}/{len(importance_dict)} layer groups")

    else:
        # ── Old API (torch_pruning < 1.0) ────────────────────────────────
        DG = tp.DependencyGraph().build_dependency(
            pruned_model, example_inputs=dummy
        )
        module_map = dict(pruned_model.named_modules())

        for i, (name, _) in enumerate(ga_pruner.prunable_layers):
            actual_mod = module_map.get(name)
            if actual_mod is None:
                continue

            n_keep     = channels[i]
            orig       = actual_mod.out_channels
            n_prune    = orig - n_keep
            if n_prune <= 0:
                continue

            raw_scores = ga_pruner.layer_stats[i]['raw']
            sorted_idx = np.argsort(raw_scores)

            if strategies[i] == 'min':
                prune_idx = sorted_idx[:n_prune].tolist()
            else:
                global_ref = ga_pruner.ranking_tables['avg'][i]
                g_low      = np.percentile(global_ref, 25)
                g_high     = np.percentile(global_ref, 75)
                outer      = np.where(
                    (raw_scores < g_low) | (raw_scores > g_high)
                )[0]
                prune_idx  = (outer[:n_prune] if len(outer) >= n_prune
                              else sorted_idx[:n_prune]).tolist()

            try:
                plan = DG.get_pruning_plan(
                    actual_mod, tp.prune_conv_out_channels, idxs=prune_idx
                )
                plan.exec()
                print(f"   ✅ {name}: {orig} → {n_keep} [{strategies[i]}]")
            except Exception as e:
                print(f"   ⚠️  {name}: skip ({e})")

    pruned_model = pruned_model.to(config.device)
    return pruned_model


# ==============================================================================
# KNOWLEDGE DISTILLATION FINE-TUNING
# ==============================================================================
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.9):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        hard = F.cross_entropy(student_logits, labels)
        soft = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits   / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T ** 2)
        return self.alpha * soft + (1 - self.alpha) * hard


def fine_tune_kd(student: nn.Module,
                 teacher: nn.Module,
                 train_loader, test_loader,
                 config: Config,
                 save_path: str,
                 resume_epoch: int = 0) -> float:
    """
    Fine-tune student with knowledge distillation.
    Supports resuming from checkpoint.
    """
    print(f"\n[Step 4] Fine-tuning with KD ({config.kd_epochs} epochs)...")

    criterion = DistillationLoss(config.kd_temp, config.kd_alpha)
    optimizer = optim.SGD(student.parameters(), lr=config.kd_lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.kd_epochs)

    # torch.compile — ~20-30% speedup on PyTorch >= 2.0, no accuracy impact
    try:
        student = torch.compile(student)
        teacher = torch.compile(teacher)
        print("   torch.compile enabled ✅")
    except Exception as e:
        print(f"   torch.compile skipped ({e})")

    # Fast-forward scheduler if resuming
    for _ in range(resume_epoch):
        scheduler.step()

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    best_acc = 0.0
    log_rows  = []

    for epoch in range(resume_epoch, config.kd_epochs):
        student.train()
        train_loss  = 0.0
        correct     = 0
        total       = 0
        t_epoch     = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            optimizer.zero_grad()

            with torch.no_grad():
                t_logits = teacher(inputs)
            s_logits = student(inputs)
            loss     = criterion(s_logits, t_logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            correct    += s_logits.max(1)[1].eq(targets).sum().item()
            total      += targets.size(0)

        scheduler.step()

        # Validation
        student.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                val_correct += student(inputs).max(1)[1].eq(targets).sum().item()
                val_total   += targets.size(0)
        val_acc = 100. * val_correct / val_total

        elapsed = time.time() - t_epoch
        print(f"  Ep {epoch+1:3d}/{config.kd_epochs} | "
              f"Loss: {train_loss/len(train_loader):.3f} | "
              f"Train: {100.*correct/total:.2f}% | "
              f"Val: {val_acc:.2f}% | "
              f"{elapsed:.0f}s")

        log_rows.append({'epoch': epoch+1, 'val_acc': val_acc,
                         'train_acc': 100.*correct/total})

        # Save checkpoint every epoch (enables resume)
        ckpt = {
            'epoch'      : epoch + 1,
            'model'      : student.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'scheduler'  : scheduler.state_dict(),
            'val_acc'    : val_acc,
        }
        torch.save(ckpt, save_path.replace('.pth', '_latest.pth'))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, save_path)
            print(f"    💾 Best saved ({best_acc:.2f}%)")

    return best_acc, log_rows


# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_pareto(pareto, knee_sol, save_path):
    macs   = np.array([s.macs   / 1e9 for s in pareto])
    accs   = np.array([s.acc            for s in pareto])
    params = np.array([s.params / 1e6  for s in pareto])

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(macs, accs, c=params, cmap='plasma', s=80,
                    edgecolors='black', linewidth=0.3, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Parameters (M)')

    ax.scatter(knee_sol.macs/1e9, knee_sol.acc,
               c=[[knee_sol.params/1e6]], cmap='plasma',
               vmin=params.min(), vmax=params.max(),
               s=160, edgecolors='black', linewidth=1.5,
               zorder=10, label='Knee Point')
    ax.annotate(
        f'Knee Point\nMACs={knee_sol.macs/1e9:.3f}G\n'
        f'Proxy={knee_sol.acc:.3f}\nParams={knee_sol.params/1e6:.2f}M',
        xy=(knee_sol.macs/1e9, knee_sol.acc),
        xytext=(knee_sol.macs/1e9 + 0.005, knee_sol.acc - 0.02),
        arrowprops=dict(arrowstyle='->', lw=1),
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.9),
    )
    ax.set_xlabel('MACs (G)')
    ax.set_ylabel('Taylor Proxy Fitness')
    ax.set_title('NSGA-II Pareto Front — ResNet-56 (representative run)')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Pareto plot saved → {save_path}")


def plot_training(log_rows, save_path):
    epochs = [r['epoch']    for r in log_rows]
    accs   = [r['val_acc']  for r in log_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, accs, color='steelblue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('Fine-tuning Progress (KD)')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Training curve saved → {save_path}")


# ==============================================================================
# CHECKPOINT HELPERS
# ==============================================================================
def save_json(obj, path):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.integer, np.int32, np.int64)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, cls=NumpyEncoder)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def checkpoint_exists(path):
    return os.path.isfile(path)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      default='CIFAR10',
                        choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--target_macs',  default=0.30, type=float,
                        help='Target MACs ratio (e.g. 0.30 = 30%% of baseline)')
    parser.add_argument('--pop_size',     default=50,  type=int)
    parser.add_argument('--generations',  default=30,  type=int)
    parser.add_argument('--kd_epochs',    default=150, type=int)
    parser.add_argument('--resume',       default='auto',
                        help='"auto" resumes from latest checkpoint; '
                             '"none" starts fresh')
    parser.add_argument('--n_rank_batches', default=None, type=int,
                        help='Limit ranking to N batches (None=full epoch)')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--n_search_runs', default=1, type=int,
                        help='Independent NSGA-II runs for search statistics')
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    cfg = Config()
    cfg.dataset_class = args.dataset
    cfg.pop_size      = args.pop_size
    cfg.generations   = args.generations
    cfg.kd_epochs     = args.kd_epochs
    cfg.seed          = args.seed

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.use_cuda:
        torch.cuda.manual_seed_all(cfg.seed)

    for d in [cfg.checkpoint_dir, cfg.log_dir, cfg.figure_dir]:
        os.makedirs(d, exist_ok=True)

    nc = 10 if cfg.dataset_class == 'CIFAR10' else 100
    tag = (f"resnet56_{cfg.dataset_class.lower()}"
          f"_macs{int(args.target_macs*100)}"
          f"_seed{cfg.seed}")

    paths = {
        'baseline'    : f"{cfg.checkpoint_dir}/baseline_{cfg.dataset_class}.pth",
        'pareto_json' : f"{cfg.log_dir}/{tag}_pareto.json",
        'knee_json'   : f"{cfg.log_dir}/{tag}_knee.json",
        'student_best': f"{cfg.checkpoint_dir}/{tag}_student_best.pth",
        'student_last': f"{cfg.checkpoint_dir}/{tag}_student_latest.pth",
        'pareto_fig'  : f"{cfg.figure_dir}/{tag}_pareto.png",
        'train_fig'   : f"{cfg.figure_dir}/{tag}_training.png",
        'summary_json': f"{cfg.log_dir}/{tag}_summary.json",
        'search_stats': f"{cfg.log_dir}/{tag}_search_stats.json",
    }

    print("=" * 70)
    print(f"ResNet-56 GA Pruning | {cfg.dataset_class} | "
          f"Target MACs: {args.target_macs*100:.0f}%")
    print(f"Device: {cfg.device}")
    print("=" * 70)

    train_loader, test_loader = get_data_loaders(cfg)
    criterion_ce = nn.CrossEntropyLoss()

    # ── Stage 0: Baseline training (skip if checkpoint exists) ────────────────
    # Stage 0 always reuses existing baseline regardless of --resume
    if checkpoint_exists(paths['baseline']):
        print(f"\n[Stage 0] Loading baseline from {paths['baseline']}")
        ckpt    = torch.load(paths['baseline'], map_location=cfg.device,
                          weights_only=False)
        teacher = build_resnet56(nc).to(cfg.device)
        # Strip _orig_mod prefix if checkpoint was saved from compiled model
        state = ckpt['model']
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        teacher.load_state_dict(state)
        print(f"  Baseline acc: {ckpt.get('acc', 'N/A')}%")
    else:
        print("\n[Stage 0] Training baseline ResNet-56...")
        print("  ⏱  Estimated time: ~2h on RTX 4090 (200 epochs, CIFAR-10)")
        teacher   = build_resnet56(nc).to(cfg.device)
        # CIFAR-100 needs different schedule than CIFAR-10
        # Use cosine annealing + warmup for better convergence
        n_epochs   = 200
        warmup_ep  = 5
        base_lr    = 0.1
        wd         = 5e-4 if cfg.dataset_class == 'CIFAR100' else 1e-4

        optimizer = optim.SGD(teacher.parameters(), lr=base_lr,
                              momentum=0.9, weight_decay=wd, nesterov=True)

        # Cosine annealing after warmup — better than MultiStepLR for CIFAR-100
        def warmup_cosine(epoch):
            if epoch < warmup_ep:
                return (epoch + 1) / warmup_ep
            progress = (epoch - warmup_ep) / (n_epochs - warmup_ep)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)

        # torch.compile for baseline training speedup
        # Keep reference to uncompiled model for MACs counting later
        teacher_uncompiled = teacher
        try:
            teacher = torch.compile(teacher)
            print("   torch.compile enabled for baseline training ✅")
        except Exception as e:
            print(f"   torch.compile skipped ({e})")

        best_base = 0.0
        for epoch in range(n_epochs):
            teacher.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
                optimizer.zero_grad()
                loss = criterion_ce(teacher(inputs), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
                optimizer.step()
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                teacher.eval()
                correct = total = 0
                with torch.no_grad():
                    for inp, tgt in test_loader:
                        inp, tgt = inp.to(cfg.device), tgt.to(cfg.device)
                        correct += teacher(inp).max(1)[1].eq(tgt).sum().item()
                        total   += tgt.size(0)
                acc = 100. * correct / total
                lr_now = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}: {acc:.2f}% (lr={lr_now:.5f})")
                if acc > best_base:
                    best_base = acc
                    # Save uncompiled weights (strip _orig_mod prefix if compiled)
                    state = teacher.state_dict()
                    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
                    torch.save({'model': state, 'acc': acc,
                                'epoch': epoch+1}, paths['baseline'])
        print(f"  Baseline training done. Best acc: {best_base:.2f}%")

    # ── Stage 1: Build ranking tables ─────────────────────────────────────────
    # Use uncompiled model for torch_pruning compatibility
    teacher_for_pruning = getattr(teacher, '_orig_mod', teacher)
    ga_pruner = GAPrunerResNet(teacher_for_pruning, cfg)

    ranking_path = f"{cfg.log_dir}/{tag}_ranking_tables.npz"
    if checkpoint_exists(ranking_path) and args.resume != 'none':
        print(f"\n[Stage 1] Loading cached ranking tables from {ranking_path}")
        data = np.load(ranking_path, allow_pickle=True)
        ga_pruner.ranking_tables = data['ranking_tables'].item()
        ga_pruner.layer_stats    = data['layer_stats'].item()
    else:
        print("\n[Stage 1] Building Taylor ranking tables...")
        print("  ⏱  Estimated time: ~90s (full epoch) or ~15s (50 batches)")
        ga_pruner.build_ranking_tables(
            train_loader, criterion_ce,
            n_batches=args.n_rank_batches
        )
        np.savez(ranking_path,
                 ranking_tables=ga_pruner.ranking_tables,
                 layer_stats=ga_pruner.layer_stats)
        print(f"  Ranking tables cached → {ranking_path}")

    # ── Stage 2: NSGA-II search ───────────────────────────────────────────────
    if checkpoint_exists(paths['pareto_json']) and args.resume != 'none':
        print(f"\n[Stage 2] Loading cached Pareto front from {paths['pareto_json']}")
        pareto_data = load_json(paths['pareto_json'])
        pareto = [NSGASolution(**d) for d in pareto_data]
        knee_data = load_json(paths['knee_json'])
        knee_sol  = NSGASolution(**knee_data)
    else:
        print("\n[Stage 2] Running NSGA-II search...")
        print("  ⏱  Estimated time: ~20s (pop=50, gen=30)")
        nsga   = NSGAPrunerResNet(ga_pruner, cfg.pop_size, cfg.generations,
                                  cfg.min_ratio)
        pareto = nsga.run()
        knee_sol = nsga.select_knee(pareto)

        # Multiple search runs for statistical validation
        if args.n_search_runs > 1:
            print(f"  Running {args.n_search_runs} independent NSGA-II runs for statistics...")
            all_knee_macs   = []
            all_knee_acc    = []
            all_knee_params = []
            best_knee = knee_sol
            for run_i in range(args.n_search_runs):
                random.seed(cfg.seed + run_i)
                np.random.seed(cfg.seed + run_i)
                nsga_r = NSGAPrunerResNet(ga_pruner, cfg.pop_size,
                                          cfg.generations, cfg.min_ratio)
                pareto_r = nsga_r.run()
                knee_r   = nsga_r.select_knee(pareto_r)
                all_knee_macs.append(knee_r.macs / 1e9)
                all_knee_acc.append(knee_r.acc)
                all_knee_params.append(knee_r.params / 1e6)
                if knee_r.acc > best_knee.acc:
                    best_knee = knee_r
                if (run_i + 1) % 10 == 0:
                    print(f"    Run {run_i+1}/{args.n_search_runs}")
            knee_sol = best_knee
            search_stats = {
                'n_runs'          : args.n_search_runs,
                'knee_macs_mean'  : float(np.mean(all_knee_macs)),
                'knee_macs_std'   : float(np.std(all_knee_macs)),
                'knee_acc_mean'   : float(np.mean(all_knee_acc)),
                'knee_acc_std'    : float(np.std(all_knee_acc)),
                'knee_params_mean': float(np.mean(all_knee_params)),
                'knee_params_std' : float(np.std(all_knee_params)),
            }
            save_json(search_stats, paths['search_stats'])
            print(f"  Search stats: proxy_acc = {search_stats['knee_acc_mean']:.4f} "
                  f"± {search_stats['knee_acc_std']:.4f} over {args.n_search_runs} runs")

        pareto_data = [{'chrom': s.chrom, 'acc': s.acc,
                        'macs': s.macs, 'params': s.params} for s in pareto]
        save_json(pareto_data, paths['pareto_json'])
        save_json({'chrom': knee_sol.chrom, 'acc': knee_sol.acc,
                   'macs': knee_sol.macs, 'params': knee_sol.params},
                  paths['knee_json'])
        print(f"  Pareto front saved → {paths['pareto_json']}")

    print(f"\n  Pareto front size: {len(pareto)}")
    print(f"  Knee: MACs={knee_sol.macs/1e9:.3f}G | "
          f"Proxy={knee_sol.acc:.4f} | Params={knee_sol.params/1e6:.2f}M")

    plot_pareto(pareto, knee_sol, paths['pareto_fig'])

    # ── Stage 3: Physical pruning ─────────────────────────────────────────────
    pruned_path = f"{cfg.checkpoint_dir}/{tag}_pruned_arch.pth"
    if checkpoint_exists(pruned_path) and args.resume != 'none':
        print(f"\n[Stage 3] Loading pruned architecture from {pruned_path}")
        # Rebuild architecture then load weights
        pruned_model = physical_prune_resnet56(
            copy.deepcopy(teacher_for_pruning), ga_pruner, knee_sol, cfg)
        ckpt = torch.load(pruned_path, map_location=cfg.device)
        pruned_model.load_state_dict(ckpt['model'])
    else:
        print("\n[Stage 3] Applying physical pruning...")
        print("  ⏱  Estimated time: ~5s")
        pruned_model = physical_prune_resnet56(
            copy.deepcopy(teacher_for_pruning), ga_pruner, knee_sol, cfg)

        if HAS_TP:
            dummy = torch.randn(1, 3, 32, 32).to(cfg.device)
            final_macs, final_params = tp.utils.count_ops_and_params(pruned_model, dummy)
            print(f"  Final MACs:   {final_macs/1e9:.4f}G "
                  f"({100*(1-final_macs/ga_pruner.base_macs):.1f}% reduction)")
            print(f"  Final Params: {final_params/1e6:.4f}M "
                  f"({100*(1-final_params/ga_pruner.base_params):.1f}% reduction)")

        torch.save({'model': pruned_model.state_dict()}, pruned_path)
        print(f"  Pruned architecture saved → {pruned_path}")

    # ── Stage 4: Fine-tuning with KD ──────────────────────────────────────────
    resume_epoch = 0
    if checkpoint_exists(paths['student_last']) and args.resume != 'none':
        print(f"\n[Stage 4] Resuming KD fine-tuning from {paths['student_last']}")
        ckpt = torch.load(paths['student_last'], map_location=cfg.device)
        pruned_model.load_state_dict(ckpt['model'])
        resume_epoch = ckpt.get('epoch', 0)
        print(f"  Resuming from epoch {resume_epoch}")

    if resume_epoch < cfg.kd_epochs:
        print(f"\n  ⏱  Estimated remaining time: "
              f"~{(cfg.kd_epochs - resume_epoch) * 72 / cfg.kd_epochs:.1f}h "
              f"(CIFAR-10) or ~{(cfg.kd_epochs - resume_epoch) * 96 / cfg.kd_epochs:.1f}h "
              f"(CIFAR-100)")
        best_acc, log_rows = fine_tune_kd(
            student=pruned_model,
            teacher=teacher_for_pruning,
            train_loader=train_loader,
            test_loader=test_loader,
            config=cfg,
            save_path=paths['student_best'],
            resume_epoch=resume_epoch,
        )
        plot_training(log_rows, paths['train_fig'])
    else:
        print(f"\n[Stage 4] KD already complete ({resume_epoch} epochs).")
        best_acc = torch.load(paths['student_best'])['val_acc']

    # ── Summary ───────────────────────────────────────────────────────────────
    if HAS_TP:
        dummy      = torch.randn(1, 3, 32, 32).to(cfg.device)
        pruned_model.eval()
        fm, fp_    = tp.utils.count_ops_and_params(pruned_model, dummy)
        macs_red   = 100 * (1 - fm / ga_pruner.base_macs)
        params_red = 100 * (1 - fp_ / ga_pruner.base_params)
    else:
        fm, fp_ = knee_sol.macs, knee_sol.params
        macs_red = params_red = 0.0

    summary = {
        'dataset'        : cfg.dataset_class,
        'target_macs'    : args.target_macs,
        'baseline_macs'  : ga_pruner.base_macs / 1e9,
        'baseline_params': ga_pruner.base_params / 1e6,
        'pruned_macs'    : fm / 1e9,
        'pruned_params'  : fp_ / 1e6,
        'macs_reduction' : macs_red,
        'params_reduction': params_red,
        'best_val_acc'   : best_acc,
        'pareto_size'    : len(pareto),
        'knee_proxy_acc' : knee_sol.acc,
    }
    save_json(summary, paths['summary_json'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
    print("=" * 70)
    print(f"\nAll outputs saved to:")
    print(f"  Checkpoints : {cfg.checkpoint_dir}/")
    print(f"  Logs        : {cfg.log_dir}/")
    print(f"  Figures     : {cfg.figure_dir}/")


if __name__ == '__main__':
    main()
