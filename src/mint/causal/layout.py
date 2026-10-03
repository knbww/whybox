"""Flat unit indexing: the bridge between a model's sites and a (N, U) matrix."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..models.target import Edits, TargetModel, reduce_activation, reduce_gradient


@dataclass
class UnitLayout:
    sites: list[str]
    widths: list[int]
    offsets: dict[str, int]
    n_units: int

    @classmethod
    def of(cls, model: TargetModel) -> "UnitLayout":
        sites, widths, offsets, off = [], [], {}, 0
        for u in model.units:
            if u.site not in offsets:
                offsets[u.site] = off
                sites.append(u.site)
                widths.append(u.site_width)
                off += u.site_width
        return cls(sites, widths, offsets, off)

    def flatten(self, per_site: dict[str, torch.Tensor], grad: bool = False) -> torch.Tensor:
        red = reduce_gradient if grad else reduce_activation
        return torch.cat([red(per_site[s]) for s in self.sites], dim=1)

    def edits(self, mask: torch.Tensor, values: torch.Tensor) -> Edits:
        """(N, U) bool mask + (N, U) scalar-per-unit values -> per-site edits."""
        out: Edits = {}
        for s, w in zip(self.sites, self.widths):
            o = self.offsets[s]
            out[s] = (mask[:, o:o + w], values[:, o:o + w])
        return out

    def edits_from_sites(self, mask: torch.Tensor,
                         site_values: dict[str, torch.Tensor]) -> Edits:
        """(N, U) bool mask + full per-component values per site -> edits.

        The structure-preserving path: each unit is written with the whole
        activation tensor it takes in the reference condition, not a summary.
        """
        out: Edits = {}
        for s, w in zip(self.sites, self.widths):
            o = self.offsets[s]
            out[s] = (mask[:, o:o + w], site_values[s])
        return out

    def site_of(self, unit: int) -> tuple[str, int]:
        for s, w in zip(self.sites, self.widths):
            o = self.offsets[s]
            if o <= unit < o + w:
                return s, unit - o
        raise IndexError(unit)
