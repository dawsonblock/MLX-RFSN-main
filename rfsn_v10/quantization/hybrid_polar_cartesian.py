#!/usr/bin/env python3
"""
Hybrid Polar-Cartesian quantizer.
Correct design:
    1. Rotate into IsoQuant space.
    2. Split feature dimension into polar + cartesian partitions.
    3. PolarQuant the polar partition.
    4. Grouped Cartesian quantize the remaining partition.
    5. Dequantize both.
    6. Concatenate.
    7. Apply inverse IsoQuant rotation.
No QJL is applied in dequantize. QJL belongs in attention-score correction.
"""
from __future__ import annotations

from dataclasses import dataclass

from rfsn_v10.compat import mx

from .grouped_cartesian import GroupedCartesianQuantizer, PackedCartesianCodes
from .isoquant_precondition import IsoQuantMetadata, IsoQuantPreconditioner
from .polar_quant import PackedPolarCodes, PolarQuantizer


@dataclass
class HybridPacked:
    polar: PackedPolarCodes | None
    cartesian: PackedCartesianCodes | None
    iso_meta: IsoQuantMetadata
    original_shape: tuple[int, ...]
    split_dim: int
    feature_dim: int
    mode: str = "hybrid_polar_cartesian"


class HybridPolarCartesianQuantizer:
    def __init__(
        self,
        feature_dim: int,
        polar_ratio: float = 0.65,
        polar_levels: int = 4,
        polar_angle_bits: int = 5,
        polar_radius_bits: int = 8,
        cartesian_bits: int = 6,
        group_size: int = 64,
        rotation_seed: int = 42,
        polar_enabled: bool = True,
        adaptive_angle_range: bool = False,
    ):
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if feature_dim % 4 != 0:
            raise ValueError("feature_dim must be divisible by 4")
        if not (0.0 < polar_ratio <= 1.0):
            raise ValueError("polar_ratio must be in (0,1]")
        self.feature_dim = feature_dim
        self.polar_ratio = polar_ratio
        self.polar_levels = polar_levels
        self.polar_enabled = polar_enabled
        self.base = 2**polar_levels
        if polar_enabled:
            self.split_dim = self._choose_split_dim(
                feature_dim, polar_ratio, self.base
            )
        else:
            self.split_dim = 0
        self.iso = IsoQuantPreconditioner(
            feature_dim=feature_dim,
            seed=rotation_seed,
        )
        self.polar = PolarQuantizer(
            levels=polar_levels,
            angle_bits=polar_angle_bits,
            radius_bits=polar_radius_bits,
            radius_group_size=group_size,
            adaptive_angle_range=adaptive_angle_range,
        )
        self.cartesian = GroupedCartesianQuantizer(
            bits=cartesian_bits,
            group_size=group_size,
        )

    @staticmethod
    def _choose_split_dim(
        feature_dim: int, polar_ratio: float, base: int
    ) -> int:
        """
        Choose a polar split that is divisible by 2**levels.
        For D=64, levels=4, base=16, polar_ratio=0.65:
            target = 41.6
            nearest valid = 48
        """
        target = int(round(feature_dim * polar_ratio))
        split = int(round(target / base) * base)
        split = max(base, min(split, feature_dim))
        # If split accidentally becomes 0 or invalid, fall back.
        if split % base != 0:
            split = (split // base) * base
        if split <= 0:
            split = base
        if split > feature_dim:
            split = feature_dim
        return split

    def quantize(self, x: mx.array) -> HybridPacked:
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected D={self.feature_dim}, got D={x.shape[-1]}"
            )
        original_shape = tuple(x.shape)
        rotated, iso_meta = self.iso.forward(x)
        if self.polar_enabled:
            polar_part = rotated[..., :self.split_dim]
            cart_part = rotated[..., self.split_dim:]
            polar_packed = self.polar.quantize(polar_part)
        else:
            polar_packed = None
            cart_part = rotated
        if cart_part.shape[-1] > 0:
            cart_packed = self.cartesian.quantize(cart_part)
        else:
            cart_packed = None
        return HybridPacked(
            polar=polar_packed,
            cartesian=cart_packed,
            iso_meta=iso_meta,
            original_shape=original_shape,
            split_dim=self.split_dim,
            feature_dim=self.feature_dim,
        )

    def dequantize(self, packed: HybridPacked) -> mx.array:
        if self.polar_enabled and packed.polar is not None:
            polar_rec = self.polar.dequantize(packed.polar)
            if packed.cartesian is not None:
                cart_rec = self.cartesian.dequantize(packed.cartesian)
                rotated_rec = mx.concatenate(
                    [polar_rec, cart_rec], axis=-1
                )
            else:
                rotated_rec = polar_rec
        elif packed.cartesian is not None:
            rotated_rec = self.cartesian.dequantize(packed.cartesian)
        else:
            raise ValueError(
                "Packed data has no polar or cartesian component"
            )
        if rotated_rec.shape[-1] != packed.feature_dim:
            raise ValueError(
                f"Reconstructed D={rotated_rec.shape[-1]}, "
                f"expected D={packed.feature_dim}"
            )
        restored = self.iso.inverse(rotated_rec, packed.iso_meta)
        return restored.reshape(packed.original_shape)

    def estimate_bytes(self, packed: HybridPacked) -> int:
        total = 0
        if self.polar_enabled and packed.polar is not None:
            total += self.polar.estimate_bytes(packed.polar)
        if packed.cartesian is not None:
            total += self.cartesian.estimate_bytes(packed.cartesian)
        # IsoQuant metadata is tiny but count it honestly:
        # q_l + q_r = 8 floats.
        total += 8 * 4
        return total

    def estimate_bytes_for_shape(self, shape: tuple[int, ...]) -> int:
        """Analytical byte estimate without materialising arrays."""
        total = 0
        if self.polar_enabled:
            total += self.polar.estimate_bytes_for_shape(shape)
        total += self.cartesian.estimate_bytes_for_shape(shape)
        total += 8 * 4
        return total
