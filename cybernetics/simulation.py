"""Модели каналов, метрики и цикл Монте-Карло для отчёта по RS/BCH."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class Channel(Protocol):
    def transmit(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


@dataclass
class BSCChannel:
    """Двоичный симметричный канал: инверсия каждого бита с вероятностью p."""

    p: float

    def transmit(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        bits = np.asarray(data, dtype=np.uint8).copy()
        flips = rng.random(bits.shape) < self.p
        return bits ^ flips.astype(np.uint8)


@dataclass
class SymbolErrorChannel:
    """Канал символьных ошибок над GF(2^m): символ заменяется случайным."""

    p: float
    field_size: int

    def transmit(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        symbols = np.asarray(data, dtype=np.int64).copy()
        mask = rng.random(symbols.shape) < self.p
        replacements = rng.integers(1, self.field_size, size=symbols.shape)
        symbols[mask] = replacements[mask]
        return symbols


@dataclass
class BurstChannel:
    """Пакетный канал: пакеты длины burst_len инвертируют подряд идущие биты."""

    p: float
    burst_len: int = 8

    def transmit(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        bits = np.asarray(data, dtype=np.uint8).copy()
        n = bits.size
        i = 0
        while i < n:
            if rng.random() < self.p:
                end = min(i + self.burst_len, n)
                bits[i:end] ^= 1
                i = end
            else:
                i += 1
        return bits


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float, float]:
    """95% доверительный интервал для доли успехов (Wilson score)."""
    if trials == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / trials
    denom = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denom
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


def z_test_proportions(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Двусторонний z-тест для сравнения двух долей (FER)."""
    from math import erf, sqrt

    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return z, p_value


@dataclass
class SimulationResult:
    ber: float
    fer: float
    ber_ci: tuple[float, float, float]
    fer_ci: tuple[float, float, float]
    decode_time_sec: float
    n_blocks: int
    n_bit_errors: int
    n_frame_errors: int


def run_monte_carlo(
    encode_fn: Callable[[], np.ndarray],
    decode_fn: Callable[[np.ndarray], tuple[np.ndarray, bool]],
    to_bits: Callable[[np.ndarray], np.ndarray],
    from_bits: Callable[[np.ndarray], np.ndarray],
    channel: Channel,
    n_blocks: int,
    seed: int = 42,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    n_bit_errors = 0
    n_frame_errors = 0
    n_bits_total = 0
    t0 = time.perf_counter()

    for _ in range(n_blocks):
        codeword = encode_fn()
        bits = to_bits(codeword)
        n_bits_total += bits.size
        received_bits = channel.transmit(bits, rng)
        n_bit_errors += int(np.count_nonzero(received_bits != bits))
        received = from_bits(received_bits)
        try:
            decoded, ok = decode_fn(received)
            frame_ok = ok and np.array_equal(decoded, codeword[: len(decoded)])
        except Exception:
            frame_ok = False
        if not frame_ok:
            n_frame_errors += 1

    elapsed = time.perf_counter() - t0
    ber_ci = wilson_ci(n_bit_errors, n_bits_total)
    fer_ci = wilson_ci(n_frame_errors, n_blocks)
    return SimulationResult(
        ber=ber_ci[0],
        fer=fer_ci[0],
        ber_ci=ber_ci,
        fer_ci=fer_ci,
        decode_time_sec=elapsed,
        n_blocks=n_blocks,
        n_bit_errors=n_bit_errors,
        n_frame_errors=n_frame_errors,
    )
