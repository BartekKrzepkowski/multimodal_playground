import logging

import numpy as np
import torch, math, warnings
from torch import nn
from typing import Dict, List

class SimplicityMeter:
    r"""
    Oblicza na macierzy wag W (out, in):

        • tail_index      – wykładnik α rozkładu potęgi σ_i  (σ_i ∝ i^{-1/α})
        • participation   – PR = (Σ σ_i)^2 / Σ σ_i^2  (norm. rozproszenie energii)
        • sq_sum          – Σ σ_i^2  (norma Frobeniusa^2)
    """
    def __init__(self, max_rank: int = 1024):
        """
        max_rank – jeśli min(W.shape) > max_rank, używa losowej SVD (sketch).
        """
        self.max_rank = max_rank
    
    def layer_stats(self, layer_name: str, layer: nn.Module, metrics: Dict, postfix: str) -> Dict[str, float]:
        W = self._extract_weights(layer)        # (out, in)
        if W is None:  # warstwa bez macierzy
            return

        U, S, Vt = self._safe_svd(W)            # full lub przybliżona SVD
        S = S.double()

        # 1) participation ratio
        pr = (S**2).sum()**2 / (S**4).sum()

        # 2) tail-index (α) z 20% największych wartości
        k = max(3, int(0.2 * len(S)))
        tail = S[:k]
        log_i  = torch.log(torch.arange(1, k+1, dtype=S.dtype))
        log_s  = torch.log(tail)
        # konwersja tensorów do numpy (jeśli są torch.Tensor)
        log_i_np = log_i.detach().cpu().numpy()
        log_s_np = log_s.detach().cpu().numpy()

        α, _ = np.polyfit(log_i_np, log_s_np, deg=1)
        tail_idx = -1 / α.item()

        # 3) suma kwadratów
        sq_sum = (S**2).sum().item()

        # 4) rank macierzy
        eps = S[0] * max(W.shape) * torch.finfo(S.dtype).eps
        rank = (S > eps).sum().item()

        # 5) effective rank (Roy & Vetterli, 2007)
        p = S / S.sum()
        H = -(p * torch.log(p + 1e-12)).sum()                # entropia Shannona
        effective_rank = torch.exp(H).item()

        # 6) spectral norm
        spectral_norm = S[0].item()                          # największa σ

        # 7) stable rank (Σσ² / σ₁²)
        stable_rank = (S.pow(2).sum() / S[0]**2).item()

        # 8) long tail index
        spectral_tailness =  effective_rank / stable_rank

        # Logowanie statystyk
        metrics[f'tail_index_weights/{layer_name}{postfix}'] = tail_idx
        metrics[f'participation_weights/{layer_name}{postfix}'] = pr.item()
        metrics[f'sq_sum_weights/{layer_name}{postfix}'] = sq_sum
        metrics[f'rank_weights/{layer_name}{postfix}'] = rank
        metrics[f'effective_rank_weights/{layer_name}{postfix}'] = effective_rank
        metrics[f'spectral_norm_weights/{layer_name}{postfix}'] = spectral_norm
        metrics[f'stable_rank_weights/{layer_name}{postfix}'] = stable_rank
        metrics[f'spectral_tailness_weights/{layer_name}{postfix}'] = spectral_tailness

    def model_report(self, model, logger, global_step, scope, phase) -> List[Dict]:
        model.eval()  # przełącz model w tryb ewaluacji
        metrics = {}
        postfix = f'____{scope}____{phase}'
        for layer_name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                self.layer_stats(layer_name, layer, metrics, postfix)

        model.train()  # przywróć tryb treningowy
        logger.log_scalars(metrics, global_step) 

    # ---------- Helpers ----------
    def _extract_weights(self, layer):
        if isinstance(layer, nn.Linear):
            return layer.weight.detach().cpu()
        if isinstance(layer, nn.Conv2d):
            out_c, in_c, k1, k2 = layer.weight.shape
            return layer.weight.detach().cpu().reshape(out_c, in_c * k1 * k2)
        return None

    def _safe_svd(self, W: torch.Tensor):
        m, n = W.shape
        rank_cut = min(m, n)
        try:
            if rank_cut <= self.max_rank:
                # Spróbuj normalnego SVD
                return torch.linalg.svd(W, full_matrices=False)
            # Przybliżona SVD (randomized)
            warnings.warn(f"Using sketch SVD for shape {W.shape}", RuntimeWarning)
            q = self.max_rank
            G = torch.randn(n, q, device=W.device, dtype=W.dtype)
            Y = W @ G
            Q, _ = torch.linalg.qr(Y, mode='reduced')
            B = Q.T @ W
            Ub, S, Vt = torch.linalg.svd(B, full_matrices=False)
            U = Q @ Ub
            return U, S, Vt
        except Exception as e:
            logging.warning(f"SVD failed for matrix {W.shape}, attempting stabilization.")

            # 1. Sprawdź NaN/Inf
            if torch.isnan(W).any() or torch.isinf(W).any():
                logging.warning(f"Matrix {W.shape} contains NaN or Inf values, replacing them.")
                W = torch.nan_to_num(W, nan=0.0, posinf=1e6, neginf=-1e6)

            # 2. Dodaj bardzo mały szum
            eps = 1e-6
            W = W + eps * torch.randn_like(W)
            logging.info(f"Stabilized matrix {W.shape} by adding noise.")

            # 3. Spróbuj ponownie SVD
            try:
                if rank_cut <= self.max_rank:
                    return torch.linalg.svd(W, full_matrices=False)
                # powtórka sketch SVD
                G = torch.randn(n, self.max_rank, device=W.device, dtype=W.dtype)
                Y = W @ G
                Q, _ = torch.linalg.qr(Y, mode='reduced')
                B = Q.T @ W
                Ub, S, Vt = torch.linalg.svd(B, full_matrices=False)
                U = Q @ Ub
                return U, S, Vt
            except Exception as e2:
                logging.warning(f"Second SVD attempt failed for matrix {W.shape}, falling back to numpy.")

                # 4. Spróbuj numpy (CPU, float64)
                try:
                    W_np = W.detach().cpu().to(torch.float64).numpy()
                    import numpy as np
                    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
                    U = torch.from_numpy(U).to(W.device, W.dtype)
                    S = torch.from_numpy(S).to(W.device, W.dtype)
                    Vt = torch.from_numpy(Vt).to(W.device, W.dtype)
                    logging.info(f"Numpy SVD succeeded for matrix {W.shape}.")
                    return U, S, Vt
                except Exception as e3:
                    logging.error(f"All SVD attempts failed for matrix {W.shape}.")
                    raise RuntimeError("SVD failed for matrix even after stabilization.") from e3
