"""
tests.py
--------
Test numerici del progetto: verifica della DCT 1D e della DCT2 2D
sui blocchi di riferimento forniti nella specifica.

Eseguibile direttamente::

    python tests.py

oppure tramite l'opzione ``--test`` della GUI::

    python gui.py --test
"""

import numpy as np
from scipy.fft import dct, dctn

from dct_compression import apply_dct2


# ---------------------------------------------------------------------------
# Dati di riferimento (dalla specifica del progetto)
# ---------------------------------------------------------------------------

_TEST_VECTOR_1D = np.array([231, 32, 233, 161, 24, 71, 140, 245], dtype=float)

_EXPECTED_DCT1D = np.array([
    4.01e+02,  6.60e+00,  1.09e+02, -1.12e+02,
    6.54e+01,  1.21e+02,  1.16e+02,  2.88e+01,
])

_TEST_BLOCK_8X8 = np.array([
    [231,  32, 233, 161,  24,  71, 140, 245],
    [247,  40, 248, 245, 124, 204,  36, 107],
    [234, 202, 245, 167,   9, 217, 239, 173],
    [193, 190, 100, 167,  43, 180,   8,  70],
    [ 11,  24, 210, 177,  81, 243,   8, 112],
    [ 97, 195, 203,  47, 125, 114, 165, 181],
    [193,  70, 174, 167,  41,  30, 127, 245],
    [ 87, 149,  57, 192,  65, 129, 178, 228],
], dtype=float)

_EXPECTED_DCT2D = np.array([
    [ 1.11e+03,  4.40e+01,  7.59e+01, -1.38e+02,  3.50e+00,  1.22e+02,  1.95e+02, -1.01e+02],
    [ 7.71e+01,  1.14e+02, -2.18e+01,  4.13e+01,  8.77e+00,  9.90e+01,  1.38e+02,  1.09e+01],
    [ 4.48e+01, -6.27e+01,  1.11e+02, -7.63e+01,  1.24e+02,  9.55e+01, -3.98e+01,  5.85e+01],
    [-6.99e+01, -4.02e+01, -2.34e+01, -7.67e+01,  2.66e+01, -3.68e+01,  6.61e+01,  1.25e+02],
    [-1.09e+02, -4.33e+01, -5.55e+01,  8.17e+00,  3.02e+01, -2.86e+01,  2.44e+00, -9.41e+01],
    [-5.38e+00,  5.66e+01,  1.73e+02, -3.54e+01,  3.23e+01,  3.34e+01, -5.81e+01,  1.90e+01],
    [ 7.88e+01, -6.45e+01,  1.18e+02, -1.50e+01, -1.37e+02, -3.06e+01, -1.05e+02,  3.98e+01],
    [ 1.97e+01, -7.81e+01,  9.72e-01, -7.23e+01, -2.15e+01,  8.13e+01,  6.37e+01,  5.90e+00],
])

# Tolleranza per il confronto relativo (1% ≈ compatibile con 3 cifre significative)
_RELATIVE_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------

def _select_best_norm(
    computed_ortho: np.ndarray,
    computed_nonorm: np.ndarray,
    expected: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Confronta le due varianti (norm='ortho' e norm=None) con i valori attesi
    e restituisce quella con errore massimo assoluto minore.
    """
    err_ortho  = np.max(np.abs(computed_ortho  - expected))
    err_nonorm = np.max(np.abs(computed_nonorm - expected))
    if err_nonorm < err_ortho:
        return computed_nonorm, "None (no normalizzazione)"
    return computed_ortho, "ortho"


def _compute_errors(
    result: np.ndarray,
    expected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Calcola errori assoluti e relativi elemento per elemento.

    Returns
    -------
    abs_err, rel_err, max_rel_err, max_abs_err
    """
    abs_err = np.abs(result - expected)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(
            np.abs(expected) > 1e-6,
            abs_err / np.abs(expected),
            abs_err,
        )
    return abs_err, rel_err, float(np.max(rel_err)), float(np.max(abs_err))


# ---------------------------------------------------------------------------
# Test 1 — DCT monodimensionale
# ---------------------------------------------------------------------------

def _test_dct_1d() -> bool:
    sep = "-" * 72
    print(sep)
    print("TEST 1 \u2014 DCT monodimensionale (prima riga del blocco 8\u00d78)")
    print(sep)

    computed_ortho  = dct(_TEST_VECTOR_1D, type=2, norm="ortho")
    computed_nonorm = dct(_TEST_VECTOR_1D, type=2, norm=None)
    result, norm_used = _select_best_norm(computed_ortho, computed_nonorm, _EXPECTED_DCT1D)

    abs_err, rel_err, max_rel, max_abs = _compute_errors(result, _EXPECTED_DCT1D)
    worst_idx = int(np.argmax(rel_err))
    passed    = max_rel < _RELATIVE_TOLERANCE

    print(f"  Normalizzazione selezionata : {norm_used}")
    print(f"  Risultato  : {np.array2string(result,         precision=2, suppress_small=True)}")
    print(f"  Atteso     : {np.array2string(_EXPECTED_DCT1D, precision=2, suppress_small=True)}")
    print(
        f"  Errore relativo max : {max_rel * 100:.4f}%  "
        f"(indice {worst_idx}: calcolato={result[worst_idx]:.4f}, "
        f"atteso={_EXPECTED_DCT1D[worst_idx]:.4f}, "
        f"errore assoluto={abs_err[worst_idx]:.4f})"
    )
    print(f"  Esito      : {'PASSATO' if passed else 'FALLITO'}")
    print()
    return passed


# ---------------------------------------------------------------------------
# Test 2 — DCT2 bidimensionale
# ---------------------------------------------------------------------------

def _test_dct_2d() -> bool:
    sep = "-" * 72
    print(sep)
    print("TEST 2 \u2014 DCT2 sul blocco 8\u00d78")
    print(sep)

    computed_ortho  = apply_dct2(_TEST_BLOCK_8X8)
    computed_nonorm = dctn(_TEST_BLOCK_8X8, type=2, norm=None)
    result, norm_used = _select_best_norm(computed_ortho, computed_nonorm, _EXPECTED_DCT2D)

    abs_err, rel_err, max_rel, max_abs = _compute_errors(result, _EXPECTED_DCT2D)
    worst_idx = np.unravel_index(np.argmax(rel_err), rel_err.shape)
    passed    = max_rel < _RELATIVE_TOLERANCE

    print(f"  Normalizzazione selezionata : {norm_used}")
    print("  Risultato:")
    print(np.array2string(result,         precision=2, suppress_small=True, prefix="    "))
    print("  Atteso:")
    print(np.array2string(_EXPECTED_DCT2D, precision=2, suppress_small=True, prefix="    "))
    print(
        f"  Errore relativo max : {max_rel * 100:.4f}%  "
        f"(posizione {worst_idx}: calcolato={result[worst_idx]:.4f}, "
        f"atteso={_EXPECTED_DCT2D[worst_idx]:.4f}, "
        f"errore assoluto={abs_err[worst_idx]:.4f})"
    )
    print(f"  Esito      : {'PASSATO' if passed else 'FALLITO'}")
    print()
    return passed


# ---------------------------------------------------------------------------
# Runner principale
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Esegue tutti i test numerici e stampa il risultato complessivo."""
    sep = "-" * 72
    passed_1d = _test_dct_1d()
    passed_2d = _test_dct_2d()
    all_passed = passed_1d and passed_2d
    print(sep)
    print(
        f"RISULTATO COMPLESSIVO: "
        f"{'TUTTI I TEST PASSATI' if all_passed else 'ALCUNI TEST FALLITI'}"
    )
    print(sep)


if __name__ == "__main__":
    run_tests()
