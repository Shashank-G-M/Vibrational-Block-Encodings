"""Matrix-free single-GPU eigensolver for two-mode vibrational Hamiltonians.

This module applies a vibrational Hamiltonian directly in the modal-product
basis without factorizing the two-body tensor into global fragments.  The
Hamiltonian is assumed to contain one-mode and two-mode terms only,

    H = sum_i h_i + sum_{i>j} g_ij,

where ``obt[i, p, q]`` stores the one-mode matrix elements and
``tbt_nonsym[i, p, q, j, r, s]`` stores the two-mode matrix elements.

Two-body tensor convention
--------------------------
``tbt_nonsym`` uses the convention produced by
``tensor_utils.unsymmetrize_tbt`` in this project: physical two-mode blocks are
stored only for ``i > j``.  For one such block,

    tbt_nonsym[i, p, q, j, r, s]

multiplies |p_i><q_i| |r_j><s_j|.  Blocks with ``i <= j`` must be zero.

Compilation-cache design
------------------------
For a fixed ``nmodes``, ``nmodals`` and LOBPCG iteration limit, the compiled
program is independent of the numerical values of the Hamiltonian.  The
one-body coefficients, all lower-triangular two-mode pair matrices, and the
spectral shift are passed to the compiled eigensolver as dynamic JAX operands.
Consequently, different molecular geometries with the same tensor shapes and
solver structure can reuse the same persistent JAX/XLA compilation cache.

All lower-triangular mode pairs are represented in a fixed canonical order,
including pairs whose coefficient block is exactly zero.  For ``nmodes=12``
this gives exactly 66 pair matrices of shape ``25 x 25`` when ``nmodals=5``.
This fixed layout is important for compilation reuse across Hamiltonians.

The implementation targets one accelerator.  It does not introduce multi-GPU
sharding or multi-node communication.  Full-state Hamiltonian terms are
accumulated with rolled JAX control flow so that the 12 one-mode and 66
two-mode contributions are not simultaneously materialized as
Hilbert-space-sized temporaries.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np


# The original eigensolver uses double precision.  Enable float64 before any
# JAX arrays are constructed so that both the Hamiltonian action and LOBPCG
# operate in float64.
jax.config.update("jax_enable_x64", True)


PairModes = Tuple[Tuple[int, int], ...]


def _canonical_pair_modes(nmodes: int) -> PairModes:
    """Return all lower-triangular mode pairs in a deterministic order.

    The order is

        (1, 0), (2, 0), (2, 1), (3, 0), ...

    and depends only on ``nmodes``, never on Hamiltonian coefficients.
    """
    return tuple((i, j) for i in range(1, nmodes) for j in range(i))


def _validate_inputs(obt: np.ndarray, tbt_nonsym: np.ndarray) -> Tuple[int, int]:
    """Validate tensor shapes and the lower-triangular two-mode convention.

    Parameters
    ----------
    obt
        One-body tensor with shape ``(nmodes, nmodals, nmodals)``.
    tbt_nonsym
        Nonsymmetrized two-body tensor with shape
        ``(nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)``.  Only
        ``i > j`` mode blocks may be nonzero.

    Returns
    -------
    nmodes, nmodals
        Number of vibrational modes and number of modals per mode.
    """
    if obt.ndim != 3:
        raise ValueError(
            "obt must have shape (nmodes, nmodals, nmodals); "
            f"received an array with shape {obt.shape}."
        )

    nmodes = int(obt.shape[0])
    nmodals = int(obt.shape[1])

    if obt.shape != (nmodes, nmodals, nmodals):
        raise ValueError(
            "obt must have shape (nmodes, nmodals, nmodals); "
            f"received {obt.shape}."
        )

    expected_tbt_shape = (
        nmodes,
        nmodals,
        nmodals,
        nmodes,
        nmodals,
        nmodals,
    )
    if tbt_nonsym.shape != expected_tbt_shape:
        raise ValueError(
            "tbt_nonsym must have shape "
            "(nmodes, nmodals, nmodals, nmodes, nmodals, nmodals); "
            f"expected {expected_tbt_shape}, received {tbt_nonsym.shape}."
        )

    if nmodes < 1:
        raise ValueError("nmodes must be at least 1.")
    if nmodals < 1:
        raise ValueError("nmodals must be at least 1.")

    if not np.issubdtype(obt.dtype, np.number):
        raise TypeError("obt must contain numeric values.")
    if not np.issubdtype(tbt_nonsym.dtype, np.number):
        raise TypeError("tbt_nonsym must contain numeric values.")

    if not np.all(np.isfinite(obt)):
        raise ValueError("obt contains NaN or infinite values.")
    if not np.all(np.isfinite(tbt_nonsym)):
        raise ValueError("tbt_nonsym contains NaN or infinite values.")

    # tensor_utils.unsymmetrize_tbt stores physical two-mode blocks only in
    # the lower mode triangle.  Reject a symmetrized or upper-triangular tensor
    # rather than silently changing the Hamiltonian by double counting terms.
    scale = max(1.0, float(np.max(np.abs(tbt_nonsym))))
    triangle_atol = 1.0e-13 * scale
    for i in range(nmodes):
        for j in range(i, nmodes):
            block = tbt_nonsym[i, :, :, j, :, :]
            if not np.allclose(block, 0.0, rtol=0.0, atol=triangle_atol):
                raise ValueError(
                    "tbt_nonsym must use the lower-triangular mode convention "
                    "with nonzero blocks only for i > j.  A nonzero block was "
                    f"found at (i, j) = ({i}, {j})."
                )

    return nmodes, nmodals


def prepare_tbt_pair_operators(
    tbt_nonsym: np.ndarray,
) -> Tuple[np.ndarray, PairModes]:
    """Pack every lower-triangular two-mode block into a local operator.

    Each physical mode pair ``(i, j)`` with ``i > j`` acts on a local Hilbert
    space of dimension ``nmodals**2``.  The coefficient block

        tbt_nonsym[i, p, q, j, r, s]

    is converted to a matrix whose row index is ``(r, p)`` and whose column
    index is ``(s, q)``.  This ordering follows the wavefunction axis order:
    mode ``j`` occurs before mode ``i`` because ``j < i``.

    All lower-triangular pairs are retained, including exactly-zero blocks.
    Keeping a fixed number and order of pair matrices makes the compiled JAX
    program independent of which particular couplings vanish at a geometry.

    Parameters
    ----------
    tbt_nonsym
        Nonsymmetrized two-body tensor.  Only ``i > j`` blocks are used.

    Returns
    -------
    pair_ops
        Array with shape
        ``(nmodes * (nmodes - 1) // 2, nmodals**2, nmodals**2)``.
    pair_modes
        Canonical tuple of ``(i, j)`` mode-index pairs corresponding to
        ``pair_ops``.
    """
    tbt_nonsym = np.asarray(tbt_nonsym)
    if tbt_nonsym.ndim != 6:
        raise ValueError("tbt_nonsym must be a rank-6 tensor.")

    nmodes = int(tbt_nonsym.shape[0])
    nmodals = int(tbt_nonsym.shape[1])
    pair_dim = nmodals * nmodals
    pair_modes = _canonical_pair_modes(nmodes)

    pair_ops = np.empty(
        (len(pair_modes), pair_dim, pair_dim), dtype=tbt_nonsym.dtype
    )

    for pair_index, (mode_i, mode_j) in enumerate(pair_modes):
        # Raw local block axes are (p, q, r, s).  The state axes occur in the
        # order (j, i), so the local matrix uses output rows (r, p) and input
        # columns (s, q).  This transpose acts only on the small local block.
        block = tbt_nonsym[mode_i, :, :, mode_j, :, :]
        pair_ops[pair_index] = np.transpose(block, (2, 0, 3, 1)).reshape(
            pair_dim, pair_dim
        )

    return np.ascontiguousarray(pair_ops), pair_modes


def _apply_one_mode_operator(
    op: jax.Array,
    state: jax.Array,
    mode: int,
    nmodes: int,
    nmodals: int,
) -> jax.Array:
    """Apply one local ``nmodals x nmodals`` operator to one state axis.

    ``state`` has shape ``(nmodals,) * nmodes + (nvec,)``.  The final axis is
    an eigensolver block dimension.  Reshaping exposes the selected modal axis
    without explicitly moving wavefunction axes.
    """
    nvec = state.shape[-1]
    pre_dim = nmodals**mode
    post_dim = nmodals ** (nmodes - mode - 1)

    state_view = state.reshape((pre_dim, nmodals, post_dim * nvec))
    result_view = jnp.einsum(
        "pq,aqb->apb",
        op,
        state_view,
        optimize=True,
    )
    return result_view.reshape(state.shape)


def _apply_two_mode_operator(
    pair_op: jax.Array,
    state: jax.Array,
    mode_i: int,
    mode_j: int,
    nmodes: int,
    nmodals: int,
) -> jax.Array:
    """Apply one exact two-mode operator for a pair ``mode_i > mode_j``.

    ``pair_op`` has row index ``(r, p)`` and column index ``(s, q)``, where
    ``p, q`` belong to mode ``mode_i`` and ``r, s`` belong to mode ``mode_j``.
    The source code does not explicitly transpose or move the full wavefunction
    to make distant modal axes adjacent.

    For adjacent modes, the two local axes are contiguous and the operation is
    evaluated as an explicit ``nmodals**2 x nmodals**2`` matrix action.  For
    nonadjacent modes, the same local matrix is viewed as a rank-4 tensor and
    contracted directly against the two selected state axes.
    """
    if mode_i <= mode_j:
        raise ValueError("_apply_two_mode_operator requires mode_i > mode_j.")

    nvec = state.shape[-1]
    pre_dim = nmodals**mode_j
    middle_dim = nmodals ** (mode_i - mode_j - 1)
    post_dim = nmodals ** (nmodes - mode_i - 1)

    if mode_i == mode_j + 1:
        # Adjacent state axes are ordered (j, i).  Their combined input index is
        # (s, q) and output index is (r, p), matching pair_op exactly.
        state_view = state.reshape(
            (pre_dim, nmodals * nmodals, post_dim * nvec)
        )
        result_view = jnp.einsum(
            "uv,avb->aub",
            pair_op,
            state_view,
            optimize=True,
        )
        return result_view.reshape(state.shape)

    # For separated axes, expose the state as
    #     (pre, s, middle, q, post_and_block)
    # while preserving canonical modal order.  pair_tensor has axes
    #     (r, p, s, q).
    # Contracting s and q directly produces
    #     (pre, r, middle, p, post_and_block),
    # which is already in canonical output mode order.
    state_view = state.reshape(
        (pre_dim, nmodals, middle_dim, nmodals, post_dim * nvec)
    )
    pair_tensor = pair_op.reshape((nmodals, nmodals, nmodals, nmodals))
    result_view = jnp.einsum(
        "rpsq,asbqc->arbpc",
        pair_tensor,
        state_view,
        optimize=True,
    )
    return result_view.reshape(state.shape)


def _accumulate_obt_on_state(
    obt: jax.Array,
    state: jax.Array,
    accumulator: jax.Array,
    nmodes: int,
    nmodals: int,
) -> jax.Array:
    """Accumulate one-mode terms with a rolled, memory-bounded device loop."""
    if nmodes == 0:
        return accumulator

    branches = []
    for mode in range(nmodes):
        def branch(args, mode=mode):
            op, state_arg, acc_arg = args
            contribution = _apply_one_mode_operator(
                op, state_arg, mode, nmodes, nmodals
            )
            return acc_arg + contribution

        branches.append(branch)
    branches = tuple(branches)

    def body(mode_index, acc):
        op = jax.lax.dynamic_index_in_dim(
            obt, mode_index, axis=0, keepdims=False
        )
        return jax.lax.switch(mode_index, branches, (op, state, acc))

    return jax.lax.fori_loop(
        0, nmodes, body, accumulator, unroll=False
    )


def _accumulate_tbt_on_state(
    pair_ops: jax.Array,
    pair_modes: PairModes,
    state: jax.Array,
    accumulator: jax.Array,
    nmodes: int,
    nmodals: int,
) -> jax.Array:
    """Accumulate all two-mode terms with bounded full-state temporary memory.

    Each physical mode pair has a different static reshape because the acted-on
    modes may be arbitrarily far apart.  One branch is generated for each
    canonical pair.  A rolled ``lax.fori_loop`` executes one pair contribution
    at a time and carries only the running Hamiltonian result.
    """
    npairs = len(pair_modes)
    if npairs == 0:
        return accumulator

    branches = []
    for mode_i, mode_j in pair_modes:
        def branch(args, mode_i=mode_i, mode_j=mode_j):
            pair_op, state_arg, acc_arg = args
            contribution = _apply_two_mode_operator(
                pair_op,
                state_arg,
                mode_i,
                mode_j,
                nmodes,
                nmodals,
            )
            return acc_arg + contribution

        branches.append(branch)
    branches = tuple(branches)

    def body(pair_index, acc):
        pair_op = jax.lax.dynamic_index_in_dim(
            pair_ops, pair_index, axis=0, keepdims=False
        )
        return jax.lax.switch(
            pair_index, branches, (pair_op, state, acc)
        )

    return jax.lax.fori_loop(
        0, npairs, body, accumulator, unroll=False
    )


def _ham_on_state_block(
    obt: jax.Array,
    pair_ops: jax.Array,
    pair_modes: PairModes,
    state: jax.Array,
    nmodes: int,
    nmodals: int,
) -> jax.Array:
    """Apply the complete one- plus two-mode Hamiltonian to a state block."""
    accumulator = jnp.zeros_like(state)
    accumulator = _accumulate_obt_on_state(
        obt, state, accumulator, nmodes, nmodals
    )
    accumulator = _accumulate_tbt_on_state(
        pair_ops, pair_modes, state, accumulator, nmodes, nmodals
    )
    return accumulator


@partial(
    jax.jit,
    static_argnames=("pair_modes", "nmodes", "nmodals"),
)
def _Ham_on_state_batched_prepared(
    obt: jax.Array,
    pair_ops: jax.Array,
    block_state: jax.Array,
    pair_modes: PairModes,
    nmodes: int,
    nmodals: int,
) -> jax.Array:
    """Apply H to flattened state vectors using prepacked pair operators.

    Parameters
    ----------
    obt
        Dynamic one-body coefficients with shape
        ``(nmodes, nmodals, nmodals)``.
    pair_ops
        Dynamic local two-mode matrices in canonical pair order.
    block_state
        Array with shape ``(nmodals**nmodes, nvec)``.

    Notes
    -----
    ``obt`` and ``pair_ops`` are ordinary dynamic JAX operands.  Their values
    therefore do not determine the JIT specialization; only their shapes and
    dtypes do.
    """
    if block_state.ndim != 2:
        raise ValueError("block_state must have shape (Hilbert dimension, nvec).")

    nvec = block_state.shape[1]
    state = block_state.reshape((nmodals,) * nmodes + (nvec,))
    result = _ham_on_state_block(
        obt, pair_ops, pair_modes, state, nmodes, nmodals
    )
    return result.reshape((nmodals**nmodes, nvec))


# ---------------------------------------------------------------------------
# LOBPCG support
# ---------------------------------------------------------------------------
# JAX's public lobpcg_standard interface accepts the linear operator as a
# static callable.  If that callable closes over Hamiltonian arrays, changing
# the Hamiltonian can change the compiled program and defeat persistent-cache
# reuse.  The implementation below keeps the same LOBPCG algorithmic structure
# while threading the Hamiltonian arrays explicitly as dynamic operands of the
# top-level jitted solver.


def _lobpcg_mm(a: jax.Array, b: jax.Array) -> jax.Array:
    """High-precision matrix multiplication used by the small LOBPCG algebra."""
    return jax.lax.dot(
        a,
        b,
        precision=(jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
    )


def _lobpcg_eigh_descending(matrix: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Return Hermitian eigenpairs ordered from largest to smallest value."""
    values, vectors = jnp.linalg.eigh(matrix)
    return values[::-1], vectors[:, ::-1]


def _lobpcg_svqb(basis: jax.Array) -> jax.Array:
    """Build a numerically truncated orthonormal basis using SVQB."""
    norms = jnp.linalg.norm(basis, ord=2, axis=0, keepdims=True)
    basis = basis / jnp.where(norms == 0, 1.0, norms)

    gram = _lobpcg_mm(basis.T, basis)
    eigenvalues, eigenvectors = _lobpcg_eigh_descending(gram)

    tau = jnp.finfo(basis.dtype).eps * eigenvalues[0]
    padded = jnp.maximum(eigenvalues, tau)
    inv_sqrt = jnp.where(tau > 0, padded, 1.0) ** (-0.5)

    scaled_vectors = eigenvectors * inv_sqrt[jnp.newaxis, :]
    ortho_basis = _lobpcg_mm(basis, scaled_vectors)

    keep = ((eigenvalues > tau) * (jnp.diag(gram) > 0.0))[jnp.newaxis, :]
    ortho_basis = ortho_basis * keep.astype(ortho_basis.dtype)

    norms = jnp.linalg.norm(ortho_basis, ord=2, axis=0, keepdims=True)
    keep = keep * (norms > 0.0)
    ortho_basis = ortho_basis / jnp.where(keep, norms, 1.0)
    return ortho_basis


def _lobpcg_orthonormalize(basis: jax.Array) -> jax.Array:
    """Apply two SVQB passes for robust orthonormalization."""
    basis = _lobpcg_svqb(basis)
    basis = _lobpcg_svqb(basis)
    return basis


def _lobpcg_extend_basis(X: jax.Array, n_extra: int) -> jax.Array:
    """Construct a deterministic orthonormal extension to the basis ``X``."""
    n, k = X.shape
    Xupper, Xlower = jnp.split(X, [k], axis=0)
    u, singular_values, vt = jnp.linalg.svd(Xupper)

    y = jnp.concatenate([Xupper + _lobpcg_mm(u, vt), Xlower], axis=0)
    other = jnp.concatenate(
        [
            jnp.eye(n_extra, dtype=X.dtype),
            jnp.zeros((n - k - n_extra, n_extra), dtype=X.dtype),
        ],
        axis=0,
    )

    scale = ((2 * (1 + singular_values)) ** (-0.5))[jnp.newaxis, :]
    w = _lobpcg_mm(y, vt.T * scale)
    correction = -2 * jnp.linalg.multi_dot(
        [w, w[k:, :].T, other],
        precision=jax.lax.Precision.HIGHEST,
    )
    return correction.at[k:].add(other)


def _lobpcg_project_out(basis: jax.Array, vectors: jax.Array) -> jax.Array:
    """Project vectors out of an orthonormal basis and re-orthonormalize."""
    for _ in range(2):
        vectors = vectors - _lobpcg_mm(
            basis, _lobpcg_mm(basis.T, vectors)
        )
        vectors = _lobpcg_orthonormalize(vectors)

    # End with basis subtraction so near-zero residual columns cannot regain a
    # numerically significant component parallel to the existing search basis.
    for _ in range(2):
        vectors = vectors - _lobpcg_mm(
            basis, _lobpcg_mm(basis.T, vectors)
        )

    norms = jnp.linalg.norm(vectors, ord=2, axis=0, keepdims=True)
    vectors = vectors * (norms >= 0.99).astype(vectors.dtype)
    return vectors


def _apply_shifted_operator(
    obt: jax.Array,
    pair_ops: jax.Array,
    c_shift: jax.Array,
    block_state: jax.Array,
    pair_modes: PairModes,
    nmodes: int,
    nmodals: int,
) -> jax.Array:
    """Apply ``c_shift * I - H`` with all Hamiltonian data kept dynamic."""
    H_state = _Ham_on_state_batched_prepared(
        obt,
        pair_ops,
        block_state,
        pair_modes=pair_modes,
        nmodes=nmodes,
        nmodals=nmodals,
    )
    return c_shift * block_state - H_state


def _lobpcg_rayleigh_ritz(
    obt: jax.Array,
    pair_ops: jax.Array,
    c_shift: jax.Array,
    search_basis: jax.Array,
    pair_modes: PairModes,
    nmodes: int,
    nmodals: int,
) -> Tuple[jax.Array, jax.Array]:
    """Solve the Rayleigh-Ritz problem inside an orthonormal search basis."""
    applied = _apply_shifted_operator(
        obt,
        pair_ops,
        c_shift,
        search_basis,
        pair_modes,
        nmodes,
        nmodals,
    )
    projected = _lobpcg_mm(search_basis.T, applied)
    return _lobpcg_eigh_descending(projected)


def _lobpcg_dynamic_operator(
    obt: jax.Array,
    pair_ops: jax.Array,
    c_shift: jax.Array,
    X: jax.Array,
    tol: jax.Array,
    pair_modes: PairModes,
    nmodes: int,
    nmodals: int,
    max_iter: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Run LOBPCG with Hamiltonian coefficients as dynamic operands.

    This routine follows the standard JAX LOBPCG structure for the largest
    eigenpairs of ``c_shift * I - H``.  The desired block size is determined by
    ``X.shape[1]``; ``get_ground_eigen_state_hpc`` uses one column.
    """
    n, k = X.shape

    X = _lobpcg_orthonormalize(X)
    P = _lobpcg_extend_basis(X, k)

    AX = _apply_shifted_operator(
        obt, pair_ops, c_shift, X, pair_modes, nmodes, nmodals
    )
    theta = jnp.sum(X * AX, axis=0, keepdims=True)
    R = AX - theta * X

    def cond(state):
        iteration, _X, _P, _R, converged, _theta = state
        return jnp.logical_and(iteration < max_iter, converged < k)

    def body(state):
        iteration, X_current, P_current, R_current, _converged, _theta = state

        R_current = _lobpcg_project_out(
            jnp.concatenate((X_current, P_current), axis=1),
            R_current,
        )
        XPR = jnp.concatenate((X_current, P_current, R_current), axis=1)

        theta_full, Q = _lobpcg_rayleigh_ritz(
            obt,
            pair_ops,
            c_shift,
            XPR,
            pair_modes,
            nmodes,
            nmodals,
        )

        # Extract the current Ritz vector approximation.
        B = Q[:, :k]
        normB = jnp.linalg.norm(B, ord=2, axis=0, keepdims=True)
        B = B / normB
        X_new = _lobpcg_mm(XPR, B)
        normX = jnp.linalg.norm(X_new, ord=2, axis=0, keepdims=True)
        X_new = X_new / normX

        # Construct the locally optimal search direction while preserving
        # orthogonality in the Rayleigh-Ritz coordinate basis.
        q, _ = jnp.linalg.qr(Q[:k, k:].T)
        diff_rayleigh_ortho = _lobpcg_mm(Q[:, k:], q)
        P_new = _lobpcg_mm(XPR, diff_rayleigh_ortho)
        normP = jnp.linalg.norm(P_new, ord=2, axis=0, keepdims=True)
        P_new = P_new / jnp.where(normP == 0, 1.0, normP)

        AX_new = _apply_shifted_operator(
            obt,
            pair_ops,
            c_shift,
            X_new,
            pair_modes,
            nmodes,
            nmodals,
        )
        theta_top = theta_full[:k]
        R_new = AX_new - theta_top[jnp.newaxis, :] * X_new
        residual_norms = jnp.linalg.norm(R_new, ord=2, axis=0)

        # Match the convergence scaling used by JAX's standard LOBPCG routine.
        relative_scale = (
            jnp.linalg.norm(AX_new, ord=2, axis=0) + theta_top
        )
        relative_scale = relative_scale * n * 10
        converged = jnp.sum(residual_norms < tol * relative_scale)

        return (
            iteration + 1,
            X_new,
            P_new,
            R_new,
            converged,
            theta_top[jnp.newaxis, :],
        )

    initial_state = (0, X, P, R, 0, theta)
    final_state = jax.lax.while_loop(cond, body, initial_state)
    iterations, X_final, _P, _R, _converged, theta_final = final_state
    return theta_final[0, :], X_final, iterations


@partial(
    jax.jit,
    static_argnames=("pair_modes", "nmodes", "nmodals", "max_iter"),
)
def _solve_shifted_ground_state_prepared(
    obt: jax.Array,
    pair_ops: jax.Array,
    c_shift: jax.Array,
    init_state: jax.Array,
    tol: jax.Array,
    pair_modes: PairModes,
    nmodes: int,
    nmodals: int,
    max_iter: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compile the shifted LOBPCG solve with dynamic Hamiltonian operands.

    ``obt``, ``pair_ops`` and ``c_shift`` are explicit dynamic arguments.  For
    fixed shapes and fixed static solver structure, changing their numerical
    values does not create a new JIT specialization.
    """
    return _lobpcg_dynamic_operator(
        obt,
        pair_ops,
        c_shift,
        init_state,
        tol,
        pair_modes,
        nmodes,
        nmodals,
        max_iter,
    )


# ---------------------------------------------------------------------------
# Public Hamiltonian helpers
# ---------------------------------------------------------------------------


def obt_on_state(
    obt: Union[np.ndarray, jax.Array],
    state: Union[np.ndarray, jax.Array],
) -> jax.Array:
    """Apply the one-body Hamiltonian directly to a tensor-product state."""
    obt = jnp.asarray(obt)
    state = jnp.asarray(state)
    nmodes = int(obt.shape[0])
    nmodals = int(obt.shape[-1])

    if state.shape != (nmodals,) * nmodes:
        raise ValueError(
            f"state must have shape {(nmodals,) * nmodes}; received {state.shape}."
        )

    block_state = state[..., None]
    result = _accumulate_obt_on_state(
        obt,
        block_state,
        jnp.zeros_like(block_state),
        nmodes,
        nmodals,
    )
    return result[..., 0]


def tbt_on_state(
    tbt_nonsym: Union[np.ndarray, jax.Array],
    state: Union[np.ndarray, jax.Array],
) -> jax.Array:
    """Apply the nonsymmetrized two-body Hamiltonian directly to a state.

    Only lower-triangular mode blocks ``i > j`` are used.  No symmetrization,
    global tensor factorization, or eigenvalue cutoff is performed.
    """
    tbt_np = np.asarray(tbt_nonsym)
    state = jnp.asarray(state)

    nmodes = int(tbt_np.shape[0])
    nmodals = int(tbt_np.shape[1])
    expected_state_shape = (nmodals,) * nmodes
    if state.shape != expected_state_shape:
        raise ValueError(
            f"state must have shape {expected_state_shape}; received {state.shape}."
        )

    pair_ops_np, pair_modes = prepare_tbt_pair_operators(tbt_np)
    pair_ops = jnp.asarray(pair_ops_np)
    block_state = state[..., None]
    result = _accumulate_tbt_on_state(
        pair_ops,
        pair_modes,
        block_state,
        jnp.zeros_like(block_state),
        nmodes,
        nmodals,
    )
    return result[..., 0]


def Ham_on_state(
    obt: Union[np.ndarray, jax.Array],
    tbt_nonsym: Union[np.ndarray, jax.Array],
    flt_state: Union[np.ndarray, jax.Array],
) -> jax.Array:
    """Apply the full one- plus two-mode Hamiltonian to a flattened state."""
    obt_np = np.asarray(obt)
    tbt_np = np.asarray(tbt_nonsym)
    nmodes, nmodals = _validate_inputs(obt_np, tbt_np)

    pair_ops_np, pair_modes = prepare_tbt_pair_operators(tbt_np)

    obt_jax = jnp.asarray(obt_np, dtype=jnp.float64)
    pair_ops_jax = jnp.asarray(pair_ops_np, dtype=jnp.float64)
    flt_state_jax = jnp.asarray(flt_state, dtype=jnp.float64)

    dim = nmodals**nmodes
    if flt_state_jax.shape != (dim,):
        raise ValueError(
            f"flt_state must have shape ({dim},); received {flt_state_jax.shape}."
        )

    result = _Ham_on_state_batched_prepared(
        obt_jax,
        pair_ops_jax,
        flt_state_jax[:, None],
        pair_modes=pair_modes,
        nmodes=nmodes,
        nmodals=nmodals,
    )
    return result[:, 0]


def get_ground_eigen_state_hpc(
    obt: np.ndarray,
    tbt_nonsym: np.ndarray,
    tbt_cutoff: Optional[float] = 1e-3,
    max_iter: int = 50,
    tol: Optional[float] = None,
    lr: float = 1e-3,
):
    """Obtain the ground eigenstate with direct matrix-free LOBPCG.

    The public calling convention mirrors ``eigen_utils.get_ground_eigen_state``
    except that the second argument is the lower-triangular nonsymmetrized
    two-body tensor.

    Parameters
    ----------
    obt : np.ndarray
        One-body tensor with shape ``(nmodes, nmodals, nmodals)``.
    tbt_nonsym : np.ndarray
        Nonsymmetrized two-body tensor with shape
        ``(nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)``.  Only blocks
        with ``i > j`` may be nonzero.
    tbt_cutoff : float or None, optional
        Accepted for call-site compatibility with the original function.  It
        is ignored because the direct two-mode Hamiltonian is applied exactly.
    max_iter : int, optional
        Maximum number of LOBPCG iterations.  This value is static for JIT
        compilation, so changing it creates a separate compiled executable.
    tol : float or None, optional
        LOBPCG convergence tolerance.  ``None`` uses machine epsilon for
        float64, matching JAX's standard LOBPCG default.
    lr : float, optional
        Accepted for call-site compatibility.  LOBPCG does not use a learning
        rate.

    Returns
    -------
    val : float
        Approximate ground-state energy.
    psi : jax.Array
        Approximate normalized ground-state wavefunction with shape
        ``(nmodals**nmodes,)``.

    Notes
    -----
    For persistent compilation-cache reuse across Hamiltonian configurations,
    keep ``nmodes``, ``nmodals``, ``max_iter``, dtype, and software/hardware
    environment unchanged.  The numerical values of ``obt``, ``tbt_nonsym``
    and ``c_shift`` are dynamic inputs and do not define the JIT specialization.
    """
    del tbt_cutoff, lr

    obt_np = np.asarray(obt)
    tbt_np = np.asarray(tbt_nonsym)
    nmodes, nmodals = _validate_inputs(obt_np, tbt_np)

    if not np.isrealobj(obt_np) or not np.isrealobj(tbt_np):
        raise TypeError(
            "get_ground_eigen_state_hpc currently expects real-valued obt and "
            "tbt_nonsym tensors."
        )

    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    # Pack every lower-triangular mode pair in a fixed canonical order.  For
    # nmodes=12 this always produces 66 matrices, independent of geometry.
    pair_ops_np, pair_modes = prepare_tbt_pair_operators(tbt_np)

    obt_jax = jnp.asarray(obt_np, dtype=jnp.float64)
    pair_ops_jax = jnp.asarray(pair_ops_np, dtype=jnp.float64)

    dim = nmodals**nmodes

    # Reference product state |0, 0, ..., 0> = [1, 0, 0, ...]^T.
    ref_state = jnp.zeros((dim,), dtype=jnp.float64)
    ref_state = ref_state.at[0].set(1.0)
    init_state = ref_state[:, None]

    # Evaluate <ref|H|ref> directly from local coefficients.  This is exactly
    # equivalent to a full Hamiltonian application on the reference product
    # state but avoids an unnecessary Hilbert-space-sized operation.
    E_ref_np = float(np.sum(obt_np[:, 0, 0], dtype=np.float64))
    for mode_i, mode_j in pair_modes:
        E_ref_np += float(tbt_np[mode_i, 0, 0, mode_j, 0, 0])
    c_shift = jnp.asarray(E_ref_np + 1.0, dtype=jnp.float64)

    if tol is None:
        tol_value = np.finfo(np.float64).eps
    else:
        tol_value = float(tol)
        if tol_value <= 0.0 or not np.isfinite(tol_value):
            raise ValueError("tol must be a positive finite number or None.")
    tol_jax = jnp.asarray(tol_value, dtype=jnp.float64)

    # The Hamiltonian coefficients and c_shift are explicit dynamic operands of
    # this compiled solve.  Changing only their numerical values therefore does
    # not require a new JIT specialization.
    shifted_val, vec, history = _solve_shifted_ground_state_prepared(
        obt_jax,
        pair_ops_jax,
        c_shift,
        init_state,
        tol_jax,
        pair_modes=pair_modes,
        nmodes=nmodes,
        nmodals=nmodals,
        max_iter=max_iter,
    )

    val_jax = c_shift - shifted_val[0]
    psi = vec[:, 0]

    # Verify the returned eigenpair against the unshifted Hamiltonian.
    Hpsi = _Ham_on_state_batched_prepared(
        obt_jax,
        pair_ops_jax,
        psi[:, None],
        pair_modes=pair_modes,
        nmodes=nmodes,
        nmodals=nmodals,
    )[:, 0]
    residual_jax = jnp.linalg.norm(Hpsi - val_jax * psi) / jnp.linalg.norm(psi)

    # Materializing scalar diagnostics also synchronizes the relevant GPU work
    # before the function returns.  The wavefunction remains a JAX array.
    val = float(val_jax)
    residual = float(residual_jax)
    iterations = int(history)

    print("Number of tbt mode pairs:", len(pair_modes), flush=True)
    print("Number of lobpcg iterations:", iterations, flush=True)
    print("Approximate ground energy:", val, "cm-1", flush=True)
    print("Eigenpair residual:", residual, "cm-1", flush=True)

    return val, psi


__all__ = [
    "Ham_on_state",
    "get_ground_eigen_state_hpc",
    "obt_on_state",
    "prepare_tbt_pair_operators",
    "tbt_on_state",
]
