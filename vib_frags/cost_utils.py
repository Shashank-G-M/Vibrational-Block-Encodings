# A module to estimate quantum resources for different fragmentation schemes
import numpy as np
from . import tensor_utils as tu
from copy import copy
import pennylane.labs.estimator_beta as qre_beta






def get_pruned_tensors_w_err(H1, H2_nonsym=None, H3_nonsym=None, MC=2, cutoff=0):
    
    # Process H1 using a memory-efficient mask
    mask1 = np.abs(H1) <= cutoff
    H1_approx = H1.copy()
    H1_approx[mask1] = 0
    err1 = np.sqrt(np.sum(H1[mask1]**2))
    
    # Process H2 using the same mask logic if it is provided
    err2 = 0.0
    H2_nonsym_approx = None
    if H2_nonsym is not None:
        mask2 = np.abs(H2_nonsym) <= cutoff
        H2_nonsym_approx = H2_nonsym.copy()
        H2_nonsym_approx[mask2] = 0
        err2 = np.sqrt(np.sum(H2_nonsym[mask2]**2))
        
    sparse_ten_error = err1 + err2
    
    H3_nonsym_approx = None
    if MC == 3 and H3_nonsym is not None:
        # Pre-allocate the output array for H3 to avoid massive copies
        H3_nonsym_approx = np.empty_like(H3_nonsym)
        err3_sq = 0.0
        
        # Process H3 slice-by-slice along the first dimension (chunking)
        for i in range(H3_nonsym.shape[0]):
            slice_h3 = H3_nonsym[i]
            mask3 = np.abs(slice_h3) <= cutoff
            
            # Accumulate the squared error strictly for pruned elements
            err3_sq += np.sum(slice_h3[mask3]**2)
            
            # Create the approximated slice
            approx_slice = slice_h3.copy()
            approx_slice[mask3] = 0
            
            # Place the finalized slice into the pre-allocated array
            H3_nonsym_approx[i] = approx_slice
            
        sparse_ten_error += np.sqrt(err3_sq)
        
    return H1_approx, H2_nonsym_approx, H3_nonsym_approx, sparse_ten_error




#Obtaining masked tensors for counting S in Pauli LCU
def mask_tensor_by_index_order(arr, axis_pairs):
    """
    arr : numpy array
    axis_pairs : list of tuples [(ax_x, ax_y), ...] with ax_x < ax_y where ax_x and ax_y 
                 are axes indices in arr that should satisfy index_x >= index_y
    returns: arr with entries zeroed where ANY pair condition is False
    """
    final_mask = None
    for ax_x, ax_y in axis_pairs:
        dim_x = arr.shape[ax_x]
        dim_y = arr.shape[ax_y]
        idx_x = np.arange(dim_x)[:, None]    # shape (dim_x, 1)
        idx_y = np.arange(dim_y)[None, :]    # shape (1, dim_y)
        pair_mask = (idx_x >= idx_y)        # shape (dim_x, dim_y)

        # build shape of ones and insert pair_mask at the two axes
        shape = [1] * arr.ndim
        shape[ax_x] = dim_x
        shape[ax_y] = dim_y
        pair_mask_reshaped = pair_mask.reshape(shape)

        if final_mask is None:
            final_mask = pair_mask_reshaped
        else:
            final_mask = final_mask & pair_mask_reshaped

    # if no pairs given, return arr unchanged
    if final_mask is None:
        return arr
    # broadcast final_mask to arr shape and multiply
    return arr * final_mask



def get_Pauli_S(H1, H2_nonsym = None, H3_nonsym = None, cutoff = 0):
  """
  Finds the number of terms in the unary iteration in Pauli PREP circuit.
  It is important to note that the two and three body tensors need to be provided in the unsymmetrized form with shapes,
  H2_nonsym : np.array((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
  H3_nonsym : np.array((nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals))
  """
  H1_masked = mask_tensor_by_index_order(H1, [(1,2)])               # p>=q
  S = np.sum(np.abs(H1_masked) > cutoff)
  if H2_nonsym is not None:
    H2_nonsym_masked = mask_tensor_by_index_order(H2_nonsym, [(1,2),(4,5)])         # p>=q and r>=s
    S += np.sum(np.abs(H2_nonsym_masked) > cutoff)
  if H3_nonsym is not None:
    H3_nonsym_masked = mask_tensor_by_index_order(H3_nonsym, [(1,2),(4,5),(7,8)])   # p>=q, r>=s, t>=u
    S += int(np.sum(np.abs(H3_nonsym_masked) > cutoff))
  return int(S)




def find_dirty_QROAM_k(S, m, dQ):
    """
    Find the optimal value of k in SelectSwap (QROAM) when the number
    of available dirty ancillas, dQ, is given.

    Returns k = 1 when standard QROM is cheaper than all feasible
    dirty-ancilla QROAM choices.
    """

    # Use standard QROM as the baseline.
    # k = 1 is only used here as a flag for standard QROM; the dirty-QROAM
    # theorem itself applies to k > 1.
    best_k = 1
    min_objective = S - 1

    # k must be a power of two.
    k = 2

    # Search over all physically allowed power-of-two values of k.
    # QROAM with parameter k requires (k - 1)*m dirty workspace qubits.
    while (k < S) and ((k - 1) * m <= dQ):

        # Toffoli cost from the dirty-ancilla QROAM theorem.
        objective_val = 2 * np.ceil(S / k) + 4 * m * (k - 1)

        # Only use QROAM if it improves upon the best cost found so far.
        # If no QROAM value beats standard QROM, best_k remains equal to 1.
        if objective_val < min_objective:
            min_objective = objective_val
            best_k = k

        # Move to the next allowed power-of-two value of k.
        k *= 2

    return int(best_k)




def find_clean_QROAM_k(S, m, cQ):
    """
    Find the optimal value of k in SelectSwap (QROAM) when the number of available clean ancillas, cQ, is given. 
    Returns k = 1 when standard QROM is cheaper than all feasible clean-ancilla QROAM configurations.
    """
    best_k = 1
    # Use standard QROM as the baseline. If no clean-QROAM configuration
    # improves upon this cost, best_k remains equal to 1.
    min_objective = S - 1

    k = 2

    # Search over allowed power-of-two values of k.
    while k < S:
        # Total clean-ancilla requirement for this value of k.
        constraint_val = (k - 1) * m + np.ceil(np.log2(S / k))

        # Stop if this value of k exceeds the available clean-ancilla budget. 
        # The required number of ancillas increases for larger k.
        if constraint_val > cQ:
            break

        # Toffoli cost of clean-ancilla QROAM for this value of k.
        objective_val = np.ceil(S / k) + m * (k - 1)

        # Once the QROAM cost begins increasing, larger powers of two will not improve the optimum.
        if objective_val > min_objective:
            break

        # Update the optimum only if QROAM improves upon the current best cost, including the standard-QROM baseline.
        if objective_val < min_objective:
            min_objective = objective_val
            best_k = k

        # Move to the next allowed power-of-two value of k.
        k *= 2

    return int(best_k)





def get_SelectCopy_QROAM_cost(S, m, dirty):
    """
    Return the Toffoli, clean-ancilla, and dirty-ancilla costs
    of PennyLane's optimized Select-Copy QROM.

    S     : number of bitstrings
    m     : number of bits per bitstring
    dirty : maximum number of available dirty ancillas
    """

    qrom = qre_beta.SelectCopyQROM(
        num_bitstrings=int(S),
        size_bitstring=int(m),
        available_dirty_aux=int(dirty),
    )

    resources = qre_beta.estimate(qrom)

    # Optimized Select-Copy parameters chosen by PennyLane.
    lam = qrom.batch_size
    mu = qrom.bits_per_iter

    # Actual dirty workspace used by the optimized circuit.
    dirty_cost = mu * (lam - 1)

    # Total dynamically allocated workspace excludes the algorithmic
    # address and output registers.
    allocated_cost = resources.zeroed_wires + resources.any_state_wires

    # Remaining allocated workspace is clean auxiliary workspace.
    clean_cost = allocated_cost - dirty_cost

    # Exact Toffoli count.
    Toffoli_cost = resources.gate_counts["Toffoli"]

    return int(Toffoli_cost), int(clean_cost), int(dirty_cost)





def get_QROAM_cost(S, m, dirty=False, clean=False, nmodes=None, nmodals=None, select_copy = False):
    """
    Calculates the Toffoli and ancilla costs for QROAM data loading.
    Evaluates the resource requirements given the size of the index register (S)
    and the bit precision (m). The function optimizes multiplexing
    parameters based on the availability of clean or dirty ancilla qubits.
    When dirty ancillas are specified, the dirty-QROAM regime takes
    precedence and the clean argument is ignored for optimization.

    Note:
        The returned ancilla costs exclude the index register (cost: log2(S))
        and the output container (cost: m). The baseline clean ancilla requirement
        is bounded by log2(S); deficits in the provided budget will default
        to this minimum.

    Args:
        S (int): Size of the index register.
        m (int): Bit precision.
        dirty (int | bool): Available dirty ancillas, or flag to calculate via modes/modals.
        clean (int | bool): Available clean ancillas, or flag assuming optimal unconstrained availability.
        nmodes (int, optional): Number of modes. Required if dirty is True.
        nmodals (int, optional): Number of modals. Required if dirty is True.
        select_copy (bool): If True and dirty is not False, utilizes PennyLane's Select_Copy QROAM.

    Returns:
        tuple: A tuple containing (Toffoli_cost, clean_cost, dirty_cost) as integers.
    """

    # Process the dirty ancilla case if requested
    if dirty:
        if isinstance(dirty, bool):
            if nmodes is None or nmodals is None:
                raise ValueError("Explicit values for nmodes and nmodals are required when 'dirty' is passed as a boolean.")
            dQ = nmodes * nmodals
        else:
            dQ = dirty
        
        #Perform PennyLane's Select-Copy if select_copy is True
        if select_copy is True:
            return get_SelectCopy_QROAM_cost(S, m, dirty)

        # Determine the optimal QROAM parameter subject to the available ancilla budget. 
        # A returned value of k = 1 indicates that standard QROM is cheaper than every feasible dirty-QROAM configuration.
        k = find_dirty_QROAM_k(S, m, dQ)

        # Fall back to standard QROM when dirty QROAM provides no advantage.
        if k == 1:
            Toffoli_cost = S - 1
            clean_cost = np.ceil(np.log2(S))
            return int(Toffoli_cost), int(clean_cost), 0

        # Dirty-ancilla QROAM resource costs for the selected value of k.
        Toffoli_cost = 2 * np.ceil(S / k) + 4 * m * (k - 1)
        clean_cost = np.ceil(np.log2(S / k))
        dirty_cost = (k - 1) * m

        return int(Toffoli_cost), int(clean_cost), int(dirty_cost)

    # Process the unconstrained optimal clean ancilla case
    if clean is True:
        print("Notice: Assuming optimal unconstrained clean ancilla availability for QROAM.")

        # Find the optimal power-of-two value of k without imposing
        # any restriction on the number of available clean ancillas.
        k = find_clean_QROAM_k(S, m, np.inf)

        # k = 1 indicates that standard QROM is cheaper than every
        # clean-ancilla QROAM configuration.
        if k == 1:
            Toffoli_cost = S - 1
            clean_cost = np.ceil(np.log2(S))
            return int(Toffoli_cost), int(clean_cost), 0

        # Clean-ancilla QROAM resource costs for the selected value of k.
        Toffoli_cost = np.ceil(S / k) + m * (k - 1)
        clean_cost = np.ceil(np.log2(S / k)) + (k - 1) * m

        return int(Toffoli_cost), int(clean_cost), 0

    # Process the constrained clean ancilla case based on a strict budget
    elif type(clean) is int:
        k = find_clean_QROAM_k(S, m, clean)

        # k = 1 indicates that standard QROM is cheaper.
        if k == 1:
            Toffoli_cost = S - 1
            clean_cost = np.ceil(np.log2(S))
            return int(Toffoli_cost), int(clean_cost), 0

        Toffoli_cost = np.ceil(S / k) + m * (k - 1)
        clean_cost = np.ceil(np.log2(S / k)) + (k - 1) * m
        return int(Toffoli_cost), int(clean_cost), 0

    # Fallback to standard QROM scaling if no valid optimization parameters are provided
    else:
        Toffoli_cost = S - 1
        clean_cost = np.ceil(np.log2(S))
        return int(Toffoli_cost), int(clean_cost), 0






def find_dirty_QROAM_inv_k(S, dQ):
    """
    Find the optimal value of k for inverse QROAM using dirty ancillas. Returns k = 1 when standard QROM uncomputation 
    is cheaper than all feasible dirty-ancilla inverse-QROAM configurations.
    """

    # Use standard QROM uncomputation as the baseline.
    # k = 1 is used only as a flag indicating standard QROM.
    best_k = 1
    min_objective = S - 1

    k = 2

    # Search over all physically allowed power-of-two values of k.
    # Inverse QROAM with parameter k requires (k - 1) dirty ancillas.
    while (k < S) and ((k - 1) <= dQ):

        # Toffoli cost from the dirty-ancilla inverse-QROAM theorem.
        objective_val = 2 * np.ceil(S / k) + 4 * k

        # Update the optimum only if inverse QROAM improves upon the current best cost, including standard QROM.
        if objective_val < min_objective:
            min_objective = objective_val
            best_k = k

        # Move to the next allowed power-of-two value of k.
        k *= 2

    return int(best_k)


def find_clean_QROAM_inv_k(S, cQ):
    """
    Find the optimal value of k for inverse QROAM using clean ancillas. Returns k = 1 when standard QROM uncomputation 
    is cheaper than all feasible clean-ancilla inverse-QROAM configurations.
    """

    # Use standard QROM uncomputation as the baseline.
    # k = 1 is used only as a flag indicating standard QROM.
    best_k = 1
    min_objective = S - 1

    k = 2

    # Search over allowed power-of-two values of k.
    while k < S:

        # Total clean-ancilla requirement for this value of k.
        constraint_val = np.ceil(np.log2(S / k)) + k

        # Stop if this value of k exceeds the available clean-ancilla budget.
        if constraint_val > cQ:
            break

        # Toffoli cost from the clean-ancilla inverse-QROAM theorem.
        objective_val = np.ceil(S / k) + k

        # Update the optimum only if inverse QROAM improves upon
        # the current best cost, including standard QROM.
        if objective_val < min_objective:
            min_objective = objective_val
            best_k = k

        # Move to the next allowed power-of-two value of k.
        k *= 2

    return int(best_k)


def get_QROAM_inv_cost(S, dirty=False, clean=False, nmodes=None, nmodals=None):
    """
    Calculates the Toffoli and ancilla costs for inverse QROAM based on measurement-based uncomputation.
    Evaluates the resource requirements given the size of the index register (S).
    The function optimizes parameters based on the availability of clean or dirty ancilla qubits.
    When dirty ancillas are specified, the dirty-QROAM regime takes
    precedence and the clean argument is ignored for optimization.

    Args:
        S (int): Size of the index register.
        dirty (int | bool): Available dirty ancillas, or flag to calculate via modes/modals.
        clean (int | bool): Available clean ancillas, or flag assuming optimal unconstrained availability.
        nmodes (int, optional): Number of modes. Required if dirty is True.
        nmodals (int, optional): Number of modals. Required if dirty is True.

    Returns:
        tuple: A tuple containing (Toffoli_cost, clean_cost, dirty_cost) as integers.
    """
    
    # Process the dirty ancilla case if requested
    if dirty:
        if isinstance(dirty, bool):
            if nmodes is None or nmodals is None:
                raise ValueError("Explicit values for nmodes and nmodals are required when 'dirty' is passed as a boolean.")
            dQ = nmodes * nmodals
        else:
            dQ = dirty
        
        # Determine the optimal inverse-QROAM parameter subject to the available dirty-ancilla budget.
        k = find_dirty_QROAM_inv_k(S, dQ)

        # k = 1 indicates that standard QROM is cheaper than every
        # feasible dirty-ancilla inverse-QROAM configuration.
        if k == 1:
            Toffoli_cost = S - 1
            clean_cost = np.ceil(np.log2(S))
            return int(Toffoli_cost), int(clean_cost), 0

        # Dirty-ancilla inverse-QROAM resource costs for the selected value of k.
        Toffoli_cost = 2 * np.ceil(S / k) + 4 * k
        clean_cost = np.ceil(np.log2(S / k)) + 1
        dirty_cost = k - 1

        return int(Toffoli_cost), int(clean_cost), int(dirty_cost)

    # Process the unconstrained optimal clean ancilla case
    if clean is True:
        print("Notice: Assuming optimal unconstrained clean ancilla availability for inverse QROAM.")
        # Find the optimal power-of-two value of k without imposing
        # any restriction on the number of available clean ancillas.
        k = find_clean_QROAM_inv_k(S, np.inf)

        # k = 1 indicates that standard QROM is cheaper than every
        # clean-ancilla inverse-QROAM configuration.
        if k == 1:
            Toffoli_cost = S - 1
            clean_cost = np.ceil(np.log2(S))
            return int(Toffoli_cost), int(clean_cost), 0

        # Clean-ancilla inverse-QROAM resource costs for the selected value of k.
        Toffoli_cost = np.ceil(S / k) + k
        clean_cost = np.ceil(np.log2(S / k)) + k

        return int(Toffoli_cost), int(clean_cost), 0

    # Process the constrained clean ancilla case based on a strict budget
    elif type(clean) is int:
        # Determine the optimal inverse-QROAM parameter subject to the available clean-ancilla budget.
        k = find_clean_QROAM_inv_k(S, clean)

        # k = 1 indicates that standard QROM is cheaper than every
        # feasible clean-ancilla inverse-QROAM configuration.
        if k == 1:
            Toffoli_cost = S - 1
            clean_cost = np.ceil(np.log2(S))
            return int(Toffoli_cost), int(clean_cost), 0

        # Clean-ancilla inverse-QROAM resource costs for the selected value of k.
        Toffoli_cost = np.ceil(S / k) + k
        clean_cost = np.ceil(np.log2(S / k)) + k

        return int(Toffoli_cost), int(clean_cost), 0

    # Fallback to standard QROM scaling if no valid optimization parameters are provided
    else:
        Toffoli_cost = S - 1
        clean_cost = np.ceil(np.log2(S))
        return int(Toffoli_cost), int(clean_cost), 0





# Toffoli and qubit cost of Pauli PREP circuit and its inverse
def get_Pauli_PREP_UNPREP_cost(nmodes, nmodals, S, aleph, MC, br = 20, dirty = False, clean = False, verbose = False, select_copy = False):
  """
  S can be obtained using get_Pauli_S function. aleph stands for the number of bits precicion used in expressing LCU coefficients.
  br is the number of bits used to specify rotation angle in generating uniform superposition state.
  """
  #Some constants:
  nM = np.ceil(np.log2(nmodes))
  nN = np.ceil(np.log2(nmodals))
  nMC = np.ceil(np.log2(MC))
  eta = (S & -S).bit_length() - 1                         #eta is the largest integer such that 2^eta is a factor of S. Used in uniform superposition
  m = 2*MC*(nM + 2*nN + 1) + 2*nMC + 2 + aleph                 #size of the register loaded by QROM

  uniform_state_cost = 3*np.ceil(np.log2(S)) - 3*eta + 2*br - 9

  QROAM_cost, QROAM_clean, QROAM_dirty = get_QROAM_cost(S, m, dirty, clean, nmodes, nmodals, select_copy)
  QROAM_inv_cost, QROAM_inv_clean, QROAM_inv_dirty = get_QROAM_inv_cost(S, dirty, QROAM_clean, nmodes, nmodals)
  
  if verbose:
    print (f"\nQROAM ancilla cost: {QROAM_clean}")
    print (f"QROAM inverse ancilla cost: {QROAM_inv_clean}")

  #coherent alias sampling (cas) cost
  cas_cost = aleph

  #index <-> alt swap cost controlled on output of cas
  swp_cost = MC*(nM + 2*nN + 1) + nMC + 1

  # PREP cost
  C_P = uniform_state_cost + QROAM_cost + cas_cost + swp_cost
  # UNPREP cost
  C_Pdag = uniform_state_cost + QROAM_inv_cost + cas_cost + swp_cost
  # Qubit cost
  C_P_qubits = np.ceil(np.log2(S)) + 2 + br + m + aleph + 1 + MC + QROAM_clean

  if QROAM_inv_clean > QROAM_clean:
    C_P_qubits += int(QROAM_inv_clean - QROAM_clean)


  return C_P, C_Pdag, C_P_qubits



# Toffoli and qubit cost of Pauli 
def get_Pauli_SELECT_cost(nmodes, nmodals, MC):
  #Cost of primitive
  C_SubSel = 2*nmodes*nmodals + 2*nmodes + 3*nmodals - 6
  #Total SELECT cost
  C_S = MC*C_SubSel + MC - 1           
  C_S_qubits = nmodes*nmodals
  return C_S, C_S_qubits




def get_Pauli_QPE_cost(one_norm, eps, nmodes, nmodals, S, MC, br = 20, aleph = 20, dirty = False, clean = False, verbose = True, select_copy = False):  
  """
  Function to obtain the cost of performing QPE to a target accuracy of eps using Pauli LCU based block encoding.
  """
  if dirty is True:
    dirty = nmodes*nmodals + aleph + br                         #The system qubits, aleph qubits from alias sampling, and the ancillas from uniform superposition are idle dirty qubits

  # PREP and PREP inverse cost
  C_P, C_Pdag, C_P_qubits = get_Pauli_PREP_UNPREP_cost(nmodes, nmodals, S, aleph, MC, br, dirty, clean, verbose, select_copy)
  # SELECT cost
  C_S, C_S_qubits = get_Pauli_SELECT_cost(nmodes, nmodals, MC)

  #Block encoding cost
  C_BE = C_P + C_S + C_Pdag
  # C_BE = C_P + C_S + C_P

  QPE_nsteps = np.ceil(np.pi*one_norm/(2*eps))           # Multiplier to Walker operator cost to get QPE cost

  #Total energy estimation cost
  C_Pauli = QPE_nsteps*(C_BE + np.ceil(np.log2(S)) + aleph + 4)
  N_Pauli = 2*np.ceil(np.log2(QPE_nsteps + 1)) - 1 + C_P_qubits + C_S_qubits

  if verbose:
    print (f"\nCost of PREP: {C_P: 0.2e}")
    print (f"Cost of SEL: {C_S: 0.2e}")
    print (f"Cost of BE: {C_BE: 0.2e}")
    print (f"LCU 1-norm: {one_norm: 0.2e} cm-1")
    print (f"# of walks: {QPE_nsteps: 0.2e}")
    print (f"Toffoli cost of Sparse Pauli LCU: {C_Pauli: 0.5e}")
    print (f"Qubit cost of Sparse Pauli LCU: {N_Pauli}")
  
  return C_Pauli, N_Pauli














#__________________________________________________________________________________________________________________
#THC cost utils
#__________________________________________________________________________________________________________________
def get_THC_BE_cost(nmodes, nmodals, S, R, nthc, MC, br = 20, aleph = 20, beth = 20, dirty = False, clean = False, verbose = False, select_copy = False):
  """
  Function to obtain the Toffoli and qubit costs of block encoding a THC LCU
  """
  #PREP and UNPREP cost
  #Some constants:
  nM = np.ceil(np.log2(nmodes))
  nR = np.ceil(np.log2(R))
  nMC = np.ceil(np.log2(MC))
  #eta is the largest integer such that 2^eta is a factor of S. Used in uniform superposition
  eta = (S & -S).bit_length() - 1                         
  #Data loader output size
  m = 2*MC*(nM + nR) + 2*nMC + 2 + aleph

  #Cost of uniform superposition of S states
  uniform_state_cost = 3*np.ceil(np.log2(S)) - 3*eta + 2*br - 9

  PREP_QROAM_cost, PREP_QROAM_clean, _ = get_QROAM_cost(S, m, dirty, clean, nmodes, nmodals, select_copy)
  UNPREP_QROAM_inv_cost, UNPREP_QROAM_inv_clean, _ = get_QROAM_inv_cost(S, dirty, PREP_QROAM_clean, nmodes, nmodals)
  
  if verbose:
    print (f"\nQROAM ancilla cost: {PREP_QROAM_clean}")
    print (f"QROAM inverse ancilla cost: {UNPREP_QROAM_inv_clean}")
  
  #coherent alias sampling (cas) cost
  cas_cost = aleph
  #index <-> alt swap cost controlled on output of cas
  swp_cost = MC*(nM + nR) + nMC + 1

  #Complete PREP cost
  C_P = uniform_state_cost + PREP_QROAM_cost + cas_cost + swp_cost
  #Complete UNPREP cost
  C_Pdag = uniform_state_cost + UNPREP_QROAM_inv_cost + cas_cost + swp_cost

  #SELECT cost
  C_S = MC - 1
  m_tilde = (nmodals - 1)*beth
  max_SEL_QROAM_clean = 0
  for l in range (MC):
     Sl = (MC - l) * sum(nthc[l:]) + (0 if l > 0 else nmodes * nmodals - sum(nthc))
     clean_new = max(PREP_QROAM_clean - m_tilde, beth - 2) if clean > 0 else False
     SEL_QROAM_cost, SEL_QROAM_clean, _ = get_QROAM_cost(Sl, m_tilde, False, clean_new, nmodes, nmodals, False)
     SEL_QROAM_inv_cost, SEL_QROAM_inv_clean, _ = get_QROAM_inv_cost(Sl, False, clean_new, nmodes, nmodals)
     C_SubSel_l = 2*(nmodes-l-1)*nmodals + SEL_QROAM_cost + SEL_QROAM_inv_cost + 4*(nmodals - 1)*(beth - 2)
     C_S += C_SubSel_l
     max_SEL_QROAM_clean = max(max_SEL_QROAM_clean, SEL_QROAM_clean, SEL_QROAM_inv_clean)

  N_BE = np.ceil(np.log2(S)) + max(br, beth) + 2 + m + aleph + 1 + nmodes*nmodals + max(max(PREP_QROAM_clean, UNPREP_QROAM_inv_clean), max(max_SEL_QROAM_clean, beth - 2) + m_tilde)

  return C_P, C_S, C_Pdag, N_BE




#Cost of QPE of THC Hamiltonian
def get_THC_QPE_cost(lmbda, eps, nmodes, nmodals, nthc, MC, br = 20, aleph = 20, beth = 20, PREP_vol = None, dirty = False, clean = False, verbose = True, select_copy = False):
  """
  Get THC QPE cost
  """
  if MC not in (2, 3):
      raise ValueError("MC must be either 2 or 3.")

  #Number of terms in LCU
  if type(nthc) is int:
    S = int(nmodes*nmodals + (nmodes*(nmodes - 1)/2)*nthc**2)        # Number of |ip> + number of |iu>|jv>
    if MC == 3:
      S += int((nmodes*(nmodes - 1)*(nmodes-2)/6)*nthc**3)           # Number of |iu>|jv>|kw>
    R = nthc
    nthc = [nthc,]*nmodes
  else:
    if len(nthc) != nmodes:
        raise ValueError("When nthc is array like, it should have exactly nmodes elements")
    R_sum = 0
    for i in range (nmodes):
      for j in range (i):
        R_sum += nthc[i] * nthc[j]
    if MC == 3:
      for i in range (nmodes):
        for j in range (i):
          for k in range (j):
            R_sum += nthc[i] * nthc[j] * nthc[k]
    S = int(nmodes*nmodals + R_sum)
    R = max(nthc)

  if PREP_vol is not None:
    S = PREP_vol

  if dirty is True:
    dirty = nmodes*nmodals + aleph + nmodals*beth - 2                 #The system qubits, aleph qubits from alias sampling, and the ancillas for orbital rotation are idle dirty qubits


  #Number of steps in QPE i.e. number of access to walker operator
  QPE_nsteps = np.ceil(np.pi*lmbda/(2*eps))
  #Cost of single walker = Cost of block encoding + cost of controlled reflection
  C_P, C_S, C_Pdag, N_BE = get_THC_BE_cost(nmodes, nmodals, S, R, nthc, MC, br, aleph, beth, dirty, clean, verbose, select_copy)
  C_BE = C_P + C_S + C_Pdag
  C_W = C_BE + np.ceil(np.log2(S)) + aleph + 1

  #Total Toffoli cost of THC LCU
  C_THC = QPE_nsteps*C_W

  #Totoal qubit cost of THC LCU
  N_THC = int(2*np.ceil(np.log2(QPE_nsteps + 1)) - 1 + N_BE)

  if verbose:
    print (f"\nCost of PREP: {C_P: 0.2e}")
    print (f"Cost of SEL: {C_S: 0.2e}")
    print (f"Cost of BE: {C_BE: 0.2e}")
    print (f"LCU 1-norm: {lmbda: 0.2e} cm-1")
    print (f"# of walks: {QPE_nsteps: 0.2e}")
    print (f"Toffoli cost of THC LCU: {C_THC: 0.5e}")
    print (f"Qubit cost of THC LCU: {N_THC}")
  
  return C_THC, N_THC

