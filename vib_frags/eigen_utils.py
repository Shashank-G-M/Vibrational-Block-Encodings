import os
import numpy as np


# 1. Stop JAX from preallocating all memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

#To obtain extremal eigenvalues of a vibrational Hamiltonian
import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard as lobpcg
import jax
from functools import partial
import optax
from . import tensor_utils as tu
import psutil
import os
import gc
jax.config.update("jax_enable_x64", True)
from scipy.sparse.linalg import LinearOperator, eigsh




def print_memory_usage(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # RSS (Resident Set Size) is the non-swapped physical memory user
    print(f"[{step_name}] Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")


def print_memory_usage2():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # Get system-wide memory stats
    vm = psutil.virtual_memory()
    # RSS (Resident Set Size) is the non-swapped physical memory user
    print(f"Process Usage: {mem_info.rss / 1024 ** 2:.2f} MB | "
          f"System Available: {vm.available / 1024 ** 2:.2f} MB | "
          f"System Used: {vm.percent}%")




@jax.jit
def uni_on_state(U, state):
  """
  Apply a unitary operator on a state vector represented as a tensor

  Parameters
  ----------
  U: jax.ndarray()
    A tensor of shape (nmodes, nmodals, nmodals) representing a unitary operator on each mode
  state: jax.ndarray()
    A tensor of shape (nmodals, nmodals, ... , nmodals) representing a state vector. 
    The number of dimensions of A is nmodes
    
  Returns
  -------
  jax.ndarray()
    A tensor of shape state.shape representing the state vector after applying the unitary operator
  """

  def apply_single_mode(current_state, u_matrix):
    # 1. Contract the unitary (u_matrix) with the FIRST mode (axis 0) of the state.
    #    u_matrix shape: (p, a)
    #    state shape:    (a, b, c, ...)
    #    Result shape:   (p, b, c, ...)
    res = jnp.tensordot(u_matrix, current_state, axes=([1], [0]))
    
    # 2. Move the processed axis (0) to the very end.
    #    This shifts all other modes to the left.
    #    Result shape: (b, c, ..., p)
    #    Now, the "next" mode (b) is at axis 0, ready for the next step in the scan.
    res = jnp.moveaxis(res, 0, -1)
    
    return res, None

    # Scan loops over the first dimension of U (the modes)
  final_state, _ = jax.lax.scan(apply_single_mode, state, U)
  return final_state




@jax.jit
def diag_to_ten(D):
    """
    Obtain tensor representation of a direct sum of diagonal operators
    
    Parameters
    ----------
    D: jax.ndarray
        Shape (nmodes, nmodals).
        
    Returns
    -------
    jax.ndarray
        Tensor of shape (nmodals,)*nmodes representing the direct sum of the diagonal operators.
    """
    nmodes = D.shape[0]
    nmodals = D.shape[1]

    #Think of the tensor representation of the diagonal operator as a potential
    # We will build the "Total Potential Grid" starting from zeros
    initial_potential = jnp.zeros((nmodals,)*nmodes)

    def accumulate_and_rotate(current_potential, d_vec):
        # 1. Reshape d_vec to broadcast against the FIRST dimension (Axis 0)
        #    d_vec shape: (nmodals,)
        #    We need:     (nmodals, 1, 1, ...)
        #    So it adds values to Axis 0 of current_potential
        reshaped_d = d_vec.reshape((d_vec.shape[0],) + (1,) * (nmodes - 1))
        
        # 2. Add the current diagonal to the potential
        #    (This represents the term D_k acting on mode k)
        new_potential = current_potential + reshaped_d
        
        # 3. Cyclic Shift: Move Axis 0 to the end (-1)
        #    Just like the unitary case, this brings the next physical mode 
        #    to the front (Axis 0) for the next iteration.
        new_potential = jnp.moveaxis(new_potential, 0, -1)
        
        return new_potential, None

    # Scan over D. Each step adds one mode's diagonal and rotates.
    # After 'nmodes' steps, we have rotated the tensor fully back to its original alignment.
    total_potential, _ = jax.lax.scan(accumulate_and_rotate, initial_potential, D)
    
    return total_potential




@jax.jit
def diag_on_state(D_ten, state):
  """
  Apply a direct sum of diagonal operator represented as a tensor on a state vector

  Parameters
  ----------
  D_ten: jax.ndarray()
    A tensor of shape (nmodals,)*nmodes
  state: jax.ndarray()
    A tensor of shape (nmodals,)nmodes

  Returns
  -------
  jax.ndarray()
    A tensor of shape state.shape representing the state vector after applying the direct sum of diagonal operators
  """
  return D_ten * state






@jax.jit
def eff_diag_on_state(Dk, state):
    """
    Computes sum of diagonal operators: H|psi> = (D1 + D2 + ... + Dn)|psi>
    Uses Scan to process one mode at a time.
    """
    # D_vecs shape: (nmodes, nmodals)
    
    def scan_body(carry, Dki):
        # unpack carry: (rotated_state, rotated_accumulator)
        curr_state, curr_acc = carry
        
        # 1. Multiply current axis 0 by the diagonal vector Dki
        #    'p..., p -> p...' broadcasts Dki along the first axis of curr_state
        term = jnp.einsum('p..., p -> p...', curr_state, Dki)
        
        # 2. Add to the accumulator
        new_acc = curr_acc + term
        
        # 3. Rotate BOTH the state and accumulator
        #    This moves axis 0 to the end, bringing the next mode to axis 0
        next_state = jnp.moveaxis(curr_state, 0, -1)
        next_acc = jnp.moveaxis(new_acc, 0, -1)
        # jax.debug.callback(print_memory_usage2)
        
        return (next_state, next_acc), None

    # Initial carry: (Original State, Zero Accumulator)
    init_acc = jnp.zeros_like(state)
    
    # Run scan over the diagonal vectors (modes)
    (final_state, final_acc), _ = jax.lax.scan(scan_body, (state, init_acc), Dk)
    
    # After nmodes rotations, final_acc is back in the original alignment
     
    return final_acc





@jax.jit
def obt_on_state(D, U, state):
  """
  Apply one-body operator to a state using the diagonal representation of the one-body operator

  Parameters
  ----------
  D_ten: jax.ndarray()
    A tensor of shape (nmodals,)*nmodes representing the diagonal representation of the one-body operator
  U: jax.ndarray()
    A tensor of shape (nmodes, nmodals, nmodals) representing the unitary diagonalizing the one-body operator
  state: jax.ndarray()
    A tensor of shape (nmodals, nmodals, ... , nmodals) representing a state vector. 
    The number of dimensions of A is nmodes

  Returns
  -------
  jax.ndarray()
    A tensor of shape state.shape representing the state vector after applying the one-body operator
  """
  Ustate = uni_on_state(jnp.transpose(U, (0, 2, 1)), state)
  # D_ten = diag_to_ten(D)
  # DUstate = D_ten * Ustate
  DUstate = eff_diag_on_state(D, Ustate)
  obtstate = uni_on_state(U, DUstate) 
   
  return obtstate




@jax.jit
def obt2_on_state(D, U, state):
  """
  Apply one-body operator squared to a state using the diagonal representation of the one-body operator

  Parameters
  ----------
  D_ten: jax.ndarray()
    A tensor of shape (nmodals,)*nmodes representing the diagonal representation of the one-body operator
  U: jax.ndarray()
    A tensor of shape (nmodes, nmodals, nmodals) representing the unitary diagonalizing the one-body operator
  state: jax.ndarray()
    A tensor of shape (nmodals, nmodals, ... , nmodals) representing a state vector. 
    The number of dimensions of A is nmodes

  Returns
  -------
  jax.ndarray()
    A tensor of shape state.shape representing the state vector after applying the square of the one-body operator
  """
  Ustate = uni_on_state(jnp.transpose(U, (0, 2, 1)), state)
  # D_ten = diag_to_ten(D)
  DUstate = eff_diag_on_state(D, Ustate)
  # jax.debug.callback(print_memory_usage2)
  # DUstate = D_ten * Ustate
  D2Ustate = eff_diag_on_state(D, DUstate)
  # D2Ustate = D_ten * DUstate
  obt2state = uni_on_state(U, D2Ustate) 
  # jax.debug.callback(print_memory_usage2)
   
  return obt2state




@partial(jax.jit, static_argnames=('nmodes', 'nmodals'))
def Ham_on_state(obtD, obtU, tbtD, tbtU, tbt_eigs, flt_state, nmodes, nmodals):
    # 1. Reshape flat state to tensor
    state = flt_state.reshape((nmodals,) * nmodes)
    
    # 2. Apply One-Body Term (Base)
    newstate = obt_on_state(obtD, obtU, state)

    # 3. Define the Scan Body (The Accumulator Loop)
    def scan_body(accumulated_sum, fragment_data):
        # Unpack the current fragment's data
        D_k, U_k, eig_k = fragment_data
        
        # Calculate term: eig_k * (U D^2 U^dagger state)
        op_result = obt2_on_state(D_k, U_k, state)
        contribution = eig_k * op_result
        
        # Update the running sum
        new_sum = accumulated_sum + contribution
        
        # Return: (New Carry, Output to Stack)
        # We return 'None' as the second argument. 
        # This tells JAX NOT to store the history of every fragment, saving huge memory.
        return new_sum, None


    # 5. Run the scan
    # jax.lax.scan iterates over the 0-th dimension of the tuple (tbtD, tbtU, tbt_eigs)
    total_fragment_sum, _ = jax.lax.scan(scan_body, newstate, (tbtD, tbtU, tbt_eigs))

    # 6. Sum and Flatten
    final_state = total_fragment_sum.ravel()

    return final_state







def get_ground_eigen_state(obt, tbt, tbt_cutoff = 1e-3, max_iter = 10000, tol = None, lr=1e-3):
  """
  Function to obtain the ground eigen state of the Hamiltonian (H) defined by the input one, two, and/or three body tensor. This function uses JAX's LOBPCG
  and obtains the ground state by approximating the highest excited state of c*I - H, where c is an upper bound to the gruond energy and I is the identity operator. 
  This shift ensures that the highest excited energy of c*I-H is positive.
  Parameters
  ----------
  obt: np.ndarray()
    One-body tensor of shape (nmodes, nmodals, nmodals)
  tbt: np.ndarray()
    Two-body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
  trbt: np.ndarray()
    Three-body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
  tbt_cutoff: float
    Cutoff to ignore small terms in the eigen decomposition of the reshaped tbt tensor

  Returns
  -------
  float
    Ground state energy
  np.ndarray()
    Ground state wavefunction
  """

  nmodes = obt.shape[0]
  nmodals = obt.shape[-1]

  #Obtaining eigenvalues and eigenvectors of OBT
  U0 = np.zeros((nmodes, nmodals, nmodals))
  D0 = np.zeros((nmodes, nmodals))
  for i in range (nmodes): 
    obt_i = obt[i]                                           
    D0[i], U0[i] = np.linalg.eigh(obt_i)
  # D0_ten = diag_to_ten(D0)
  
  #Performing eigen decomposition of tbt
  G_mat = tbt.reshape((nmodes*nmodals**2, nmodes*nmodals**2))
  G_val, G_vec = np.linalg.eigh(G_mat)
  G_val_mask = np.argsort(np.abs(G_val))[::-1]
  G_val = G_val[G_val_mask]
  G_vec = G_vec[:, G_val_mask]
  
  G_val_mask = np.where(np.abs(G_val) > tbt_cutoff)
  G_val = G_val[G_val_mask]                                  
  nfrags = len(G_val)
  G_vec_ten = G_vec[:, G_val_mask].reshape((nmodes, nmodals, nmodals, nfrags))  

  Us = np.zeros((nfrags, nmodes, nmodals, nmodals))
  # Ds_ten = np.zeros((nfrags,)+(nmodals,)*nmodes)
  Ds = np.zeros((nfrags, nmodes, nmodals))
  for f in range (nfrags):
    Dsf = np.zeros((nmodes, nmodals))
    for i in range (nmodes):
      obt_i = G_vec_ten[i, :, :, f]
      Dsf[i, :], Us[f, i, :, :] = np.linalg.eigh(obt_i)
    Ds[f] = Dsf
    # Ds_ten[f] = np.array(diag_to_ten(Dsf))

  print ("Diagonalizing two-body fragments complete. Procedeing to ground energy estimation.")
  print ("Number of tbt frags = ", nfrags, flush = True)

  # Usage inside ground_state function:
  print_memory_usage("Before JAX Arrays")
  D0, U0, Ds, Us, G_val = jnp.array(D0), jnp.array(U0), jnp.array(Ds), jnp.array(Us), jnp.array(G_val)
  print_memory_usage("After JAX Arrays")

  # Determining the shift c_shift (the ground energy upper bound) using variational principle
  ref_state = jnp.zeros(nmodals**nmodes, dtype=jnp.float64)
  ref_state = ref_state.at[0].set(1.0)
  H_ref = Ham_on_state(D0, U0, Ds, Us, G_val, ref_state, nmodes, nmodals)
  E_ref = jnp.vdot(ref_state, H_ref).real
  shift_margin = 1.0   # Buffer in cm^-1            
  c_shift = E_ref + shift_margin

  @jax.jit
  def matvec(flt_state):
    H_state =  Ham_on_state(D0, U0, Ds, Us, G_val, flt_state, nmodes, nmodals)
    return c_shift*flt_state - H_state

  @jax.jit
  def matvec_batched(block_state):
    return jax.vmap(matvec, in_axes=1, out_axes=1)(block_state)

  key = jax.random.PRNGKey(42)
  init_state = ref_state[:, None]
  # noise_state = jax.random.normal(key, shape=(nmodals**nmodes, 1), dtype=jnp.float64)
  # noise_state /= jnp.linalg.norm(noise_state)
  # init_state = ref_state + 1e-5*noise_state
  # init_state /= jnp.linalg.norm(init_state)

  print_memory_usage("Before lobpcg")
  shifted_val, vec, history = lobpcg(matvec_batched, init_state, m=max_iter, tol=tol)
  print_memory_usage("After lobpcg")
  print ("Number of lobpcg iterations: ", history)

  val = c_shift - float(shifted_val[0])

  #Checking the correctness of the approximated eigenpair
  psi = vec[:, 0]
  Hpsi = Ham_on_state(D0, U0, Ds, Us, G_val, psi, nmodes, nmodals)
  residual = float(jnp.linalg.norm(Hpsi - val * psi) / jnp.linalg.norm(psi))

    
  print("Approximate ground energy:", val, "cm-1", flush = True)
  print("Eigenpair residual:", residual, "cm-1", flush = True)

  return val, psi

  #Alternate implementation based on variational minimization
  obt_mat = tu.obt2_proj_mat(obt)
  tbt_mat = tu.tbt2_proj_mat(tbt)
  H_mat = obt_mat + tbt_mat
  H_mat = jnp.array(H_mat.toarray())



  @jax.jit
  def energy_loss(psi):
      # 1. Apply H to psi
      # H_psi = matvec(psi) 
      psi_vec = psi.reshape((nmodals**nmodes, 1))
      H_psi = H_mat @ psi_vec
      
      # 2. Compute expectation value <psi|H|psi>
      numerator = jnp.vdot(psi, H_psi)
      denominator = jnp.vdot(psi, psi)
      
      return (numerator / denominator).real

  # Optimize
  optimizer = optax.adam(learning_rate=lr)
  psi = np.random.random((nmodals**nmodes,1))
  psi = np.zeros(nmodals**nmodes,)
  psi[0] = 1
  opt_state = optimizer.init(psi)

  @jax.jit
  def update_step(psi, opt_state):
      loss, grads = jax.value_and_grad(energy_loss)(psi)
      updates, opt_state = optimizer.update(grads, opt_state, psi)
      psi = optax.apply_updates(psi, updates)
      return psi, opt_state, loss

  for i in range(max_iter):
      psi, opt_state, loss = update_step(psi, opt_state)
      if i%50 == 0:
        print(f"Ground State Energy: {loss}")
  
  return loss, psi












def get_ground_state_scipy(obt, tbt, tbt_cutoff = 1e-3, which='SA', k=1, max_iter = 10000, tol = 1e-3, ncv=5, return_eigenvectors = False, verbose = False):
  """
  Function to obtain the ground state of the Hamiltonian defined by the input one, two, and/or three body tensor.

  Parameters
  ----------
  obt: np.ndarray()
    One-body tensor of shape (nmodes, nmodals, nmodals)
  tbt: np.ndarray()
    Two-body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
  trbt: np.ndarray()
    Three-body tensor of shape (nmodes, nmodals, nmodals, nmodes, nmodals, nmodals, nmodes, nmodals, nmodals)
  tbt_cutoff: float
    Cutoff to ignore small terms in the eigen decomposition of the reshaped tbt tensor

  Returns
  -------
  float
    Ground state energy
  np.ndarray()
    Ground state wavefunction
  """

  nmodes = obt.shape[0]
  nmodals = obt.shape[-1]

  #Obtaining eigenvalues and eigenvectors of OBT
  U0 = np.zeros((nmodes, nmodals, nmodals))
  D0 = np.zeros((nmodes, nmodals))
  for i in range (nmodes): 
    obt_i = obt[i]                                           # The negative sign is used since lobpcg finds largest and not smallest eigenvector
    D0[i], U0[i] = np.linalg.eigh(obt_i)
  # D0_ten = diag_to_ten(D0)
  
  #Performing eigen decomposition of tbt
  G_mat = tbt.reshape((nmodes*nmodals**2, nmodes*nmodals**2))
  G_val, G_vec = np.linalg.eigh(G_mat)
  G_val_mask = np.argsort(np.abs(G_val))[::-1]
  G_val = G_val[G_val_mask]
  G_vec = G_vec[:, G_val_mask]
  
  G_val_mask = np.where(np.abs(G_val) > tbt_cutoff)
  G_val = G_val[G_val_mask]                                  # The negative sign is used since lobpcg finds largest and not smallest eigenvector
  nfrags = len(G_val)
  G_vec_ten = G_vec[:, G_val_mask].reshape((nmodes, nmodals, nmodals, nfrags))  

  Us = np.zeros((nfrags, nmodes, nmodals, nmodals))
  # Ds_ten = np.zeros((nfrags,)+(nmodals,)*nmodes)
  Ds = np.zeros((nfrags, nmodes, nmodals))
  for f in range (nfrags):
    Dsf = np.zeros((nmodes, nmodals))
    for i in range (nmodes):
      obt_i = G_vec_ten[i, :, :, f]
      Dsf[i, :], Us[f, i, :, :] = np.linalg.eigh(obt_i)
    Ds[f] = Dsf
    # Ds_ten[f] = np.array(diag_to_ten(Dsf))

  if verbose:
    print ("Diagonalizing two-body fragments complete. Procedeing to ground energy estimation.")
    print ("Number of tbt frags = ", nfrags)

  # Usage inside ground_state function:
  if verbose:
    print_memory_usage("Before JAX Arrays")
  D0, U0, Ds, Us, G_val = jnp.array(D0), jnp.array(U0), jnp.array(Ds), jnp.array(Us), jnp.array(G_val)
  if verbose:
    print_memory_usage("After JAX Arrays")


  def matvec_wrapper(flt_state):
    return Ham_on_state(D0, U0, Ds, Us, G_val, flt_state, nmodes, nmodals)
    # return (H_mat @ flt_state.reshape((nmodals**nmodes,1))).ravel()

  dim = nmodals**nmodes

  A_op = LinearOperator((dim, dim), matvec=matvec_wrapper, dtype=np.float64)

  # 3. Solve for the Ground State
  if verbose:
    print("Starting iterative solver...")
    print_memory_usage("Before eigsh")
  if return_eigenvectors:
    eigenvalues, eigenvectors = eigsh(A_op, k=k, which=which, maxiter=max_iter, tol=tol, ncv=ncv, return_eigenvectors=True)
  else:
    eigenvalues = eigsh(A_op, k=k, which=which, maxiter=max_iter, tol=tol, ncv=ncv, return_eigenvectors=False)
  if verbose:
    print_memory_usage("After eigsh")

   

  del U0, D0, Us, Ds, G_val, G_mat, G_vec, G_vec_ten, A_op
  gc.collect()


  if return_eigenvectors:
    return eigenvalues, eigenvectors
  else:
    return eigenvalues










