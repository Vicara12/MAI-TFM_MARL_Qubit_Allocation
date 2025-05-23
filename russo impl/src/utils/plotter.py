import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import reduce
from typing import Tuple



def drawCircuit(circuit_slice_gates: Tuple[Tuple[Tuple[int, int], ...], ...],
                num_lq, title="",
                figsize_scale: float=1.0):
  ''' Draw the quantum circuit with the time slices.

  Arguments follow the CircuitSampler convention.
  '''
  vlines = [0]
  for circuit_slice in circuit_slice_gates:
    vlines.append(vlines[-1] + len(circuit_slice))
  vlines = vlines[1:-1]
  circuit_gates = reduce(lambda a,b: a+b, circuit_slice_gates, ())
  num_steps = len(circuit_gates)
  _, ax = plt.subplots(figsize=(num_steps * figsize_scale, num_lq))
  for q in range(num_lq):
    ax.hlines(y=q, xmin=0, xmax=num_steps, color='black', linewidth=1)
  for x in vlines:
    ax.vlines(x-0.5, ymin=-0.5, ymax=num_lq + 0.5, linestyles='dotted', colors='gray', linewidth=1)
  for i, (q1, q2) in enumerate(circuit_gates):
    y1, y2 = min(q1, q2), max(q1, q2)
    ax.plot([i]*2, [y1, y2], color='black', linewidth=2, marker='o')
  ax.set_yticks(range(num_lq))
  ax.set_yticklabels([f'q[{i}]' for i in range(num_lq)])
  ax.set_xticks(range(num_steps))
  ax.set_xlim(-1, num_steps)
  ax.set_ylim(-1, num_lq)
  ax.invert_yaxis()
  ax.set_title(title)
  plt.tight_layout()
  plt.show()


def drawQubitAllocation(qubit_allocation: torch.Tensor,
                        core_sizes: Tuple[int, ...]=None,
                        circuit_slice_gates: Tuple[Tuple[Tuple[int, int], ...], ...]=None,
                        figsize_scale: float=1.0):
    """ Draws the flow of qubit allocations across columns (time steps).
    
    Parameters:
    - qubit_allocation: tensor in which each column indicates a qubit allocation for a time step and
        each row indicates which logical qubit is assigned to a certain physical qubit.
    - core_sizes (optional): size of each core. If provided the plot will contain horizontal
        lines separating the physical qubits of each core. It is assumed that the qubits of the core
        are consecutive.
    - circuit_slice_gates: follows the CircuitSampler convention.
    """
    Path = matplotlib.path.Path
    num_steps = qubit_allocation.shape[1]
    num_pq = qubit_allocation.shape[0]
    
    # Extract all unique qubit IDs
    all_qubits = sorted(set(q.item() for col in qubit_allocation.T for q in col))
    color_map = {q: matplotlib.cm.viridis(i / len(all_qubits)) for i, q in enumerate(all_qubits)}

    _, ax = plt.subplots(figsize=(num_steps * figsize_scale, num_pq*num_pq/num_steps))

    if core_sizes is not None:
      assert (sum(core_sizes) == num_pq), "sum of core sizes does not match number of physical qubits"
      core_line_pos = [0]
      for core_size in core_sizes:
        core_line_pos.append(core_line_pos[-1]+core_size)
      core_line_pos = core_line_pos[1:-1]
      for cl_pos in core_line_pos:
        ax.hlines(y=num_pq-cl_pos-0.5, xmin=-0.3, xmax=num_steps+0.3, color='gray', linestyles='dotted', linewidth=1)
    
    if circuit_slice_gates is not None:
      for t, circuit_slice in enumerate(circuit_slice_gates):
         alloc_slice = qubit_allocation[:,t].squeeze().tolist()
         for gate in circuit_slice:
            pq0 = alloc_slice.index(gate[0])
            pq1 = alloc_slice.index(gate[1])
            verts = [(t - 0.3,       num_pq - pq0 - 1),
                     (t - 0.3 - 0.2, num_pq - pq0 - 1),
                     (t - 0.3 - 0.2, num_pq - pq1 - 1),
                     (t - 0.3,       num_pq - pq1 - 1)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', edgecolor='black', lw=1.25, alpha=0.6)
            ax.add_patch(patch)

    # Draw nodes and flows
    last_q_positions = {}
    for t in range(num_steps):
        column = qubit_allocation[:,t].squeeze().tolist()
        for y, qubit in enumerate(column):
            y = num_pq - y - 1
            qubit = int(qubit)
            # Draw square
            color = color_map[qubit]
            rect = patches.Rectangle((t - 0.3, y - 0.3), 0.6, 0.6, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.text(t, y, f"lq {qubit}", ha='center', va='center', fontsize=6, color='white')

            # Draw flow from previous timestep
            if qubit in last_q_positions:
                prev_t, prev_y = last_q_positions[qubit]
                if prev_y != y:
                  verts = [
                      (prev_t, prev_y),
                      (prev_t + 0.4, prev_y),
                      (t - 0.4, y),
                      (t, y)
                  ]
                  codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                  path = Path(verts, codes)
                  patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=2, alpha=0.5)
                  ax.add_patch(patch)
            last_q_positions[qubit] = (t, y)
    ax.set_xlim(-0.5 if circuit_slice_gates is None else -0.75, num_steps - 0.5)
    ax.set_ylim(-0.5, num_pq - 0.5)
    ax.set_xticks(range(num_steps))
    ax.set_yticks(range(num_pq))
    ax.set_yticklabels(list(range(num_pq))[::-1])
    ax.set_xlabel("Time")
    ax.set_ylabel("Physical qubit")
    ax.set_aspect(num_pq/num_steps)
    plt.grid(False)
    plt.show()