import torch
import json
import pandas as pd
from utils.timer import Timer
from utils.customtypes import Hardware, Circuit
from sampler.randomcircuit import RandomCircuit
from utils.plotter import drawCircuit
from utils.allocutils import check_sanity, swaps_from_alloc, count_swaps, check_sanity_swap, get_all_checkpoints
from qalloczero.alg.ts import TSConfig
# from qalloczero.alg.alphazero import AlphaZero
from qalloczero.alg.directalloc import DirectAllocator


def validate(model_name: str, chkpt: int):
  torch.manual_seed(42)
  n_qubits = 16
  n_slices = 32
  n_circuits = 16
  core_caps = torch.tensor([4]*4, dtype=torch.int)
  n_cores = core_caps.shape[0]
  core_conn = torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  hardware = Hardware(core_capacities=core_caps, core_connectivity=core_conn)
  algos = dict(
    da_seq  = DirectAllocator.load(model_name,    device="cuda", checkpoint=chkpt).set_mode(DirectAllocator.Mode.Sequential),
    da_par  = DirectAllocator.load(model_name,    device="cuda", checkpoint=chkpt).set_mode(DirectAllocator.Mode.Parallel),
    # azero =               AlphaZero.load("trained/da_v2_ft", device="cpu"),
  )
  print(f"Loaded checkpoint: {list(algos.values())[0].checkpoint}")
  cfg = TSConfig(
    target_tree_size=512,
    noise=0.2,
    dirichlet_alpha=1.0,
    discount_factor=0.0,
    action_sel_temp=0,
    ucb_c1=0.125,
    ucb_c2=500,
  )
  sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices, reflow=False)
  circuits = [sampler.sample() for _ in range(n_circuits)]
  for (name, algo) in algos.items():
    print(f"[*] Optimizing {name}")
    with Timer.get('t'):
      if isinstance(algo, DirectAllocator):
        results = []
        for circ in circuits:
          results.append(algo.optimize(circ, hardware=hardware, verbose=True))
      elif isinstance(algo, AlphaZero):
        results = algo.optimize_mult(circuits, cfg, hardware=hardware, verbose=True)
      else:
        raise Exception("Unrecognized algorithm type")
    norm_res = torch.tensor([res[1]/circ.n_gates_norm for (res, circ) in zip(results,circuits)])
    for (res, circuit) in zip(results, circuits):
      check_sanity(allocs=res[0], circuit=circuit, hardware=hardware)
    norm_swaps = [
      count_swaps(swaps_from_alloc(res[0], n_cores))/circ.n_gates_norm for (res, circ) in zip(results,circuits)
    ]
    print(f" + t={Timer.get('t').time:.2f}s avg_cost={norm_res.mean().item():.4f} ({norm_res.std().item():.2f}) avg_swaps={sum(norm_swaps)/len(norm_swaps):.4f}")


def benchmark(model_name: str, chkpt: int):
  circuit_names = [
    "qft", # Exact
    # "quantum_volume",
    "graph_state", # Exact
    # "drapper_adder",
    # "cuccaro_adder", # A bit over
    # "qnn",
    "deutsch_jozsa", # Exact
  ]
  circuits = {name: Circuit.from_qasm(f'circuits/{name}_100.qasm', 100) for name in circuit_names}
  # A2A configuration
  hardware = Hardware(
    core_capacities=torch.tensor([10]*10),
    core_connectivity=(torch.ones(size=(10,10)) - torch.eye(10)),
  )


  algos = dict(
    da_sequential = DirectAllocator.load(model_name,    device="cuda", checkpoint=chkpt).set_mode(DirectAllocator.Mode.Sequential),
    da_parallel   = DirectAllocator.load(model_name,    device="cuda", checkpoint=chkpt).set_mode(DirectAllocator.Mode.Parallel),
    # azero =               AlphaZero.load("trained/da_v2_ft", device="cpu"),
  )
  cfg = TSConfig(
    target_tree_size=512,
    noise=0.2,
    dirichlet_alpha=1.0,
    discount_factor=0.0,
    action_sel_temp=0,
    ucb_c1=0.125,
    ucb_c2=500,
  )

  for (name, algo) in algos.items():
    print(f"[*] Optimizing with {name}")
    for cname, circ in circuits.items():
      with Timer.get('t'):
        if isinstance(algo, DirectAllocator):
          allocs, cost = algo.optimize(circ, hardware=hardware, verbose=True)
        elif isinstance(algo, AlphaZero):
          allocs, cost, _, er = algo.optimize(circ, cfg, hardware=hardware, verbose=True)
        else:
          raise Exception("Unrecognized algorithm type")
      print(f" + {cname}: t={Timer.get('t').time:.2f}s cost={cost} ({cost/(circ.n_gates_norm+1):.2f})")


def compare_w_sota(model_name: str, chkpt: int, data_dir: str):
  n_qubits=100
  base_res = pd.read_csv(f'{data_dir}/sota_cost_{n_qubits}.csv', index_col=0)
  base_times = pd.read_csv(f'{data_dir}/sota_time_{n_qubits}.csv', index_col=0)
  base_res.index.name = 'circuit'
  base_times.index.name = 'circuit'

  with open(f'{data_dir}/all_{n_qubits}.json', 'r') as f:
    data = json.load(f)

  n_qubits = data['n_qubits']
  circuit_slices = data['circuits']
  circuits = {}
  for (name, slices) in circuit_slices.items():
    circuits[name] = Circuit(slice_gates=slices, n_qubits=n_qubits)
  
  algos = dict(
    # da_sequential = DirectAllocator.load(model_name, device="cuda", checkpoint=chkpt).set_mode(DirectAllocator.Mode.Sequential),
    da_parallel   = DirectAllocator.load(model_name, device="cuda", checkpoint=chkpt).set_mode(DirectAllocator.Mode.Parallel),
  )

  n_cores = n_qubits//10
  hardware = Hardware(
    core_capacities=torch.tensor([10]*n_cores),
    core_connectivity=(torch.ones(size=(n_cores,n_cores)) - torch.eye(n_cores)),
  )

  my_results = {}
  my_times = {}

  # for (name, algo) in algos.items():
  #   my_results[name] = {}
  #   my_times[name] = {}
  #   print(f"[*] Optimizing with {name}")
  #   for cname, circ in circuits.items():
  #     with Timer.get('t'):
  #       if isinstance(algo, DirectAllocator):
  #         allocs, cost = algo.optimize(circ, hardware=hardware, verbose=True)
  #       elif isinstance(algo, AlphaZero):
  #         pass
  #         # allocs, cost, _, er = algo.optimize(circ, cfg, hardware=hardware, verbose=True)
  #       else:
  #         raise Exception("Unrecognized algorithm type")
  #     print(f" + {cname}: t={Timer.get('t').time:.2f}s cost={cost} ({cost/(circ.n_gates_norm+1):.2f})")
  #     my_results[name][cname] = cost
  #     my_times[name][cname] = Timer.get('t').time
  
  # print(my_results)
  # print(my_times)

  my_results_seq = {'random0': 321.0, 'random1': 311.0, 'random2': 286.0, 'random3': 345.0, 'random4': 320.0, 'random5': 352.0, 'random6': 340.0, 'random7': 317.0, 'random8': 300.0, 'random9': 291.0, 'random10': 329.0, 'random11': 327.0, 'random12': 314.0, 'random13': 300.0, 'random14': 334.0, 'random15': 258.0, 'random16': 320.0, 'random17': 306.0, 'random18': 329.0, 'random19': 304.0, 'random20': 313.0, 'random21': 302.0, 'random22': 327.0, 'random23': 313.0, 'random24': 330.0, 'random25': 349.0, 'random26': 319.0, 'random27': 328.0, 'random28': 337.0, 'random29': 272.0, 'random30': 263.0, 'random31': 333.0, 'random32': 325.0, 'random33': 288.0, 'random34': 297.0, 'random35': 326.0, 'random36': 338.0, 'random37': 344.0, 'random38': 307.0, 'random39': 300.0, 'random40': 329.0, 'random41': 303.0, 'random42': 316.0, 'random43': 268.0, 'random44': 304.0, 'random45': 311.0, 'random46': 309.0, 'random47': 310.0, 'random48': 332.0, 'random49': 326.0, 'random50': 280.0, 'random51': 298.0, 'random52': 340.0, 'random53': 307.0, 'random54': 333.0, 'random55': 309.0, 'random56': 321.0, 'random57': 277.0, 'random58': 278.0, 'random59': 312.0, 'random60': 244.0, 'random61': 314.0, 'random62': 340.0, 'random63': 328.0, 'cuccaro_adder': 102.0, 'deutsch_jozsa': 36.0, 'drapper_adder': 1010.0, 'graph_state': 2017.0, 'qft': 1150.0, 'qnn': 4923.0, 'quantum_volume': 4545.0}
  my_times_seq   = {'random0': 3.0933570861816406, 'random1': 2.416994571685791, 'random2': 2.4034454822540283, 'random3': 2.708552360534668, 'random4': 2.5132906436920166, 'random5': 2.8054702281951904, 'random6': 2.772214889526367, 'random7': 2.489854335784912, 'random8': 2.4200291633605957, 'random9': 2.387716770172119, 'random10': 2.736172676086426, 'random11': 2.6547930240631104, 'random12': 2.5881853103637695, 'random13': 2.478419780731201, 'random14': 2.6044390201568604, 'random15': 2.1175825595855713, 'random16': 2.5271430015563965, 'random17': 2.538989543914795, 'random18': 2.7186532020568848, 'random19': 2.4445924758911133, 'random20': 2.497272491455078, 'random21': 2.5069034099578857, 'random22': 2.566999673843384, 'random23': 2.492326498031616, 'random24': 2.8206372261047363, 'random25': 2.818507432937622, 'random26': 2.6748158931732178, 'random27': 2.6805646419525146, 'random28': 2.6851537227630615, 'random29': 2.22737717628479, 'random30': 2.2086150646209717, 'random31': 2.65437650680542, 'random32': 2.5882229804992676, 'random33': 2.4918837547302246, 'random34': 2.495954990386963, 'random35': 2.610290288925171, 'random36': 2.8802037239074707, 'random37': 2.773049831390381, 'random38': 2.6312170028686523, 'random39': 2.4847097396850586, 'random40': 2.676625967025757, 'random41': 2.580989122390747, 'random42': 2.5647389888763428, 'random43': 2.3986191749572754, 'random44': 2.588270902633667, 'random45': 2.5192084312438965, 'random46': 2.6174004077911377, 'random47': 2.4453065395355225, 'random48': 2.599383592605591, 'random49': 2.646714925765991, 'random50': 2.305136203765869, 'random51': 2.573187828063965, 'random52': 2.6622722148895264, 'random53': 2.5514047145843506, 'random54': 2.57346248626709, 'random55': 2.401538848876953, 'random56': 2.605832576751709, 'random57': 2.2034144401550293, 'random58': 2.346606969833374, 'random59': 2.575636625289917, 'random60': 2.1261796951293945, 'random61': 2.497699022293091, 'random62': 2.693202257156372, 'random63': 2.648653030395508, 'cuccaro_adder': 1.4160172939300537, 'deutsch_jozsa': 0.7630608081817627, 'drapper_adder': 7.403338670730591, 'graph_state': 11.316722631454468, 'qft': 6.442526578903198, 'qnn': 22.612739324569702, 'quantum_volume': 18.04974627494812}

  my_results_par = {'random0': 287.0, 'random1': 275.0, 'random2': 258.0, 'random3': 308.0, 'random4': 290.0, 'random5': 315.0, 'random6': 292.0, 'random7': 261.0, 'random8': 250.0, 'random9': 237.0, 'random10': 291.0, 'random11': 280.0, 'random12': 264.0, 'random13': 243.0, 'random14': 300.0, 'random15': 223.0, 'random16': 265.0, 'random17': 282.0, 'random18': 277.0, 'random19': 270.0, 'random20': 286.0, 'random21': 272.0, 'random22': 280.0, 'random23': 288.0, 'random24': 294.0, 'random25': 313.0, 'random26': 273.0, 'random27': 282.0, 'random28': 293.0, 'random29': 233.0, 'random30': 232.0, 'random31': 275.0, 'random32': 297.0, 'random33': 258.0, 'random34': 272.0, 'random35': 283.0, 'random36': 311.0, 'random37': 294.0, 'random38': 267.0, 'random39': 263.0, 'random40': 285.0, 'random41': 273.0, 'random42': 273.0, 'random43': 237.0, 'random44': 249.0, 'random45': 278.0, 'random46': 259.0, 'random47': 285.0, 'random48': 294.0, 'random49': 279.0, 'random50': 251.0, 'random51': 256.0, 'random52': 286.0, 'random53': 265.0, 'random54': 301.0, 'random55': 287.0, 'random56': 297.0, 'random57': 255.0, 'random58': 249.0, 'random59': 286.0, 'random60': 215.0, 'random61': 291.0, 'random62': 304.0, 'random63': 302.0, 'cuccaro_adder': 74.0, 'deutsch_jozsa': 66.0, 'drapper_adder': 1191.0, 'graph_state': 1966.0, 'qft': 1519.0, 'qnn': 3803.0, 'quantum_volume': 4201.0}
  my_times_par   = {'random0': 3.6561942100524902, 'random1': 2.512378692626953, 'random2': 2.5349738597869873, 'random3': 2.8436174392700195, 'random4': 2.7066149711608887, 'random5': 2.866131544113159, 'random6': 2.8941895961761475, 'random7': 2.719475030899048, 'random8': 2.4267520904541016, 'random9': 2.403184413909912, 'random10': 2.766892433166504, 'random11': 2.615229606628418, 'random12': 2.5680553913116455, 'random13': 2.3507888317108154, 'random14': 2.396566867828369, 'random15': 2.1217212677001953, 'random16': 2.2135229110717773, 'random17': 2.3695785999298096, 'random18': 2.3695766925811768, 'random19': 2.347443103790283, 'random20': 2.4734861850738525, 'random21': 2.224553108215332, 'random22': 2.296598196029663, 'random23': 2.561000347137451, 'random24': 2.663309097290039, 'random25': 2.7099201679229736, 'random26': 2.1056337356567383, 'random27': 2.3808670043945312, 'random28': 2.514529228210449, 'random29': 2.2824819087982178, 'random30': 2.115140438079834, 'random31': 2.384944438934326, 'random32': 2.4220917224884033, 'random33': 2.2361135482788086, 'random34': 2.521413803100586, 'random35': 2.432438850402832, 'random36': 2.6107091903686523, 'random37': 2.4572651386260986, 'random38': 2.4505233764648438, 'random39': 2.3509585857391357, 'random40': 2.4949893951416016, 'random41': 2.433225154876709, 'random42': 2.499342441558838, 'random43': 2.1619319915771484, 'random44': 2.3260724544525146, 'random45': 2.4944207668304443, 'random46': 2.4427871704101562, 'random47': 2.5606040954589844, 'random48': 2.686518907546997, 'random49': 2.672487258911133, 'random50': 2.3058478832244873, 'random51': 2.4585697650909424, 'random52': 2.6865668296813965, 'random53': 2.3654723167419434, 'random54': 2.64902663230896, 'random55': 2.488974094390869, 'random56': 2.673936367034912, 'random57': 2.397840976715088, 'random58': 2.229001998901367, 'random59': 2.465003490447998, 'random60': 2.188922882080078, 'random61': 2.576011896133423, 'random62': 2.6830086708068848, 'random63': 2.7534027099609375, 'cuccaro_adder': 1.2348334789276123, 'deutsch_jozsa': 1.105422019958496, 'drapper_adder': 8.434072971343994, 'graph_state': 12.465535879135132, 'qft': 8.940690279006958, 'qnn': 19.75545048713684, 'quantum_volume': 18.460174083709717}

  my_results = {'da_fast_sequential': my_results_seq, 'da_fast_parallel': my_results_par}
  my_times = {'da_fast_sequential': my_times_seq, 'da_fast_parallel': my_times_par}

  my_results = pd.DataFrame.from_dict(my_results, orient="index").T
  all_res = pd.concat([base_res, my_results], axis=1)
  all_res.to_csv(f'{data_dir}/my_cost_{n_qubits}_.csv', index=True)
  my_times = pd.DataFrame.from_dict(my_times, orient="index").T
  all_times = pd.concat([base_times, my_times], axis=1)
  all_times.to_csv(f'{data_dir}/my_time_{n_qubits}_.csv', index=True)
