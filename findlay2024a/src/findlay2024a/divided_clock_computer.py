import functools
import itertools

import graphiit
import numpy as np
import pyphi.utils
from tqdm.auto import tqdm

from . import findlay2024a as cc


@functools.cache
def get_target_labels(k):
    n = 2**k
    return [f"T{i}" for i in range(n)]


@functools.cache
def get_program_labels(k):
    n = 2**k
    s = 2**n
    return [[f"{l}/{i}" for l in get_target_labels(k)] for i in range(s)]


@functools.cache
def get_instruction_register_labels(k):
    n = 2**k
    s = 2**n
    return [f"IR/{i}" for i in range(s)]


@functools.cache
def get_clock_labels(k, flat=False):
    c = ["C0"]
    x = [f"X{i + 1}" for i in range(k)]
    a = [f"A{i + 1}" for i in range(k)]
    if flat:
        return c + [xa for pair in zip(x, a) for xa in pair]
    else:
        return c, x, a


@functools.cache
def get_data_register_labels(k):
    return [[f"{l}R0", f"{l}R1", f"{l}R2"] for l in get_target_labels(k)]


@functools.cache
def get_data_register_output_labels(k):
    return [dr[0] for dr in get_data_register_labels(k)]


@functools.cache
def get_data_register_enable_labels(k):
    return [dr[1] for dr in get_data_register_labels(k)]


@functools.cache
def get_buffer_labels(k):
    n = 2**k
    return [f"B{i}" for i in range(2 * n - 5)]


@functools.cache
def get_mux_labels(k):
    n = 2**k
    s = 2**n
    return [f"M{i}" for i in range(s)]


@functools.cache
def get_mux_out_label():
    return "O"


@functools.cache
def get_clock_config(k, feedback):
    c, x, a = get_clock_labels(k)

    # Configure the oscillating core of the clock itself
    if feedback:
        c_config = [
            (c[0], "NOR", c[0], *get_data_register_enable_labels(k)),
        ]  # Needs to start ON
    else:
        c_config = [(c[0], "NOT", c[0])]  # Needs to start ON

    # Configure the frequency dividers
    ca = c + a
    x_config = [(x[i], "XOR", x[i], ca[i]) for i in range(k)]

    def _and(inputs):
        return (sum(inputs[:-1]) == len(inputs[:-1])) and not inputs[-1]

    cx = c + x
    a_config = [(a[i], _and, *cx[: i + 2], a[i]) for i in range(k)]

    return c_config + [xa for pair in zip(x_config, a_config) for xa in pair]


@functools.cache
def get_program_config(k, feedback):
    n = 2**k
    s = 2**n
    p = get_program_labels(k)
    p_config = [[(line[i], "COPY", line[(i + 1) % n]) for i in range(n)] for line in p]

    if feedback:
        fb_src = get_instruction_register_labels(k)
        fb_tgt = -2
        for i in range(s):
            p_config[i][fb_tgt] = (p[i][fb_tgt], "OR", p[i][fb_tgt + 1], fb_src[i])
    return p_config


@functools.cache
def get_instruction_register_config(k):
    def _func(inputs):
        assert len(inputs) == 2, f"IR unit expected 2 inputs, got {len(inputs)}"
        return inputs[0] and not inputs[1]

    _, x, _ = get_clock_labels(k)
    return [
        (m, _func, line[0], x[-1])
        for m, line in zip(get_instruction_register_labels(k), get_program_labels(k))
    ]


@functools.cache
def get_data_register_config(k):
    _, _, a = get_clock_labels(k)
    bmo = get_buffer_labels(k) + [get_mux_out_label()]
    return [
        [
            (dr[0], "XOR", dr[0], dr[1]),
            (dr[1], "AND", dr[2], a[-1]),
            (dr[2], "XOR", dr[0], bmo[i]),
        ]
        for i, dr in enumerate(get_data_register_labels(k))
    ]


@functools.cache
def get_buffer_config(k):
    bmo = get_buffer_labels(k) + [get_mux_out_label()]
    return [(bmo[i], "COPY", bmo[i + 1]) for i in range(len(bmo) - 1)]


@functools.cache
def get_mux_function(selector_state):
    n = len(selector_state)

    def _func(inputs):
        assert len(inputs) == n + 1, f"{n}-bit mux must receive {n + 1} inputs."
        selector_inputs = inputs[:n]
        return (tuple(selector_inputs) == tuple(selector_state)) and inputs[n]

    return _func


@functools.cache
def get_mux_config(k):
    n = 2**k
    return [
        (m, get_mux_function(s), *get_data_register_output_labels(k), ir)
        for m, s, ir in zip(
            get_mux_labels(k),
            pyphi.utils.all_states(n),
            get_instruction_register_labels(k),
        )
    ]


@functools.cache
def get_mux_out_config(k):
    return [(get_mux_out_label(), "OR", *get_mux_labels(k))]


def flat(l):
    return list(itertools.chain.from_iterable(l))


@functools.cache
def get_graph_config(k, feedback):
    return (
        get_clock_config(k, feedback)
        + flat(get_data_register_config(k))
        + flat(get_program_config(k, feedback))
        + get_instruction_register_config(k)
        + get_mux_config(k)
        + get_mux_out_config(k)
        + get_buffer_config(k)
    )


def get_initial_state_config(target_tpm, target_initial_state, feedback):
    tpm = np.array(target_tpm)
    state = np.array(target_initial_state)

    k = int(np.log2(state.size))
    program_labels = np.array(get_program_labels(k))
    data_labels = np.array(get_data_register_output_labels(k))
    c, _, _ = get_clock_labels(k)

    program_on = program_labels[np.where(tpm)].tolist()
    data_on = data_labels[np.where(state)].tolist()
    return {"on": program_on + data_on + c}


def get_graphiit(
    k,
    target_tpm,
    target_initial_state,
    feedback,
):
    graph_config = get_graph_config(
        k,
        feedback,
    )
    state_config = get_initial_state_config(
        target_tpm,
        target_initial_state,
        feedback,
    )
    return graphiit.Graph(graph_config, state_config)


def get_pQrS_graphiit(feedback):
    tpm = np.array(
        [
            [0, 0, 0, 0],  # 0000
            [0, 0, 0, 1],  # 0001
            [0, 0, 1, 0],  # 0010
            [0, 0, 1, 1],  # 0011
            [0, 1, 0, 0],  # 0100
            [0, 1, 1, 1],  # 0101
            [1, 0, 0, 1],  # 0110
            [1, 0, 0, 0],  # 0111
            [1, 0, 1, 1],  # 1000
            [0, 1, 1, 0],  # 1001
            [1, 1, 1, 0],  # 1010
            [0, 1, 0, 1],  # 1011
            [1, 1, 0, 1],  # 1100
            [1, 1, 1, 1],  # 1101
            [1, 0, 1, 0],  # 1110
            [1, 1, 0, 0],  # 1111
        ]
    )
    state = (0, 1, 0, 1)  # PQRS
    n = len(state)
    k = int(np.log2(n))
    return get_graphiit(k, tpm, state, feedback)


def get_WXyZ_graphiit(feedback):
    tpm = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1.0, 1],
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
        ]
    )

    state = (1, 1, 0, 1)  # WXyZ
    n = len(state)
    k = int(np.log2(n))
    return get_graphiit(k, tpm, state, feedback)


def check_tpm_simulation(tpm, feedback: bool):
    network = pyphi.Network(tpm)
    cycles = cc.get_cycles(network)

    k = int(np.log2(len(network)))
    n = 2**k
    outputs = get_data_register_output_labels(k)
    for cycle in cycles:
        initial_state = cycle[0]
        graph = get_graphiit(k, tpm, initial_state, feedback)
        graph.tic(1)
        simulated_cycle = []
        for state in cycle:
            simulated_cycle.append(tuple(graph.get_state_array(outputs)))
            if not tuple(graph.get_state_array(outputs)) == tuple(state):
                msg = (
                    f"Expected state: {tuple(state)}.\n"
                    f"Observed state: {tuple(graph.get_state_array(outputs))}.\n"
                    f"Cycle: {cycle}.\n"
                    f"TPM: {tpm}.\n"
                )
                raise ValueError(msg)
            graph.tic(2 * n)


def check_k_simulation(k, n_trials, feedback: bool):
    n = 2**k
    all_states = np.array(list(pyphi.utils.all_states(n)))
    for _ in tqdm(range(n_trials)):
        ixs = np.random.choice(2**n, 2**n, replace=True)
        tpm = all_states[ixs]
        check_tpm_simulation(tpm, feedback)
