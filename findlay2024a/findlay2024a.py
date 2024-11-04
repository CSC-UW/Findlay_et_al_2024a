from pathlib import Path

import graphiit
import matplotlib.pyplot as plt
import numpy as np
import pyphi
import pyphi.compute
import pyphi.convert

# import pyphi.exceptions
# import pyphi.network_generator
import pyphi.new_big_phi
import pyphi.utils

# import pyphi.validate
import pyphi.visualize


def get_sia_and_ces(
    graph: graphiit.Graph, unit_labels: list[str], vokram_blanket: bool = True
):
    if vokram_blanket:
        g = graph.vokram_blanket(unit_labels)
    else:
        g = graph.subgraph(unit_labels)
    network = g.pyphi_network()
    state = g.get_state_array(dtype=int)
    subsystem_indices = tuple(list(g.nodes).index(u) for u in unit_labels)
    subsystem = pyphi.Subsystem(network, state, subsystem_indices)
    sia = pyphi.new_big_phi.sia(subsystem)
    ces = pyphi.new_big_phi.phi_structure(subsystem, sia)
    return sia, ces, state


# Shouldn't be necessary anymore.
# def reachable_subsystems(network, indices, state, **kwargs):
#     """A generator over all subsystems in a valid state."""
#     pyphi.validate.is_network(network)

#     # Return subsystems largest to smallest to optimize parallel
#     # resource usage.
#     for subset in pyphi.utils.powerset(indices, nonempty=True, reverse=True):
#         try:
#             cause_subsystem = pyphi.Subsystem(
#                 network, state, subset, backward_tpm=True, **kwargs
#             )
#             effect_subsystem = pyphi.Subsystem(
#                 network, state, subset, backward_tpm=False, **kwargs
#             )
#             yield (cause_subsystem, effect_subsystem)
#         except pyphi.exceptions.StateUnreachableError:
#             pass


# def all_complexes(network, state, subsystem_indices=None, **kwargs):
#     """Yield SIAs for all subsystems of the network."""
#     if subsystem_indices is None:
#         subsystem_indices = network.node_indices
#     for cause_subsystem, effect_subsystem in reachable_subsystems(
#         network, subsystem_indices, state
#     ):
#         yield pyphi.backwards.sia(cause_subsystem, effect_subsystem, **kwargs)


# def irreducible_complexes(network, state, **kwargs):
#     """Yield SIAs for irreducible subsystems of the network."""
#     yield from filter(None, all_complexes(network, state, **kwargs))


# def maximal_complex(network, state, **kwargs):
#     return max(
#         irreducible_complexes(network, state, **kwargs),
#         default=pyphi.new_big_phi.NullSystemIrreducibilityAnalysis,
#     )


def condensed(network, state, **kwargs):
    """Return a list of maximal non-overlapping complexes."""
    result = []
    covered_nodes = set()

    for c in reversed(
        sorted(pyphi.new_big_phi.irreducible_complexes(network, state, **kwargs))
    ):
        if not any(n in covered_nodes for n in c.node_indices):
            result.append(c)
            covered_nodes = covered_nodes | set(c.node_indices)

    return result


def get_maximal_complexes_by_state(network, states=None):
    """Get the maximal non-overlapping complexes for every state of the network."""
    maximal_complexes_by_state = {}
    if states is None:
        states = pyphi.utils.all_states(network.size)
    for state in states:
        maximal_complexes_by_state[state] = condensed(network, state)
    return maximal_complexes_by_state


def print_maximal_complexes_by_state(network, mcbs):
    for state in pyphi.utils.all_states(network.size):
        print(f"Network state: {state}")
        for sia in mcbs[state]:
            print(sia)


# def get_subsystem_sia(network, network_state, subsystem_indices=None, **kwargs):
#     cause_subsystem = pyphi.Subsystem(
#         network, network_state, subsystem_indices, backward_tpm=True, **kwargs
#     )
#     effect_subsystem = pyphi.Subsystem(
#         network, network_state, subsystem_indices, backward_tpm=False, **kwargs
#     )
#     return pyphi.backwards.sia(cause_subsystem, effect_subsystem, **kwargs)


# def get_phi_structure(network, network_state, sia, **kwargs):
#     cause_subsystem = pyphi.Subsystem(
#         network, network_state, sia.node_indices, backward_tpm=True, **kwargs
#     )
#     effect_subsystem = pyphi.Subsystem(
#         network, network_state, sia.node_indices, backward_tpm=False, **kwargs
#     )
#     candidate_distinctions = pyphi.backwards.compute_combined_ces(
#         cause_subsystem, effect_subsystem
#     )
#     distinctions = candidate_distinctions.resolve_congruence(sia.system_state)
#     relations = pyphi.relations.relations(distinctions)
#     return pyphi.new_big_phi.phi_structure(
#         subsystem=effect_subsystem,
#         sia=sia,
#         distinctions=distinctions,
#         relations=relations,
#     )


def get_phi_structures_by_state(network, mcbs):
    return {
        state: [
            pyphi.new_big_phi.phi_structure(
                pyphi.Subsystem(network, state, nodes=sia.node_indices), sia
            )
            for sia in mcbs[state]
        ]
        for state in mcbs
    }


def stringify_state(state):
    return "".join(str(x) for x in state)


def stringify_complex(sia):
    return "".join([sia.node_labels[i] for i in sia.node_indices])


def plot_phi_structures_by_state(network, mcbs, psbs, savedir="", **kwargs):
    for network_state in pyphi.utils.all_states(network.size):
        state_string = stringify_state(network_state)
        print(f"Network state: {state_string}")
        for sia, phi_structure in zip(mcbs[network_state], psbs[network_state]):
            complex_string = stringify_complex(sia)
            print(f"Complex: {complex_string}")
            fig = pyphi.visualize.phi_structure.plot_phi_structure(
                phi_structure=phi_structure,
                state=network_state,
                node_labels=sia.node_labels,
                node_indices=sia.node_indices,
                **kwargs,
            )
            fig.write_html(Path(savedir) / f"{state_string}_{complex_string}.html")


def summarize_all_states(network, mcbs, psbs):
    for state in pyphi.utils.all_states(network.size):
        print(f"Network state: {state}")
        complexes = [
            "".join([sia.node_labels[i] for i in sia.node_indices])
            for sia in mcbs[state]
        ]
        print(f"Complexes: {complexes}")
        print(f"    φ_s: {[f'{sia.phi:.2f}' for sia in mcbs[state]]}")
        print(f"    Φ: {[f'{ps.big_phi:.2f}' for ps in psbs[state]]}")
        distinctions = [
            [[network.node_labels[i] for i in m] for m in ps.distinctions.mechanisms]
            for ps in psbs[state]
        ]
        print(f"    Distinctions: {distinctions:}")
        print(f"    #(relations): {[len(ps.relations) for ps in psbs[state]]}")


def get_truth_table(network):
    current_states = np.array(list(pyphi.utils.all_states(network.size)))
    sbn_2d = pyphi.convert.to_2d(network.tpm)
    assert ((sbn_2d == 1.0) | (sbn_2d == 0.0)).all(), "System is not deterministic"
    # Because the system is deterministic, it is safe to treat the rows of its TPM as a list of future states
    future_states = sbn_2d.astype(int)
    return current_states, future_states


def print_truth_table(network):
    current_states, future_states = get_truth_table(network)
    nodes = str.join(" ", network.node_labels)
    header = f"[{nodes}] : [{nodes}]"
    print(header)
    print("-" * len(header))
    for current_state, future_state in zip(current_states, future_states):
        print(f"{current_state} : {future_state}")


def plot_truth_table(network, figsize=None):
    current_states, future_states = get_truth_table(network)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.pcolormesh(current_states, edgecolors="k", linewidth=0.5, cmap="Greys")
    ax2.pcolormesh(future_states, edgecolors="k", linewidth=0.5, cmap="Greys")
    for ax in (ax1, ax2):
        ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=True)
        ax.set_xticks(
            np.arange(len(network.node_labels)) + 0.5,
            labels=network.node_labels,
            fontsize=12,
        )
        ax.set_yticks([])
        ax1.set_title("t")
        ax2.set_title("t + 1")
        ax.invert_yaxis()
    return fig, (ax1, ax2)


def plot_sbs_tpm(network, use_node_labels=True, height=None):
    def _italicize(text):
        return "$\it{" + "".join(text) + "}$"

    sbs = pyphi.convert.state_by_node2state_by_state(network.tpm)

    states_labels = list(pyphi.utils.all_states(network.size))
    if use_node_labels:
        state_labels = [
            pyphi.visualize.phi_structure.text.Labeler(
                state, network.node_labels, postprocessor=_italicize
            ).nodes(network.node_indices)
            for state in states_labels
        ]

    figsize = None if height is None else (height, height)
    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(sbs, edgecolors="k", linewidth=0.5, cmap="Greys", vmin=0, vmax=1)
    ax.tick_params(
        top=False,
        labeltop=use_node_labels,
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=use_node_labels,
    )
    if use_node_labels:
        ax.set_xticks(
            np.arange(len(state_labels)) + 0.5, labels=state_labels, rotation=90
        )
        ax.set_yticks(np.arange(len(state_labels)) + 0.5, labels=state_labels)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return fig, ax


def _get_cycle(state, current_states, next_states):
    """Get the cycle starting from a given state."""
    cycle = []
    while not (tuple(state) in cycle):
        cycle.append(tuple(state))
        state = next_states[current_states.index(state)]
    return cycle


def get_cycles(network, compact=True):
    """Return a compact representation of cycles, including those starting from unreachable states.
    Because cycles can re-enter themselves at any point, you could get a very different looking list if you
    traversed initial states in a different order. To avoid this, set compact=False"""
    # TODO: Color non-compact representation by cycle identity, using rich
    current_states, next_states = get_truth_table(network)
    current_states = current_states.tolist()
    next_states = next_states.tolist()
    visited = []
    cycles = []
    for state in current_states:
        if compact and (tuple(state) in visited):
            continue
        cycles.append(_get_cycle(state, current_states, next_states))
        visited += cycles[-1]
    return cycles


def get_pqrs_micro_example():
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
    node_labels = ("P", "Q", "R", "S")
    network = pyphi.Network(tpm, node_labels=node_labels)
    state = (0, 1, 0, 1)
    return network, state


def get_wxyz_micro_example():
    tpm = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
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
    node_labels = ("W", "X", "Y", "Z")
    network = pyphi.Network(tpm, node_labels=node_labels)
    state = (1, 1, 0, 1)
    return network, state
