from typing import Optional

import torch as th


def extract_applicable_action_components(
    action_components: th.Tensor, applicable_actions: th.Tensor, i_action_component: int
) -> th.Tensor:
    """Extract the applicable action components for a given index given previous action components

    NB: output has not a predictable shape

    Args:
        action_components: partial actions up to i_action_component -1, size (N,) with N>=i_action_component
        applicable_actions: applicable full actions from which we want to extract applicable components, size (M,A),
            where A is the action shape length (and A > i_action_component),
            where M is the number of applicable actions
        i_action_component: index of action component for which we are extracting applicable components

    Returns:
        applicable_action_components: size (m,) with applicable components indices (m<=M)

    """
    if i_action_component == 0:
        applicable_action_components = applicable_actions[:, i_action_component]
    else:
        applicable_action_components = applicable_actions[
            # previous components are the same
            (
                applicable_actions[:, :i_action_component]
                == action_components[None, :i_action_component]
            ).all(dim=1),
            # current component of the applicable action
            i_action_component,
        ]  # .unique()  # could take unique() there => what is more efficient?
    # remove -1 <=> no component
    return applicable_action_components[applicable_action_components >= 0]


def extract_action_component_mask(
    action_components, applicable_actions, i_action_component, action_component_dim
):
    applicable_action_components = extract_applicable_action_components(
        action_components=action_components,
        applicable_actions=applicable_actions,
        i_action_component=i_action_component,
    )
    # init mask (action_component_dim, )
    action_component_mask = th.zeros(
        (action_component_dim,), dtype=th.int8, device=action_components.device
    )
    # update with 1 for each applicable component
    action_component_mask[applicable_action_components] = 1
    return action_component_mask


def batched_extract_action_component_mask(
    action_components: th.Tensor,
    applicable_actions: list[th.Tensor],
    i_action_component: int,
    action_component_dim: int,
    nonzeromask_samples_mask: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Extract batchwise a mask for the given component index according to previous action components

    NB: Batch dim for action_components *and* applicable_actions

    Args:
        action_components: partial actions up to i_action_component -1, size (B,N) with N>=i_action_component
        applicable_actions: applicable full actions from which we want to extract applicable components, size B*(M,A),
            where B is the batch size
            where A is the action shape length (and A > i_action_component),
            where M is the number of applicable actions
        i_action_component: index of action component for which we are extracting a mask
        action_component_dim: action component max value
        nonzeromask_samples_mask: boolean tensor (B,). True for each sample needing a mask computation
            (else mask is only 0's). If None, we compute for each sample.

    Returns:
        action_component_mask: size (B, action_component_dim)

    """
    # init mask (B, action_component_dim, )
    action_component_mask = th.zeros(
        (
            action_components.shape[0],
            action_component_dim,
        ),
        dtype=th.int8,
        device=action_components.device,
    )
    # loop over samples
    for i_sample, applicable_actions_sample in enumerate(applicable_actions):
        if nonzeromask_samples_mask is None or nonzeromask_samples_mask[i_sample]:
            applicable_action_components_sample = extract_applicable_action_components(
                action_components=action_components[i_sample],
                applicable_actions=applicable_actions_sample,
                i_action_component=i_action_component,
            )
            # update mask with 1 for each applicable component
            action_component_mask[i_sample, applicable_action_components_sample] = 1
    return action_component_mask


def extract_applicable_action_components_v2(
    action_components: th.Tensor, applicable_actions: th.Tensor, i_action_component: int
) -> th.Tensor:
    """Extract the applicable action components for a given index given previous action components

    NB: output has a predictable shape, corresponding to applicable_actions length

    Args:
        action_components: partial actions up to i_action_component -1, size (N,) with N>=i_action_component
        applicable_actions: applicable full actions from which we want to extract applicable components, size (M,A),
            where A is the action shape length (and A > i_action_component),
            where M is the number of applicable actions
        i_action_component: index of action component for which we are extracting applicable components

    Returns:
        applicable_action_components: size (M,) with applicable components indices (-1 meaning "no component")

    """
    applicable_action_components = th.where(
        # previous components are the same
        (
            applicable_actions[:, :i_action_component]
            == action_components[None, :i_action_component]
        ).all(dim=1),
        # current component of the applicable action
        applicable_actions[:, i_action_component],
        # invalid component (<=> fake last component)
        -1,
    )
    return applicable_action_components


batched_extract_applicable_action_components_v2 = th.vmap(
    extract_applicable_action_components_v2, in_dims=(0, None, None)
)
"""Batched version of compute_extract_action_components

NB: Batch dim for action_components *but not* applicable_actions

Args:
    action_components: partial actions up to i_action_component -1, size (B,N) with N>=i_action_component
    applicable_actions: applicable full actions from which we want to extract applicable components, size (M,A),
        where A is the action shape length (and A > i_action_component),
        where M is the number of applicable actions
    i_action_component: index of action component for which we are extracting applicable components

Returns:
    applicable_action_components: size (B, M) with applicable components indices (-1 meaning "no component")

"""


def batched_extract_action_component_mask_v2(
    action_components, applicable_actions, i_action_component, action_component_dim
):
    """Extract batchwise a mask for the given component index according to previous action components

    NB: Batch dim for action_components *but not* applicable_actions

    Args:
        action_components: partial actions up to i_action_component -1, size (B,N) with N>=i_action_component
        applicable_actions: applicable full actions from which we want to extract applicable components, size (M,A),
            where A is the action shape length (and A > i_action_component),
            where M is the number of applicable actions
        i_action_component: index of action component for which we are extracting a mask
        action_component_dim: action component max value

    Returns:
        action_component_mask: size (B, action_component_dim)

    """
    # get applicable action components for each batch, according to given applicable actions
    applicable_action_components = batched_extract_applicable_action_components_v2(
        action_components, applicable_actions, i_action_component
    )
    # init mask (batch * action_component_dim +1) => last component <=> no component
    action_component_mask = th.zeros(
        (len(action_components), action_component_dim + 1), dtype=th.int8
    )
    # update with 1 for each applicable component
    action_component_mask[
        th.arange(len(action_components), dtype=int)[:, None],
        applicable_action_components,
    ] = 1
    # remove last component (meant "no component")
    return action_component_mask[:, :-1]


def extract_action_component_mask_v2(
    action_components, applicable_actions, i_action_component, action_component_dim
):
    """Extract a mask for the given component index according to previous action components

    Args:
        action_components: partial actions up to i_action_component -1, size (N,) with N>=i_action_component
        applicable_actions: applicable full actions from which we want to extract applicable components, size (M,A),
            where A is the action shape length (and A > i_action_component),
            where M is the number of applicable actions
        i_action_component: index of action component for which we are extracting a mask
        action_component_dim: action component max value

    Returns:
        action_component_mask: size (action_component_dim, )

    """
    # get applicable action components for each batch, according to given applicable actions
    applicable_action_components = extract_applicable_action_components_v2(
        action_components=action_components,
        applicable_actions=applicable_actions,
        i_action_component=i_action_component,
    )
    # init mask (action_component_dim +1, ) => last component <=> no component
    action_component_mask = th.zeros((action_component_dim + 1,), dtype=th.int8)
    # update with 1 for each applicable component
    action_component_mask[applicable_action_components] = 1
    # remove last component (meant "no component")
    return action_component_mask[:-1]
