from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from skdecide import D, autocastable
from skdecide.core import Mask

if TYPE_CHECKING:
    # avoid circular import
    from skdecide import Domain


__all__ = ["ApplicableActions", "Maskable"]


class ApplicableActions:
    """A solver must inherit this class if he can use information about applicable action.

    This characteristic will be checked during rollout so that `retrieve_applicable_actions()` will be called before
    each call to `step()`. For instance, this is the case for solvers using action masks (see `Maskable`).

    """

    def using_applicable_actions(self):
        """Tell if the solver is able to use applicable actions information.

        For instance, action masking could be possible only if
        considered domain action space is enumerable for each agent.

        The default implementation returns always True.

        """
        return True

    def retrieve_applicable_actions(self, domain: Domain) -> None:
        """Retrieve applicable actions and use it for future call to `self.step()`.

        To be called during rollout to get the actual applicable actions from the actual domain used in rollout.

        """
        raise NotImplementedError


class Maskable(ApplicableActions):
    """A solver must inherit this class if he can use action masks to sample actions.

    For instance, it can be the case for wrappers around RL solvers like `sb3_contrib.MaskablePPO` or `ray.rllib` with
    custom model making use of action masking.

    An action mask is a format for specifying applicable actions when the action space is enumerable and finite. It is
    an array with 0's (for non-applicable actions) and 1's (for applicable actions). See `Events.get_action_mask()` for
    more information.

    """

    _action_mask: Optional[D.T_agent[Mask]] = None

    def retrieve_applicable_actions(self, domain: Domain) -> None:
        """Retrieve applicable actions and use it for future call to `self.step()`.

        To be called during rollout to get the actual applicable actions from the actual domain used in rollout.
        Transform applicable actions into an action_mask to be use when sampling action.

        """
        self.set_action_mask(domain.get_action_mask())

    @autocastable
    def set_action_mask(self, action_mask: Optional[D.T_agent[Mask]]) -> None:
        """Set the action mask.

        To be called during rollout before `self.sample_action()`, assuming that
        `self.sample_action()` knows what to do with it.

        Autocastable so that it can use action_mask from original domain during rollout.

        """
        self._set_action_mask(action_mask=action_mask)

    def _set_action_mask(self, action_mask: Optional[D.T_agent[Mask]]) -> None:
        """Set the action mask.

        To be called during rollout before `self.sample_action()`, assuming that
        `self.sample_action()` knows what to do with it.


        """

        self._action_mask = action_mask

    def get_action_mask(self) -> Optional[D.T_agent[Mask]]:
        """Retrieve stored action masks.

        To be used by `self.sample_action()`.
        Returns None if `self.set_action_mask()` was not called.

        """
        return self._action_mask
