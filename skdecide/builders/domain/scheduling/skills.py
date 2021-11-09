# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List, Set

__all__ = ["WithResourceSkills", "WithoutResourceSkills"]


class WithResourceSkills:
    """A domain must inherit this class if its resources (either resource types or resource units)
    have different set of skills."""

    def get_skills_names(self) -> Set[str]:
        """Return a list of all skill names as a list of str. Skill names are defined in the 2 dictionaries returned
        by the get_all_resources_skills and get_all_tasks_skills functions."""
        all_names = set()
        skill_dict = self.get_all_resources_skills()
        for key1 in skill_dict.keys():
            for key2 in skill_dict[key1].keys():
                all_names.add(key2)

        skill_dict = self.get_all_tasks_skills()
        for key1 in skill_dict.keys():
            for mode in skill_dict[key1].keys():
                for key2 in skill_dict[key1][mode].keys():
                    all_names.add(key2)
        return all_names

    def get_all_resources_skills(self) -> Dict[str, Dict[str, Any]]:
        """Return a nested dictionary where the first key is the name of a resource type or resource unit
        and the second key is the name of a skill. The value defines the details of the skill.
         E.g. {unit: {skill: (detail of skill)}}"""
        return self._get_all_resources_skills()

    def _get_all_resources_skills(self) -> Dict[str, Dict[str, Any]]:
        """Return a nested dictionary where the first key is the name of a resource type or resource unit
        and the second key is the name of a skill. The value defines the details of the skill.
         E.g. {unit: {skill: (detail of skill)}}"""
        raise NotImplementedError

    def get_skills_of_resource(self, resource: str) -> Dict[str, Any]:
        """Return the skills of a given resource"""
        return self.get_all_resources_skills()[resource]

    def get_all_tasks_skills(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """Return a nested dictionary where the first key is the name of a task
        and the second key is the name of a skill. The value defines the details of the skill.
         E.g. {task: {skill: (detail of skill)}}"""
        return self._get_all_tasks_skills()

    def _get_all_tasks_skills(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """Return a nested dictionary where the first key is the name of a task
        and the second key is the name of a skill. The value defines the details of the skill.
         E.g. {task: {skill: (detail of skill)}}"""
        raise NotImplementedError

    def get_skills_of_task(self, task: int, mode: int) -> Dict[str, Any]:
        """Return the skill requirements for a given task"""
        return {
            s: self.get_all_tasks_skills()[task][mode][s]
            for s in self.get_all_tasks_skills()[task][mode]
            if self.get_all_tasks_skills()[task][mode][s] > 0
        }

    def find_one_ressource_to_do_one_task(self, task: int, mode: int) -> List[str]:
        """
        For the common case when it is possible to do the task by one resource unit.
        For general case, it might just return no possible ressource unit.
        """
        skill_of_task = self.get_skills_of_task(task, mode)
        resources = []
        if len(skill_of_task) == 0:
            return [None]
        for resource in self.get_all_resources_skills():
            if all(
                self.get_skills_of_resource(resource=resource).get(s, 0)
                >= skill_of_task[s]
                for s in skill_of_task
            ):
                resources += [resource]
        # print("Ressources ", resources, " can do the task")
        return resources

    def check_if_skills_are_fulfilled(
        self, task: int, mode: int, resource_used: Dict[str, int]
    ):
        skill_of_task = self.get_skills_of_task(task, mode)
        if len(skill_of_task) == 0:
            return True  # No need of skills here.
        skills = {s: 0 for s in skill_of_task}
        for r in resource_used:
            skill_of_ressource = self.get_skills_of_resource(resource=r)
            for s in skill_of_ressource:
                if s in skills:
                    skills[s] += skill_of_ressource[s]
        # print("Ressource used : ", skills)
        # print("Skills required", skill_of_task)
        return all(skills[s] >= skill_of_task[s] for s in skill_of_task)


class WithoutResourceSkills(WithResourceSkills):
    """A domain must inherit this class if no resources skills have to be considered."""

    def _get_all_resources_skills(self) -> Dict[str, Dict[str, Any]]:
        """Return a nested dictionary where the first key is the name of a resource type or resource unit
        and the second key is the name of a skill. The value defines the details of the skill.
         E.g. {unit: {skill: (detail of skill)}}"""
        return {}

    def get_skills_of_resource(self, resource: str) -> Dict[str, Any]:
        """Return the skills of a given resource"""
        return {}

    def _get_all_tasks_skills(self) -> Dict[int, Dict[str, Any]]:
        """Return a nested dictionary where the first key is the name of a task
        and the second key is the name of a skill. The value defines the details of the skill.
         E.g. {task: {skill: (detail of skill)}}"""
        return {}

    def get_skills_of_task(self, task: int, mode: int) -> Dict[str, Any]:
        """Return the skill requirements for a given task"""
        return {}

    def check_if_skills_are_fulfilled(
        self, task: int, mode: int, resource_used: Dict[str, int]
    ):
        return True
