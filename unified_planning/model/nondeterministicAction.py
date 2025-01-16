from typing import List, Tuple

import unified_planning as up
from unified_planning.model import InstantaneousAction
from unified_planning.shortcuts import TRUE, Effect, Fluent, FluentExp, Bool


class NondeterministicAction(InstantaneousAction):
    @property
    def effects(self):
        return self._effects

    def __init__(self, _name: str, effects: List[Tuple[Fluent, float, bool]], **kwargs: "up.model.types.Type"):
        super().__init__(_name, **kwargs)
        self.name = _name
        self._effects = [
            Effect(FluentExp(fluent), TRUE(), Bool(value)) for fluent, _, value in effects
        ]

    def generate_effects(self):
        """
        Returns all possible effects.
        """
        return [(effect.condition, effect) for effect in self._effects]

    @effects.setter
    def effects(self, value):
        self._effects = value

    def generate_branches(self):
        """
        Returns all possible branches obtained from this action.
        """
        branches = []
        for effect in self._effects:

            if effect.condition.is_true():
                fluent = effect.fluent
                value = effect.value.bool_constant_value()
                fluent_name = fluent.fluent().name

                branches.append((fluent_name, value))
            else:
                branches.append((effect.fluent.fluent().name, not effect.value.bool_constant_value()))
        return branches

    def generate_sequences(self, current_state, remaining_actions, path=[]):
        """
        Generates all possible sequences for the current action and for the remaining ones.
        """
        sequences = []

        # Check on preconditions satisfiability
        if all(current_state.get(pre.fluent().name, False) for pre in self.preconditions):
            branches = self.generate_branches()
            for fluent_name, value in branches:
                new_state = current_state.copy()
                new_state[fluent_name] = value

                new_path = path + [f"{self.name} ({fluent_name}={value})"]

                if remaining_actions:
                    for action in remaining_actions:
                        sequences.extend(action.generate_sequences(
                            new_state,
                            [a for a in remaining_actions if a != action],
                            new_path
                        ))

                if not remaining_actions:
                    sequences.append(new_path)
        else:
            sequences.append(path)

        return sequences
