from collections import defaultdict
from .value_function import ValueFunction


class TabularValueFunction(ValueFunction):
    def __init__(self, default_value=0.0):
        self.value_table = defaultdict(lambda: default_value)

    def update(self, state, value):
        self.value_table[state] = value

    def add(self, state, value):
        self.value_table[state] = value

    def merge(self, value_table):
        for state, value in value_table.value_table.items():
            self.value_table[state] = value

    def get_value(self, state):
        return self.value_table[state]
