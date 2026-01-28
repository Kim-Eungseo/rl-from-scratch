from collections import defaultdict


class QTable:
    def __init__(self, alpha=0.1, default_value=0.0):
        self.alpha = alpha
        self.default_value = default_value
        self.q_table = defaultdict(lambda: defaultdict(lambda: self.default_value))

    def get_q_value(self, state, action):
        return self.q_table[state][action]

    def update(self, state, action, value):
        """Q-값을 업데이트합니다. alpha=1.0이면 완전히 덮어씁니다."""
        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.alpha * (value - old_value)

    def get_max_q(self, state, actions):
        """주어진 상태에서 가능한 액션들 중 최대 Q-값을 반환합니다."""
        if not actions:
            return self.default_value
        return max(self.q_table[state][action] for action in actions)

    def get_best_action(self, state, actions):
        """주어진 상태에서 가장 높은 Q-값을 가진 액션을 반환합니다."""
        if not actions:
            return None
        return max(actions, key=lambda a: self.q_table[state][a])

    def get_argmax_q(self, state, actions):
        """get_best_action의 별칭 - argmax_a Q(s,a)를 반환합니다."""
        return self.get_best_action(state, actions)
