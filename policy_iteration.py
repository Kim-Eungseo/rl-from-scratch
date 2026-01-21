from tabular_policy import TabularPolicy
from tabular_value_function import TabularValueFunction
from qtable import QTable


class PolicyIteration:
    """
    Policy Iteration 알고리즘 구현
    
    Algorithm: Policy Iteration
    1: Randomly initialize policy π₀
    2: for each k = 0, 1, 2, ..., ∞ do
    3:     Q^πk ← Policy evaluation with πk
    4:     Policy improvement: πk+1 = G(Q^πk)
    5: end for
    """

    def __init__(self, mdp, policy):
        """
        Args:
            mdp: MDP 환경
            policy: 초기 정책 π₀ (Step 1: Randomly initialize policy)
        """
        self.mdp = mdp
        self.policy = policy

    def policy_evaluation(self, policy, values, theta=0.001):
        """
        Step 3: Q^πk ← Policy evaluation with πk
        현재 정책 πk에 대한 가치 함수를 계산합니다.
        """
        while True:
            delta = 0.0
            new_values = TabularValueFunction()
            for state in self.mdp.get_states():
                # Calculate the value of V(s)
                actions = self.mdp.get_actions(state)
                old_value = values.get_value(state)
                new_value = values.get_q_value(
                    self.mdp, state, policy.select_action(state, actions)
                )
                values.add(state, new_value)
                delta = max(delta, abs(old_value - new_value))

            # terminate if the value function has converged
            if delta < theta:
                break

        return values

    def policy_iteration(self, max_iterations=100, theta=0.001):
        """
        Policy Iteration 메인 루프
        Returns: 수렴까지 실행된 반복 횟수
        """
        values = TabularValueFunction()

        # Step 2: for each k = 0, 1, 2, ..., ∞ do
        for i in range(1, max_iterations + 1):
            policy_changed = False

            # Step 3: Q^πk ← Policy evaluation with πk
            values = self.policy_evaluation(self.policy, values, theta)

            # Step 4: Policy improvement: πk+1 = G(Q^πk)
            for state in self.mdp.get_states():
                actions = self.mdp.get_actions(state)
                old_action = self.policy.select_action(state, actions)

                # 모든 액션에 대해 Q(s,a) 계산
                q_values = QTable(alpha=1.0)
                for action in self.mdp.get_actions(state):
                    new_value = values.get_q_value(self.mdp, state, action)
                    q_values.update(state, action, new_value)

                # G(Q^πk): greedy policy - πk+1(s) = argmax_a Q^πk(s,a)
                new_action = q_values.get_argmax_q(state, self.mdp.get_actions(state))
                self.policy.update(state, new_action)

                policy_changed = (
                    True if new_action is not old_action else policy_changed
                )

            # 정책이 변하지 않으면 수렴 (Step 5: end for)
            if not policy_changed:
                return i

        return max_iterations