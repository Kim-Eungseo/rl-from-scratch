import random
from collections import defaultdict
from .tabular_value_function import TabularValueFunction
from .tabular_policy import TabularPolicy


class TDLambda:
    """
    TD(λ) 알고리즘 구현 (Tabular with Replacing Traces)
    
    Algorithm: TD(λ) with Replacing Traces
    각 transition (X, R, Y)마다:
    1. TD error 계산: δ = R + γ·V[Y] - V[X]
    2. 모든 상태 x에 대해:
       - z[x] ← γ·λ·z[x]  (decay)
       - if X = x: z[x] ← 1  (replacing trace)
       - V[x] ← V[x] + α·δ·z[x]  (value update)
    """

    def __init__(self, env, alpha=0.1, epsilon=0.1, gamma=0.9, lambda_=0.8):
        """
        Args:
            env: 환경 (GridWorld 등)
            alpha: 학습률 (learning rate)
            epsilon: ε-greedy의 epsilon 값
            gamma: 할인율 (discount factor) γ
            lambda_: trace decay parameter λ (0 ≤ λ ≤ 1)
                    λ=0: TD(0) - one-step TD
                    λ=1: Monte Carlo와 유사
        """
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # Value function: V(s)
        self.value_function = TabularValueFunction(default_value=0.0)
        
        # Eligibility traces: z(s)
        self.traces = defaultdict(float)
        
        # Policy: greedy policy based on value function
        self.policy = TabularPolicy(default_action=None)

    def epsilon_greedy_action(self, state, actions):
        """
        ε-greedy 정책으로 액션 선택
        
        Args:
            state: 현재 상태
            actions: 가능한 액션 리스트
        
        Returns:
            선택된 액션
        """
        if not actions:
            return None
        
        # ε 확률로 랜덤 액션 선택 (exploration)
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        # 1-ε 확률로 greedy 액션 선택 (exploitation)
        # Value function 기반으로 최선의 액션 선택
        best_action = None
        best_value = float('-inf')
        
        for action in actions:
            # 각 액션에 대해 예상 다음 상태의 value 확인
            next_states = self.env.get_transitions(state, action)
            expected_value = 0.0
            for next_state, prob in next_states:
                reward = self.env.get_reward(state, action, next_state)
                expected_value += prob * (reward + self.gamma * self.value_function.get_value(next_state))
            
            if expected_value > best_value:
                best_value = expected_value
                best_action = action
        
        return best_action if best_action else random.choice(actions)

    def td_lambda_update(self, X, R, Y):
        """
        Algorithm 3: TD(λ) with Replacing Traces
        
        This function must be called after each transition.
        
        Args:
            X: 이전 상태 (last state)
            R: 즉시 보상 (immediate reward)
            Y: 다음 상태 (next state)
        
        Algorithm 3:
        1: δ ← R + γ · V[Y] − V[X]
        2: for all x ∈ X do
        3:     z[x] ← γ · λ · z[x]
        4:     if X = x then
        5:         z[x] ← 1
        6:     end if
        7:     V[x] ← V[x] + α · δ · z[x]
        8: end for
        9: return (V, z)
        """
        # Step 1: δ ← R + γ · V[Y] − V[X]
        V_X = self.value_function.get_value(X)
        V_Y = self.value_function.get_value(Y)
        delta = R + self.gamma * V_Y - V_X
        
        # Step 2: for all x ∈ X do
        # 효율성을 위해 traces가 0이 아닌 상태들과 현재 상태 X만 업데이트
        states_to_update = list(self.traces.keys()) + [X]
        states_to_update = list(set(states_to_update))  # 중복 제거
        
        for x in states_to_update:
            # Step 3: z[x] ← γ · λ · z[x]
            self.traces[x] = self.gamma * self.lambda_ * self.traces[x]
            
            # Step 4-6: if X = x then z[x] ← 1
            if X == x:
                self.traces[x] = 1.0
            
            # Step 7: V[x] ← V[x] + α · δ · z[x]
            current_value = self.value_function.get_value(x)
            new_value = current_value + self.alpha * delta * self.traces[x]
            self.value_function.update(x, new_value)
            
            # 메모리 효율을 위해 매우 작은 trace는 제거
            if abs(self.traces[x]) < 1e-8:
                del self.traces[x]
        
        # Step 9: return (V, z) - 암묵적으로 self에 저장됨

    def reset_traces(self):
        """에피소드 시작 시 eligibility traces 초기화"""
        self.traces.clear()

    def run_episode(self):
        """
        한 에피소드 실행 및 TD(λ) 업데이트
        
        Returns:
            total_reward: 에피소드의 총 보상
            steps: 에피소드의 스텝 수
        """
        # Eligibility traces 초기화
        self.reset_traces()
        
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        
        max_steps = 1000  # 무한 루프 방지
        for _ in range(max_steps):
            actions = self.env.get_actions(state)
            if not actions:  # 터미널 상태
                break
            
            # ε-greedy로 액션 선택
            action = self.epsilon_greedy_action(state, actions)
            
            # 환경에서 한 스텝 실행
            next_state, reward, done = self.env.step(action)
            
            # TD(λ) 업데이트
            self.td_lambda_update(state, reward, next_state)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
            
            state = next_state
        
        return total_reward, steps

    def extract_policy(self):
        """
        현재 value function으로부터 greedy policy 추출
        
        π(s) = argmax_a E[R + γ·V(s') | s, a]
        """
        policy = TabularPolicy(default_action=None)
        
        for state in self.env.get_states():
            actions = self.env.get_actions(state)
            if not actions:
                continue
            
            best_action = None
            best_value = float('-inf')
            
            for action in actions:
                # 각 액션에 대해 예상 value 계산
                expected_value = 0.0
                for next_state, prob in self.env.get_transitions(state, action):
                    reward = self.env.get_reward(state, action, next_state)
                    expected_value += prob * (reward + self.gamma * self.value_function.get_value(next_state))
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action
            
            if best_action:
                policy.update(state, best_action)
        
        return policy

    def train(self, num_episodes=1000, verbose=False):
        """
        TD(λ) 학습 메인 루프
        
        Args:
            num_episodes: 학습할 에피소드 수
            verbose: True면 진행상황 출력
        
        Returns:
            policy: 학습된 정책
            episode_rewards: 에피소드별 보상 리스트
        """
        episode_rewards = []
        
        for episode_num in range(num_episodes):
            total_reward, steps = self.run_episode()
            episode_rewards.append(total_reward)
            
            if verbose and (episode_num + 1) % 100 == 0:
                avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
                print(f"Episode {episode_num + 1}/{num_episodes} - "
                      f"Avg Reward (last 100): {avg_reward:.3f}")
        
        # 최종 정책 추출
        policy = self.extract_policy()
        self.policy = policy
        
        return policy, episode_rewards

    def get_value_function(self):
        """학습된 value function 반환"""
        return self.value_function
