import random
from .tabular_value_function import TabularValueFunction
from .tabular_policy import TabularPolicy


class TD0:
    """
    TD(0) 알고리즘 구현 (Tabular One-Step Temporal Difference Learning)
    
    Algorithm: TD(0)
    각 transition (X, R, Y)마다:
    1. δ ← R + γ·V[Y] - V[X]  (TD error)
    2. V[X] ← V[X] + α·δ      (value update)
    
    가장 기본적인 TD learning 알고리즘으로, 한 스텝만 보고 즉시 업데이트합니다.
    """

    def __init__(self, env, alpha=0.1, epsilon=0.1, gamma=0.9):
        """
        Args:
            env: 환경 (GridWorld 등)
            alpha: 학습률 (learning rate) α
            epsilon: ε-greedy의 epsilon 값
            gamma: 할인율 (discount factor) γ
        """
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Value function: V(s)
        self.value_function = TabularValueFunction(default_value=0.0)
        
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

    def td0_update(self, X, R, Y):
        """
        TD(0) 업데이트 함수
        
        각 transition (X, R, Y) 후 호출됨
        
        Args:
            X: 이전 상태 (last state)
            R: 즉시 보상 (immediate reward)
            Y: 다음 상태 (next state)
        """
        # 1. TD error 계산: δ = R + γ·V[Y] - V[X]
        V_X = self.value_function.get_value(X)
        V_Y = self.value_function.get_value(Y)
        delta = R + self.gamma * V_Y - V_X
        
        # 2. V[X] ← V[X] + α·δ (현재 상태만 업데이트)
        new_value = V_X + self.alpha * delta
        self.value_function.update(X, new_value)

    def run_episode(self):
        """
        한 에피소드 실행 및 TD(0) 업데이트
        
        Returns:
            total_reward: 에피소드의 총 보상
            steps: 에피소드의 스텝 수
        """
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
            
            # TD(0) 업데이트
            self.td0_update(state, reward, next_state)
            
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
        TD(0) 학습 메인 루프
        
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
