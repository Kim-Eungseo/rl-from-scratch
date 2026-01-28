import random
from collections import defaultdict
from .qtable import QTable
from .tabular_policy import TabularPolicy


class MonteCarlo:
    """
    Monte Carlo Control 알고리즘 구현
    
    Algorithm: Monte Carlo Control
    1. Policy Improvement: 현재 Q에 대해 greedy하게 policy 개선
       π(s) = argmax_a Q(s, a)
    
    2. Generate Episode: ε-greedy로 episode 생성하여 exploration/exploitation 균형
    
    3. Estimate Q: episode로부터 Q 추정
       q_π(s,a) = E[G_t | S_t=s, A_t=a]
       G_t = Σ(k=0 to T-t-1) γ^k * R_(t+k+1)
    """

    def __init__(self, env, epsilon=0.1, discount=0.9, first_visit=True):
        """
        Args:
            env: 환경 (GridWorld 등)
            epsilon: ε-greedy의 epsilon 값
            discount: 할인율 γ
            first_visit: True면 first-visit MC, False면 every-visit MC
        """
        self.env = env
        self.epsilon = epsilon
        self.discount = discount
        self.first_visit = first_visit
        
        # Q-table: Q(s, a) 값 저장
        self.qtable = QTable(alpha=1.0, default_value=0.0)
        
        # Returns: 각 (s, a)에 대한 return 리스트
        self.returns = defaultdict(list)
        
        # Policy: greedy policy
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
        return self.qtable.get_best_action(state, actions)

    def generate_episode(self):
        """
        2. Generate Episode: ε-greedy 정책으로 에피소드 생성
        
        Returns:
            episode: [(state, action, reward), ...] 리스트
        """
        episode = []
        state = self.env.reset()
        
        max_steps = 1000  # 무한 루프 방지
        for _ in range(max_steps):
            actions = self.env.get_actions(state)
            if not actions:  # 터미널 상태
                break
            
            action = self.epsilon_greedy_action(state, actions)
            next_state, reward, done = self.env.step(action)
            
            episode.append((state, action, reward))
            
            if done:
                break
            
            state = next_state
        
        return episode

    def calculate_returns(self, episode):
        """
        에피소드로부터 각 time step의 return G_t 계산
        
        G_t = R_(t+1) + γ*R_(t+2) + γ^2*R_(t+3) + ... + γ^(T-t-1)*R_T
        
        Args:
            episode: [(state, action, reward), ...] 리스트
        
        Returns:
            returns: [G_0, G_1, G_2, ...] 리스트
        """
        returns = []
        G = 0.0
        
        # 역순으로 계산 (T-1부터 0까지)
        for state, action, reward in reversed(episode):
            G = reward + self.discount * G
            returns.insert(0, G)
        
        return returns

    def update_q_values(self, episode):
        """
        3. Estimate Q: 에피소드로부터 Q 값 업데이트
        
        q_π(s,a) = average of returns following (s,a)
        """
        returns = self.calculate_returns(episode)
        visited_state_actions = set()
        
        for t, (state, action, reward) in enumerate(episode):
            state_action = (state, action)
            
            # First-visit MC: 처음 방문한 (s,a)만 업데이트
            if self.first_visit and state_action in visited_state_actions:
                continue
            
            visited_state_actions.add(state_action)
            
            # Return 저장
            G_t = returns[t]
            self.returns[state_action].append(G_t)
            
            # Q(s,a) = average of returns
            new_q_value = sum(self.returns[state_action]) / len(self.returns[state_action])
            self.qtable.update(state, action, new_q_value)

    def improve_policy(self):
        """
        1. Policy Improvement: Q에 대해 greedy하게 정책 개선
        
        π(s) = argmax_a Q(s, a)
        """
        for state in self.env.get_states():
            actions = self.env.get_actions(state)
            if actions:
                best_action = self.qtable.get_argmax_q(state, actions)
                self.policy.update(state, best_action)

    def train(self, num_episodes=1000, verbose=False):
        """
        Monte Carlo Control 학습 메인 루프
        
        Args:
            num_episodes: 학습할 에피소드 수
            verbose: True면 진행상황 출력
        
        Returns:
            policy: 학습된 정책
        """
        for episode_num in range(num_episodes):
            # 2. Generate Episode
            episode = self.generate_episode()
            
            # 3. Estimate Q
            self.update_q_values(episode)
            
            # 1. Policy Improvement
            self.improve_policy()
            
            if verbose and (episode_num + 1) % 100 == 0:
                print(f"Episode {episode_num + 1}/{num_episodes} completed")
        
        return self.policy

    def get_q_values(self):
        """학습된 Q-table 반환"""
        return self.qtable

    def get_value_function(self):
        """
        학습된 Q로부터 V(s) = max_a Q(s,a) 계산하여 반환
        """
        from .tabular_value_function import TabularValueFunction
        value_function = TabularValueFunction(default_value=0.0)
        
        for state in self.env.get_states():
            actions = self.env.get_actions(state)
            if actions:
                max_q = self.qtable.get_max_q(state, actions)
                value_function.update(state, max_q)
            else:
                value_function.update(state, 0.0)
        
        return value_function
