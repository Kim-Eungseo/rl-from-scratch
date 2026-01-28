from envs import GridWorld
from agents import TDLambda


def main():
    print("=" * 50)
    print("Grid World TD(λ) 테스트 (10x10)")
    print("=" * 50)

    # 10x10 Grid World 생성 - 계단 패턴
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9)],
        obstacles=[
            # 계단 형태
            (7, 1), (7, 2),
            (6, 3), (6, 4),
            (5, 5), (5, 6),
            (4, 7), (4, 8),
            # 추가 섬
            (2, 2), (8, 5)
        ],
        discount=0.95,
        start_state=(9, 0)  # 좌하단 시작
    )

    print("\n[Grid World 설정]")
    print(f"크기: {gridworld.width} x {gridworld.height}")
    print(f"시작 상태: {gridworld.start_state}")
    print(f"목표 상태: {gridworld.goal_states}")
    print(f"장애물: {gridworld.obstacles}")
    print(f"할인율: {gridworld.discount}")

    # TD(λ) 학습
    td_lambda = TDLambda(
        env=gridworld,
        alpha=0.1,      # 학습률
        epsilon=0.1,    # ε-greedy의 epsilon
        gamma=0.95,     # 할인율
        lambda_=0.8     # trace decay parameter (0 ≤ λ ≤ 1)
    )

    print("\n[TD(λ) 학습 시작]")
    print(f"에피소드 수: 1000")
    print(f"Learning rate (α): {td_lambda.alpha}")
    print(f"Epsilon (ε): {td_lambda.epsilon}")
    print(f"Lambda (λ): {td_lambda.lambda_}")
    
    # 학습 실행
    policy, episode_rewards = td_lambda.train(num_episodes=1000, verbose=True)

    print("\n[학습 완료]")
    
    # 학습된 정책 출력
    print("\n[학습된 정책]")
    gridworld.print_policy(policy)
    
    # 학습된 Value Function 출력
    value_function = td_lambda.get_value_function()
    gridworld.print_values(value_function)
    
    # 특정 상태들의 Value 확인
    print("[주요 상태별 Value]")
    test_states = [(0, 0), (0, 5), (5, 5), (9, 0), (9, 9)]
    for state in test_states:
        if state not in gridworld.obstacles:
            value = value_function.get_value(state)
            print(f"  V{state} = {value:.4f}")


def compare_lambda_values():
    """
    다양한 λ 값에 따른 성능 비교
    λ=0: TD(0) - one-step TD
    λ=1: Monte Carlo와 유사
    """
    print("\n" + "=" * 50)
    print("다양한 λ 값 비교 (10x10)")
    print("=" * 50)

    # 10x10 Grid World - 십자 패턴
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9)],
        obstacles=[
            # 십자 형태
            (3, 5), (4, 5), (5, 5), (6, 5), (7, 5),  # 세로
            (5, 3), (5, 4), (5, 6), (5, 7)  # 가로
        ],
        discount=0.95,
        start_state=(9, 0)
    )

    lambda_values = [0.0, 0.3, 0.5, 0.8, 0.9, 1.0]
    results = {}
    
    for lambda_val in lambda_values:
        print(f"\n[λ = {lambda_val:.1f} 학습 중...]")
        
        td = TDLambda(
            env=gridworld,
            alpha=0.1,
            epsilon=0.1,
            gamma=0.95,
            lambda_=lambda_val
        )
        
        policy, episode_rewards = td.train(num_episodes=1000, verbose=False)
        results[lambda_val] = {
            'policy': policy,
            'value_function': td.get_value_function(),
            'rewards': episode_rewards
        }
        
        # 최근 100 에피소드 평균 보상
        avg_reward = sum(episode_rewards[-100:]) / 100
        print(f"  평균 보상 (최근 100 에피소드): {avg_reward:.3f}")
    
    print("\n[학습 완료 - 각 λ 값별 정책 비교]")
    
    for lambda_val in lambda_values:
        print(f"\n--- λ = {lambda_val:.1f} ---")
        gridworld.print_policy(results[lambda_val]['policy'])


def test_td0_vs_mc():
    """
    TD(0) vs TD(1) 비교
    TD(0): one-step TD learning
    TD(1): Monte Carlo와 유사한 학습
    """
    print("\n" + "=" * 50)
    print("TD(0) vs TD(1) 비교 (10x10)")
    print("=" * 50)

    # 10x10 Grid World - 다이아몬드 패턴
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9)],
        obstacles=[
            # 다이아몬드 형태
            (5, 5),  # 중심
            (4, 4), (4, 6),
            (3, 5),
            (6, 4), (6, 6),
            (7, 5)
        ],
        discount=0.95,
        start_state=(9, 0)
    )

    # TD(0)
    print("\n[TD(0) 학습]")
    td0 = TDLambda(
        env=gridworld,
        alpha=0.1,
        epsilon=0.1,
        gamma=0.95,
        lambda_=0.0  # TD(0)
    )
    policy_td0, rewards_td0 = td0.train(num_episodes=1000, verbose=True)
    
    print("\n[TD(0) 학습된 정책]")
    gridworld.print_policy(policy_td0)
    gridworld.print_values(td0.get_value_function())

    # TD(1)
    print("\n[TD(1) 학습 - Monte Carlo와 유사]")
    td1 = TDLambda(
        env=gridworld,
        alpha=0.1,
        epsilon=0.1,
        gamma=0.95,
        lambda_=1.0  # TD(1) ≈ Monte Carlo
    )
    policy_td1, rewards_td1 = td1.train(num_episodes=1000, verbose=True)
    
    print("\n[TD(1) 학습된 정책]")
    gridworld.print_policy(policy_td1)
    gridworld.print_values(td1.get_value_function())


def test_different_alpha():
    """
    다양한 학습률(α) 비교
    """
    print("\n" + "=" * 50)
    print("다양한 학습률(α) 비교 (10x10)")
    print("=" * 50)

    # 10x10 Grid World - 화살표 패턴
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9)],
        obstacles=[
            # 화살표 형태 (위를 가리키는)
            (7, 5),  # 꼭지점
            (6, 4), (6, 5), (6, 6),
            (5, 3), (5, 4), (5, 6), (5, 7),
            (4, 5)
        ],
        discount=0.95,
        start_state=(9, 0)
    )

    alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5]
    
    for alpha_val in alpha_values:
        print(f"\n[α = {alpha_val:.2f} 학습 중...]")
        
        td = TDLambda(
            env=gridworld,
            alpha=alpha_val,
            epsilon=0.1,
            gamma=0.95,
            lambda_=0.8
        )
        
        policy, episode_rewards = td.train(num_episodes=1000, verbose=False)
        
        # 최근 100 에피소드 평균 보상
        avg_reward = sum(episode_rewards[-100:]) / 100
        print(f"  평균 보상 (최근 100 에피소드): {avg_reward:.3f}")


if __name__ == "__main__":
    main()
    compare_lambda_values()
    test_td0_vs_mc()
    test_different_alpha()
