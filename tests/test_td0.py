from envs import GridWorld
from agents import TD0


def main():
    print("=" * 50)
    print("Grid World TD(0) 테스트 (10x10)")
    print("=" * 50)

    # 10x10 Grid World 생성 - 지그재그 패턴
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9)],
        obstacles=[
            # 지그재그 벽
            (2, 1), (2, 2), (2, 3),
            (4, 3), (4, 4), (4, 5),
            (6, 5), (6, 6), (6, 7),
            (8, 7), (8, 8),
            # 추가 장애물
            (3, 8), (5, 2)
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

    # TD(0) 학습
    td0 = TD0(
        env=gridworld,
        alpha=0.1,      # 학습률
        epsilon=0.1,    # ε-greedy의 epsilon
        gamma=0.95      # 할인율
    )

    print("\n[TD(0) 학습 시작]")
    print(f"에피소드 수: 1000")
    print(f"Learning rate (α): {td0.alpha}")
    print(f"Epsilon (ε): {td0.epsilon}")
    print("알고리즘: One-step Temporal Difference")
    
    # 학습 실행
    policy, episode_rewards = td0.train(num_episodes=1000, verbose=True)

    print("\n[학습 완료]")
    
    # 학습된 정책 출력
    print("\n[학습된 정책]")
    gridworld.print_policy(policy)
    
    # 학습된 Value Function 출력
    value_function = td0.get_value_function()
    gridworld.print_values(value_function)
    
    # 특정 상태들의 Value 확인
    print("[주요 상태별 Value]")
    test_states = [(0, 0), (0, 5), (5, 5), (9, 0), (9, 9)]
    for state in test_states:
        if state not in gridworld.obstacles:
            value = value_function.get_value(state)
            print(f"  V{state} = {value:.4f}")


def test_different_alpha():
    """
    다양한 학습률(α) 비교
    """
    print("\n" + "=" * 50)
    print("다양한 학습률(α) 비교 (10x10)")
    print("=" * 50)

    # 10x10 Grid World - 체스판 패턴
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9)],
        obstacles=[
            # 체스판 패턴
            (2, 2), (2, 4), (2, 6), (2, 8),
            (4, 1), (4, 3), (4, 5), (4, 7),
            (6, 2), (6, 4), (6, 6), (6, 8),
            (8, 1), (8, 3), (8, 5)
        ],
        discount=0.95,
        start_state=(9, 0)
    )

    alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5]
    
    for alpha_val in alpha_values:
        print(f"\n[α = {alpha_val:.2f} 학습 중...]")
        
        td0 = TD0(
            env=gridworld,
            alpha=alpha_val,
            epsilon=0.1,
            gamma=0.95
        )
        
        policy, episode_rewards = td0.train(num_episodes=1000, verbose=False)
        
        # 최근 100 에피소드 평균 보상
        avg_reward = sum(episode_rewards[-100:]) / 100
        print(f"  평균 보상 (최근 100 에피소드): {avg_reward:.3f}")


def test_different_epsilon():
    """
    다양한 epsilon 값 비교
    """
    print("\n" + "=" * 50)
    print("다양한 Epsilon(ε) 비교 (10x10)")
    print("=" * 50)

    # 10x10 Grid World - 원형 패턴
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9)],
        obstacles=[
            # 원형 근사
            (4, 4), (4, 5), (4, 6),
            (3, 3), (3, 7),
            (5, 3), (5, 7),
            (6, 4), (6, 5), (6, 6)
        ],
        discount=0.95,
        start_state=(9, 0)
    )

    epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    for eps_val in epsilon_values:
        print(f"\n[ε = {eps_val:.2f} 학습 중...]")
        
        td0 = TD0(
            env=gridworld,
            alpha=0.1,
            epsilon=eps_val,
            gamma=0.95
        )
        
        policy, episode_rewards = td0.train(num_episodes=1000, verbose=False)
        
        # 최근 100 에피소드 평균 보상
        avg_reward = sum(episode_rewards[-100:]) / 100
        print(f"  평균 보상 (최근 100 에피소드): {avg_reward:.3f}")


def test_small_grid():
    """
    작은 4x4 그리드에서 빠른 학습 테스트
    """
    print("\n" + "=" * 50)
    print("작은 Grid World (4x4) TD(0) 테스트")
    print("=" * 50)

    # 4x4 Grid World - 작은 미로
    gridworld = GridWorld(
        width=4,
        height=4,
        goal_states=[(0, 3)],
        obstacles=[(1, 1), (1, 2), (2, 2)],  # 작은 미로
        discount=0.9,
        start_state=(3, 0)
    )

    print("\n[Grid World 설정]")
    print(f"크기: {gridworld.width} x {gridworld.height}")
    print(f"시작 상태: {gridworld.start_state}")
    print(f"목표 상태: {gridworld.goal_states}")
    print(f"장애물: {gridworld.obstacles}")

    # TD(0) 학습
    td0 = TD0(
        env=gridworld,
        alpha=0.1,
        epsilon=0.1,
        gamma=0.9
    )

    print("\n[TD(0) 학습 시작]")
    print(f"에피소드 수: 500")
    
    policy, episode_rewards = td0.train(num_episodes=500, verbose=True)

    print("\n[학습 완료]")
    print("\n[학습된 정책]")
    gridworld.print_policy(policy)
    
    value_function = td0.get_value_function()
    gridworld.print_values(value_function)


def compare_convergence_speed():
    """
    수렴 속도 비교
    """
    print("\n" + "=" * 50)
    print("학습 수렴 속도 분석 (10x10)")
    print("=" * 50)

    # 10x10 Grid World - T자 패턴
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9)],
        obstacles=[
            # T자 형태
            (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),  # 가로
            (4, 5), (5, 5), (6, 5), (7, 5)  # 세로
        ],
        discount=0.95,
        start_state=(9, 0)
    )

    print("\n[TD(0) 학습 중...]")
    td0 = TD0(env=gridworld, alpha=0.1, epsilon=0.1, gamma=0.95)
    policy, rewards = td0.train(num_episodes=2000, verbose=False)
    
    # 수렴 지점 찾기 (연속 100 에피소드 평균이 0.95 이상)
    converged_at = -1
    for i in range(100, len(rewards)):
        avg = sum(rewards[i-100:i]) / 100
        if avg >= 0.95:
            converged_at = i
            break
    
    if converged_at > 0:
        print(f"✓ 수렴 달성: ~{converged_at} 에피소드")
        print(f"  (연속 100 에피소드 평균 보상 ≥ 0.95)")
    
    # 최종 성능
    final_avg = sum(rewards[-100:]) / 100
    print(f"✓ 최종 평균 보상 (마지막 100 에피소드): {final_avg:.3f}")
    
    print("\n[최종 학습된 정책]")
    gridworld.print_policy(policy)


if __name__ == "__main__":
    main()
    test_different_alpha()
    test_different_epsilon()
    test_small_grid()
    compare_convergence_speed()
