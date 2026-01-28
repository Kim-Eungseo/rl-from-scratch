from envs import GridWorld
from agents import MonteCarlo


def main():
    print("=" * 50)
    print("Grid World Monte Carlo Control 테스트 (10x10)")
    print("=" * 50)

    # 10x10 Grid World 생성 - 미로 패턴 (좁은 통로)
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9)],
        obstacles=[
            # 세로 벽 1 (왼쪽)
            (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1),
            # 세로 벽 2 (중앙)
            (1, 4), (2, 4), (3, 4), (5, 4), (6, 4), (7, 4),
            # 가로 벽 (하단)
            (8, 2), (8, 3), (8, 4), (8, 5),
            # 세로 벽 3 (오른쪽)
            (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)
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

    # Monte Carlo 학습
    mc = MonteCarlo(
        env=gridworld,
        epsilon=0.1,  # ε-greedy의 epsilon
        discount=0.95,
        first_visit=True  # First-visit MC
    )

    print("\n[Monte Carlo 학습 시작]")
    print(f"에피소드 수: 20000")
    print(f"Epsilon (ε): {mc.epsilon}")
    print(f"First-visit MC: {mc.first_visit}")
    
    # 학습 실행
    policy = mc.train(num_episodes=20000, verbose=True)

    print("\n[학습 완료]")
    
    # 학습된 정책 출력
    print("\n[학습된 정책]")
    gridworld.print_policy(policy)
    
    # 학습된 Value Function 출력
    value_function = mc.get_value_function()
    gridworld.print_values(value_function)
    
    # 특정 상태들의 Q-값 확인 (예시로 좌상단, 중앙, 우상단, 좌하단, 우하단)
    print("[주요 상태별 Q-값]")
    test_states = [(0, 0), (0, 5), (5, 5), (9, 0), (9, 9)]
    for state in test_states:
        if state not in gridworld.obstacles:
            print(f"\n  State {state}:")
            for action in gridworld.ACTIONS:
                q_value = mc.qtable.get_q_value(state, action)
                print(f"    Q({state}, {action:>5}) = {q_value:7.4f}")


def test_larger_grid():
    print("\n" + "=" * 50)
    print("더 큰 Grid World (10x10) Monte Carlo 테스트")
    print("=" * 50)

    # 10x10 Grid World - 대각선 패턴
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9)],  # 우상단
        obstacles=[
            # 대각선 장애물 1
            (3, 2), (4, 3), (5, 4), (6, 5),
            # 대각선 장애물 2
            (2, 6), (3, 7), (4, 8),
            # 작은 섬들
            (7, 1), (8, 6), (6, 8)
        ],
        discount=0.95,
        start_state=(9, 0)  # 좌하단
    )

    print("\n[Grid World 설정]")
    print(f"크기: {gridworld.width} x {gridworld.height}")
    print(f"시작 상태: {gridworld.start_state}")
    print(f"목표 상태: {gridworld.goal_states}")
    print(f"장애물: {gridworld.obstacles}")
    print(f"할인율: {gridworld.discount}")

    # Monte Carlo 학습
    mc = MonteCarlo(
        env=gridworld,
        epsilon=0.15,  # 더 큰 exploration
        discount=0.95,
        first_visit=True
    )

    print("\n[Monte Carlo 학습 시작]")
    print(f"에피소드 수: 30000")
    print(f"Epsilon (ε): {mc.epsilon}")
    
    # 학습 실행
    policy = mc.train(num_episodes=30000, verbose=True)

    print("\n[학습 완료]")
    
    # 학습된 정책 출력
    print("\n[학습된 정책]")
    gridworld.print_policy(policy)
    
    # 학습된 Value Function 출력
    value_function = mc.get_value_function()
    gridworld.print_values(value_function)


def test_every_visit_mc():
    print("\n" + "=" * 50)
    print("Every-Visit Monte Carlo 테스트 (10x10)")
    print("=" * 50)

    # 10x10 Grid World - 복잡한 미로
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9)],
        obstacles=[
            # U자 형태
            (3, 3), (4, 3), (5, 3), (6, 3),
            (3, 5), (4, 5), (5, 5), (6, 5),
            (6, 4),
            # 추가 장애물
            (1, 7), (2, 7), (8, 2), (8, 8)
        ],
        discount=0.95,
        start_state=(9, 0)
    )

    print("\n[Grid World 설정]")
    print(f"크기: {gridworld.width} x {gridworld.height}")
    print(f"Every-visit MC로 학습")

    # Every-visit Monte Carlo
    mc = MonteCarlo(
        env=gridworld,
        epsilon=0.1,
        discount=0.95,
        first_visit=False  # Every-visit MC
    )

    print("\n[Monte Carlo 학습 시작]")
    print(f"에피소드 수: 20000")
    print(f"First-visit MC: {mc.first_visit}")
    
    policy = mc.train(num_episodes=20000, verbose=True)

    print("\n[학습 완료]")
    print("\n[학습된 정책]")
    gridworld.print_policy(policy)
    
    value_function = mc.get_value_function()
    gridworld.print_values(value_function)


if __name__ == "__main__":
    main()
    test_larger_grid()
    test_every_visit_mc()
