from envs import GridWorld
from agents import TabularPolicy, PolicyIteration


def main():
    print("=" * 50)
    print("Grid World Policy Iteration 테스트")
    print("=" * 50)

    # 4x4 Grid World 생성
    gridworld = GridWorld(
        width=4,
        height=4,
        goal_states=[(0, 3)],
        obstacles=[(1, 1)],
        discount=0.9
    )

    print("\n[Grid World 설정]")
    print(f"크기: {gridworld.width} x {gridworld.height}")
    print(f"목표 상태: {gridworld.goal_states}")
    print(f"장애물: {gridworld.obstacles}")
    print(f"할인율: {gridworld.discount}")

    # 초기 정책 생성 (기본 액션: "up")
    policy = TabularPolicy(default_action="up")

    print("\n[초기 정책]")
    gridworld.print_policy(policy)

    # Policy Iteration 실행
    pi = PolicyIteration(gridworld, policy)
    iterations = pi.policy_iteration(max_iterations=100, theta=0.0001)

    print(f"\n[결과]")
    print(f"수렴까지 반복 횟수: {iterations}")

    # 최종 정책 출력
    print("\n[최종 정책]")
    gridworld.print_policy(policy)


def test_larger_grid():
    print("\n" + "=" * 50)
    print("더 큰 Grid World (10x10) Policy Iteration 테스트")
    print("=" * 50)

    # 10x10 Grid World
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9), (9, 9)],
        obstacles=[(1, 3), (2, 3), (3, 3), (4, 3), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 3), (7, 4), (7, 5)],
        discount=0.95
    )

    print("\n[Grid World 설정]")
    print(f"크기: {gridworld.width} x {gridworld.height}")
    print(f"목표 상태: {gridworld.goal_states}")
    print(f"장애물: {gridworld.obstacles}")
    print(f"할인율: {gridworld.discount}")

    # 초기 정책: 모두 오른쪽으로
    policy = TabularPolicy(default_action="right")

    print("\n[초기 정책]")
    gridworld.print_policy(policy)

    pi = PolicyIteration(gridworld, policy)
    iterations = pi.policy_iteration(max_iterations=100, theta=0.0001)

    print(f"\n[결과]")
    print(f"수렴까지 반복 횟수: {iterations}")

    print("\n[최종 정책]")
    gridworld.print_policy(policy)


if __name__ == "__main__":
    main()
    test_larger_grid()
