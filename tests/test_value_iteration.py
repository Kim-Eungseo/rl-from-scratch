from envs import GridWorld
from agents import TabularValueFunction, ValueIteration


def main():
    print("=" * 50)
    print("Grid World Value Iteration 테스트")
    print("=" * 50)

    # 4x4 Grid World 생성 - L자 장애물
    # 목표: (0, 3) - 우상단
    gridworld = GridWorld(
        width=4,
        height=4,
        goal_states=[(0, 3)],
        obstacles=[(1, 1), (2, 1), (1, 2)],  # L자 형태
        discount=0.9
    )

    print("\n[Grid World 설정]")
    print(f"크기: {gridworld.width} x {gridworld.height}")
    print(f"목표 상태: {gridworld.goal_states}")
    print(f"장애물: {gridworld.obstacles}")
    print(f"할인율: {gridworld.discount}")

    # Value Function 초기화
    values = TabularValueFunction(default_value=0.0)

    # Value Iteration 실행
    vi = ValueIteration(gridworld, values)
    iterations = vi.value_iteration(max_iterations=100, theta=0.0001)

    print(f"\n[결과]")
    print(f"수렴까지 반복 횟수: {iterations}")

    # Value Function 출력
    gridworld.print_values(values)

    # Policy 추출 및 출력
    policy = values.extract_policy(gridworld)
    gridworld.print_policy(policy)

    # 특정 상태들의 값 확인
    print("[주요 상태별 Value]")
    test_states = [(0, 0), (0, 2), (1, 0), (2, 2), (3, 3)]
    for state in test_states:
        if state not in gridworld.obstacles:
            print(f"  V{state} = {values.get_value(state):.4f}")


def test_larger_grid():
    print("\n" + "=" * 50)
    print("더 큰 Grid World (6x6) 테스트")
    print("=" * 50)

    # 10x10 Grid World - 섬 패턴 (여러 작은 장애물)
    gridworld = GridWorld(
        width=10,
        height=10,
        goal_states=[(0, 9), (9, 9)],  # 두 개의 목표
        obstacles=[
            # 작은 섬들 (2x2 블록)
            (2, 2), (2, 3), (3, 2), (3, 3),
            (5, 5), (5, 6), (6, 5), (6, 6),
            # 작은 섬들 (단일)
            (1, 7), (4, 1), (7, 3), (8, 8), (3, 8)
        ],
        discount=0.95
    )

    print("\n[Grid World 설정]")
    print(f"크기: {gridworld.width} x {gridworld.height}")
    print(f"목표 상태: {gridworld.goal_states}")
    print(f"장애물: {gridworld.obstacles}")
    print(f"할인율: {gridworld.discount}")

    values = TabularValueFunction(default_value=0.0)
    vi = ValueIteration(gridworld, values)
    iterations = vi.value_iteration(max_iterations=200, theta=0.0001)

    print(f"\n[결과]")
    print(f"수렴까지 반복 횟수: {iterations}")

    gridworld.print_values(values)
    
    policy = values.extract_policy(gridworld)
    gridworld.print_policy(policy)


if __name__ == "__main__":
    main()
    test_larger_grid()
