class GridWorld:
    """
    간단한 Grid World MDP 환경
    
    Grid 예시 (4x4):
    +---+---+---+---+
    | S |   |   | G |
    +---+---+---+---+
    |   | X |   |   |
    +---+---+---+---+
    |   |   |   |   |
    +---+---+---+---+
    |   |   |   |   |
    +---+---+---+---+
    
    S: 시작점, G: 목표 (reward +1), X: 장애물
    """

    ACTIONS = ["up", "down", "left", "right"]

    def __init__(self, width=4, height=4, goal_states=None, obstacles=None, discount=0.9):
        self.width = width
        self.height = height
        self.discount = discount
        
        # 기본 목표: 우상단 (0, width-1)
        self.goal_states = goal_states if goal_states else [(0, width - 1)]
        # 장애물
        self.obstacles = obstacles if obstacles else []

    def get_states(self):
        """모든 상태(좌표) 반환"""
        states = []
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) not in self.obstacles:
                    states.append((row, col))
        return states

    def get_actions(self, state):
        """주어진 상태에서 가능한 액션들 반환"""
        if state in self.goal_states:
            return []  # 목표 상태는 터미널 상태
        return self.ACTIONS

    def _get_next_state(self, state, action):
        """액션 수행 후 다음 상태 계산"""
        row, col = state
        
        if action == "up":
            next_state = (row - 1, col)
        elif action == "down":
            next_state = (row + 1, col)
        elif action == "left":
            next_state = (row, col - 1)
        elif action == "right":
            next_state = (row, col + 1)
        else:
            next_state = state

        # 벽이나 장애물에 부딪히면 제자리
        next_row, next_col = next_state
        if (
            next_row < 0
            or next_row >= self.height
            or next_col < 0
            or next_col >= self.width
            or next_state in self.obstacles
        ):
            return state
        
        return next_state

    def get_transitions(self, state, action):
        """
        (next_state, probability) 튜플 리스트 반환
        결정적 환경: 선택한 액션대로 100% 이동
        """
        next_state = self._get_next_state(state, action)
        return [(next_state, 1.0)]

    def get_reward(self, state, action, next_state):
        """보상 반환"""
        if next_state in self.goal_states:
            return 1.0
        return 0.0  # 기본 보상

    def get_discount_factor(self):
        return self.discount

    def print_values(self, value_function):
        """Value function을 그리드 형태로 출력"""
        print("\n=== Value Function ===")
        for row in range(self.height):
            row_str = ""
            for col in range(self.width):
                if (row, col) in self.obstacles:
                    row_str += "  [X]  "
                elif (row, col) in self.goal_states:
                    row_str += "  [G]  "
                else:
                    value = value_function.get_value((row, col))
                    row_str += f"{value:6.3f} "
            print(row_str)
        print()

    def print_policy(self, policy):
        """Policy를 그리드 형태로 출력"""
        action_symbols = {
            "up": "↑",
            "down": "↓",
            "left": "←",
            "right": "→",
            None: "·"
        }
        
        print("\n=== Policy ===")
        for row in range(self.height):
            row_str = ""
            for col in range(self.width):
                if (row, col) in self.obstacles:
                    row_str += " X "
                elif (row, col) in self.goal_states:
                    row_str += " G "
                else:
                    action = policy.select_action((row, col), self.ACTIONS)
                    row_str += f" {action_symbols.get(action, '?')} "
            print(row_str)
        print()
