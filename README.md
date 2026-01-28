# RL From Scratch

## 프로젝트 구조

```
rl-from-scratch/
├── envs/              # 환경 (Environments)
│   ├── __init__.py
│   └── gridworld.py   # Grid World MDP 환경
│
├── agents/            # 에이전트 및 알고리즘
│   ├── __init__.py
│   ├── policy.py                    # 정책 베이스 클래스
│   ├── tabular_policy.py            # 테이블 기반 정책
│   ├── value_function.py            # 가치 함수 베이스 클래스
│   ├── tabular_value_function.py    # 테이블 기반 가치 함수
│   ├── qtable.py                    # Q-테이블
│   ├── policy_iteration.py          # Policy Iteration 알고리즘
│   └── value_iteration.py           # Value Iteration 알고리즘
│
└── tests/             # 테스트 코드
    ├── __init__.py
    ├── test_policy_iteration.py     # Policy Iteration 테스트
    └── test_value_iteration.py      # Value Iteration 테스트
```

## 구현된 알고리즘

### 1. Value Iteration
- 동적 프로그래밍 기반 최적 가치 함수 계산
- Bellman Optimality Equation 사용

### 2. Policy Iteration
- 정책 평가(Policy Evaluation)와 정책 개선(Policy Improvement) 반복
- 최적 정책 도출

## 사용 방법

### Policy Iteration 테스트
```bash
python3 -m tests.test_policy_iteration
```

### Value Iteration 테스트
```bash
python3 -m tests.test_value_iteration
```

## 환경 설명

### Grid World
- 4x4 또는 10x10 그리드 환경
- 목표 상태(Goal)와 장애물(Obstacle) 설정 가능
- 4가지 액션: up, down, left, right
- 목표 도달 시 보상 +1

## 요구사항

- Python 3.x
- 추가 라이브러리 불필요 (순수 Python 구현)
