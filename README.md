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
│   ├── value_iteration.py           # Value Iteration 알고리즘
│   ├── monte_carlo.py               # Monte Carlo Control 알고리즘
│   ├── td0.py                       # TD(0) 알고리즘
│   └── td_lambda.py                 # TD(λ) 알고리즘
│
└── tests/             # 테스트 코드
    ├── __init__.py
    ├── test_policy_iteration.py     # Policy Iteration 테스트
    ├── test_value_iteration.py      # Value Iteration 테스트
    ├── test_monte_carlo.py          # Monte Carlo 테스트
    ├── test_td0.py                  # TD(0) 테스트
    └── test_td_lambda.py            # TD(λ) 테스트
```

## 구현된 알고리즘

### 1. Value Iteration (동적 프로그래밍)
- 동적 프로그래밍 기반 최적 가치 함수 계산
- Bellman Optimality Equation 사용
- 모델 기반 (Model-based): 환경의 transition과 reward 정보 필요

### 2. Policy Iteration (동적 프로그래밍)
- 정책 평가(Policy Evaluation)와 정책 개선(Policy Improvement) 반복
- 최적 정책 도출
- 모델 기반 (Model-based): 환경의 transition과 reward 정보 필요

### 3. Monte Carlo Control (샘플 기반 학습)
- **Model-free**: 환경의 dynamics를 몰라도 학습 가능
- 에피소드 샘플링을 통한 Q-value 추정
- ε-greedy 정책으로 exploration과 exploitation 균형
- First-visit MC와 Every-visit MC 지원

**알고리즘 구조:**
1. **Policy Improvement**: Q(s,a)에 대해 greedy하게 정책 개선
   - π(s) = argmax_a Q(s, a)
2. **Generate Episode**: ε-greedy로 에피소드 생성
   - exploration과 exploitation 균형
3. **Estimate Q**: 에피소드로부터 Q 값 추정
   - q_π(s,a) = E[G_t | S_t=s, A_t=a]
   - G_t = Σ(γ^k * R_(t+k+1))

### 4. TD(0) - One-Step Temporal Difference Learning
- **Model-free**: 환경의 dynamics를 몰라도 학습 가능
- **Online learning**: 에피소드 종료를 기다리지 않고 매 스텝마다 즉시 업데이트
- 가장 기본적인 TD learning 알고리즘
- 한 스텝 보고 즉시 업데이트 (bootstrapping)

**알고리즘 (각 transition 후):**
```
δ ← R + γ·V[Y] - V[X]  (TD error)
V[X] ← V[X] + α·δ      (현재 상태만 업데이트)
```

**특징:**
- 빠른 학습 속도 (매 스텝마다 업데이트)
- 낮은 variance, 약간의 bias (bootstrapping)
- 에피소드가 끝나지 않아도 학습 가능

### 5. TD(λ) - Temporal Difference Learning with Eligibility Traces
- **Model-free**: 환경의 dynamics를 몰라도 학습 가능
- **Online learning**: 에피소드 종료를 기다리지 않고 매 스텝마다 업데이트
- Eligibility traces로 TD(0)와 Monte Carlo의 장점 결합
- Replacing traces 구현

**알고리즘 (각 transition 후):**
```
δ ← R + γ·V[Y] - V[X]  (TD error)
for all states x:
    z[x] ← γ·λ·z[x]      (decay)
    if X = x:
        z[x] ← 1          (replacing trace)
    V[x] ← V[x] + α·δ·z[x]  (update)
```

**λ 파라미터:**
- λ=0: TD(0)와 동일 - one-step TD learning
- λ=1: Monte Carlo와 유사
- 0<λ<1: TD(0)와 MC의 중간 형태

## 사용 방법

### Value Iteration 테스트
```bash
python3 -m tests.test_value_iteration
```

### Policy Iteration 테스트
```bash
python3 -m tests.test_policy_iteration
```

### Monte Carlo Control 테스트
```bash
python3 -m tests.test_monte_carlo
```

### TD(0) 테스트
```bash
python3 -m tests.test_td0
```

### TD(λ) 테스트
```bash
python3 -m tests.test_td_lambda
```

## 환경 설명

### Grid World
- 다양한 크기의 그리드 환경 (4x4, 6x6, 10x10 등)
- 목표 상태(Goal)와 장애물(Obstacle) 설정 가능
- 시작 상태 지정 가능
- 4가지 액션: up, down, left, right
- 목표 도달 시 보상 +1
- 에피소드 생성 기능 (Monte Carlo 학습용)
  - `reset()`: 환경 초기화
  - `step(action)`: 액션 수행 및 결과 반환

## 요구사항

- Python 3.x
- 추가 라이브러리 불필요 (순수 Python 구현)
