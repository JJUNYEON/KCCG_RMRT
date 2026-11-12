"""
===============================================================================
Filename    : SLON_Separtation.py
Description : 습식자력선별기 분배비율 계산 (최소제곱법)
Author      : 김준연 SW
Created     : 2026-10-13
Version     : 1.0.0
===============================================================================
"""

import numpy as np
from scipy.optimize import minimize

# 1. 실측 데이터 입력 (예시: 습식자력선별기 2호 데이터 기준)
actual_feed = np.array([0.895, 0.258])  # [Al2O3 실측값, Fe2O3 실측값]
C = np.array([0.74, 0.16])  # Concentrate 성분 [Al2O3, Fe2O3]
M = np.array([0.715, 0.145])  # Middling 성분 [Al2O3, Fe2O3]
T = np.array([2.63, 2.482])  # Tail 성분 [Al2O3, Fe2O3]


# 2. 오차 제곱 합(Objective Function) 정의
def objective(r):
    rc, rm, rt = r
    # 계산된 Feed 성분 = (Conc 비율 * Conc 성분) + (Middling 비율 * Middling 성분) + (Tail 비율 * Tail 성분)
    calc_feed = rc * C + rm * M + rt * T
    # 실측 Feed와 계산된 Feed의 차이(오차) 제곱의 합
    error = np.sum((calc_feed - actual_feed) ** 2)
    return error


# 3. 제약 조건 및 변수 범위 설정
# 조건 1: 질량 분배율의 합은 정확히 1 (100%)
constraints = {"type": "eq", "fun": lambda r: np.sum(r) - 1.0}
# 조건 2: 각 분배율은 0 이상 1 이하
bounds = ((0.001, 1), (0.001, 1), (0.001, 1))

# 4. 초기 추정값 설정 (Conc: 50%, Middling: 40%, Tail: 10%)
initial_guess = [0.5, 0.4, 0.1]

# 5. 최적화(Optimization) 수행
result = minimize(
    objective, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints
)

# 6. 결과 출력
if result.success:
    print("--- 물질수지 최적화 결과 ---")
    print(f"Concentrate 분배율: {result.x[0]*100:.2f}%")
    print(f"Middling 분배율:    {result.x[1]*100:.2f}%")
    print(f"Tail 분배율:        {result.x[2]*100:.2f}%")
else:
    print("최적화 실패:", result.message)
