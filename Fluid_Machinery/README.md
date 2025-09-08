# Pump Curve Modeling with GPR

## 프로젝트 소개
이 프로젝트는 펌프의 운전 데이터를 기반으로  
Gaussian Process Regression (GPR)을 활용하여  
Head, Power, Efficiency 특성 곡면을 모델링합니다.  

시스템 곡선(System Curve)과의 교점을 계산하여  
펌프가 실제 시스템에서 운전 가능한 지점을 도출했고,  
효율 최적화 탐색을 하여  
학습된 곡면 전체를 대상으로 효율이 높은 Top N 후보점을 계산했습니다.

---

## 데이터 안내
- 실제 데이터는 회사 소유로 포함되지 않음  
- 아래는 데이터 구조 예시  

### Pump 데이터 (`<machine_name>_data.csv`)
| Flowrate | dP   | Head | Power | Efficiency | Stroke |
|----------|------|------|-------|------------|--------|
| float    | float| float| float | float      | float  |

### 시스템 곡선 데이터 (`<machine_name>_system.csv`)
| Flowrate | Head |
|----------|------|
| float    | float|

---

## 기능 요약
- GPR(Gaussian Process Regressor)을 이용한 **Head / Power / Efficiency 곡면 학습**  
- Flowrate와 Stroke 기반의 **2D/3D 시스템 곡선 교점 계산 및 시각화**  
- Flowrate와 Stroke를 기반으로 **Head 곡면 학습**  
- Head 기반으로 **Power 곡면 학습**  
- Head 기반으로 **Efficiency 곡면 학습**  
- 학습된 GPR Head 및 Efficiency 곡면을 기반으로 **Top N 효율 후보 계산**  
- 각 후보점의 Flowrate, Stroke, Head, Efficiency, Power 출력  
- 실제 운전 조건(예: Stroke 제한 등)은 포함하지 않음  
- 후보점은 전달용이며, 실제 운전/실험 조건은 실무자가 결정  

---

## 필요 패키지
- numpy
- matplotlib
- scipy
- scikit-learn
