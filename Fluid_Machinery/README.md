# Pump Curve Modeling with GPR

## 프로젝트 소개
이 프로젝트는 **Pump의 유량(Flowrate)**과 **제어 변수(Stroke)**를 기반으로  
**Head, Power, Efficiency 곡면**을 학습하고, 시스템 곡선과의 교점을 찾아  
**효율 최적 후보점**을 도출하는 작업을 수행했습니다.  

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
