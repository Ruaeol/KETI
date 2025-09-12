# Motor Energy Consumption Analysis

## 프로젝트 소개
이 프로젝트는 **모터 개선 효과를 정량적으로 입증**하기 위해 수행되었습니다.  
데이터는 실제 현장에서 수집된 **전력 소비 로그**를 기반으로 하였으며,  
개선 전후의 소비 패턴을 비교하여 성능 향상 여부를 검증하는 데 목적이 있습니다.  

- 본 프로젝트: **모터 개선 효과 분석** 담당  

---

## 관련 자료
- [SPC PPT 보기](docs/SPC_PPT.pdf)
 **설명:** 총 41페이지, SPC 데이터 분석 내용 보고
  
---

## 데이터 안내
- 실제 데이터는 회사 소유로 포함되지 않음  
- 아래는 데이터 구조 예시  

### 원본 CSV (`*_with_header.csv`, `log_*.csv`)
| collect_time | machine_code | Load_Active_Power | Load_Total_Power_Consumption | ... |
|--------------|--------------|-------------------|------------------------------|-----|
| datetime     | string       | float             | float                        | ... |

### 전처리 후 Parquet (`SPC-*.parquet`)
| collect_time | machine_code | Load_Total_Power_Consumption |
|--------------|--------------|------------------------------|
| datetime     | string       | float                        |

---

## 분석 절차

1. **데이터 전처리**
   - CSV → Parquet 변환 (`process_csv_to_parquet`)  
   - 주요 컬럼만 추출 및 저장 (`process_and_save_filtered_parquet`)  
   - 기계별 분리 저장 (`combine_and_split_by_machine`)  
   - 누적 전력 소비량 재계산 (`compute_cumulative_energy_per_machine`)  

2. **청크 분할**
   - 다운타임(≥1시간) 기준으로 분할  
   - 24시간 이상 구간만 추출 및 저장 (`split_and_save_24h_chunks`)  
   - 의미 없는 구간(Δ < 0.01) 제거 (`filter_chunks_with_diff_threshold`)  

3. **변화율 분석**
   - `compare_chunk_slope_first_last`  
     → 첫/마지막 청크 비교
   - `print_subchunk_slopes_by_flat_hold`  
     → 평탄 구간 기준으로 분할 후 기울기 분석
   - `find_max_of_min_rate_interval`  
     → 여러 기계의 공통 구간을 찾아 **최소 변화율이 최대가 되는 기간** 도출  

---

## 기능 요약
- CSV → Parquet 변환 및 정리  
- 모터별 24h 단위 청크 분할 및 필터링  
- 전력 소비 곡선 기반 변화량·변화율 분석  
- 최종적으로 **개선 효과를 가장 잘 보여주는 구간 탐색**  

---

## 필요 패키지
- pandas
- numpy
- matplotlib
- pyarrow
