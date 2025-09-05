import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from itertools import combinations

plt.rcParams['font.family'] = 'Times New Roman'

pd.set_option('display.max_columns', None)

base_path = "C:/data/EEMS"
data_path = os.path.join(base_path, "data")
machine_ids = ["FEMS101_01", "FEMS101_02", "FEMS101_03", "FEMS101_04"]


def process_csv_to_parquet():
    """
    csv 파일을 parquet 형식으로 변환하여 저장
    """
    patterns = [
        "SPC팩 회전자개조_*_with_header.csv",
        "log_250716_Modification.csv"
    ]
    file_paths = []
    for pattern in patterns:
        file_paths.extend(glob.glob(os.path.join(data_path, pattern)))

    for csv_path in file_paths:
        filename = os.path.basename(csv_path)

        df = pd.read_csv(csv_path)
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)

        original_name = os.path.splitext(filename)[0]
        new_name = f"SPC-unfiltered-composed-{original_name}.parquet"
        parquet_path = os.path.join(data_path, new_name)

        df.to_parquet(parquet_path, engine='pyarrow')
        print(f"[Unfiltered] Saved: {parquet_path}")
# process_csv_to_parquet()


def process_and_save_filtered_parquet():
    """
    숫자형 컬럼은 float32로 변환
    일부 컬럼만 추출하여 filtered Parquet 파일로 저장
    """
    keep_columns = ['collect_time', 'machine_code', 'Load_Total_Power_Consumption']
    pattern = os.path.join(data_path, "SPC-unfiltered-composed-*.parquet")

    parquet_files = glob.glob(pattern)

    for parquet_path in parquet_files:
        filename = os.path.basename(parquet_path)
        df = pd.read_parquet(parquet_path)

        numeric_cols = df[keep_columns].select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)

        df = df[keep_columns]

        new_name = filename.replace("unfiltered", "filtered")
        new_path = os.path.join(data_path, new_name)

        df.to_parquet(new_path, engine='pyarrow')
        print(f"[Filtered] Saved: {new_path}")
# process_and_save_filtered_parquet()


def combine_and_split_by_machine():
    """
    machine_code 별로 분리하여 저장
    """
    pattern = os.path.join(data_path, "SPC-filtered-composed-*.parquet")
    parquet_files = glob.glob(pattern)

    df_list = [pd.read_parquet(f) for f in parquet_files]
    full_df = pd.concat(df_list, ignore_index=True)

    full_df['collect_time'] = pd.to_datetime(full_df['collect_time'], errors='coerce')
    full_df = full_df.reset_index(drop=True)

    for machine_id in machine_ids:
        df_machine = full_df[full_df['machine_code'] == machine_id]

        save_path = os.path.join(data_path, f"SPC-filtered-decomposed-{machine_id}.parquet")
        df_machine.to_parquet(save_path, engine='pyarrow', index=False)
        print(f"Saved {save_path} with {len(df_machine)} rows")
# combine_and_split_by_machine()


def compute_cumulative_energy_per_machine():
    """
    Load_Active_Power를 이용해 Load_Total_Power_Consumption 재계산
    """
    for machine_id in machine_ids:
        file_path = os.path.join(data_path, f"SPC-filtered-decomposed-{machine_id}.parquet")
        df = pd.read_parquet(file_path)
        df = df.sort_values('collect_time').reset_index(drop=True)

        df['time_diff_hr'] = df['collect_time'].diff().dt.total_seconds() / 3600
        df['avg_active_power'] = (df['Load_Active_Power'] + df['Load_Active_Power'].shift(1)) / 2
        df['interval_energy_wh'] = df['avg_active_power'] * df['time_diff_hr']
        df['Load_Total_Power_Consumption'] = df['interval_energy_wh'].cumsum().fillna(0)

        df = df.drop(columns=['time_diff_hr', 'avg_active_power', 'interval_energy_wh'])
        df.to_parquet(file_path, engine='pyarrow', index=False)
        print(f" 누적 에너지 계산 완료 및 저장: {file_path} ({len(df)} rows)")
# compute_cumulative_energy_per_machine()


def split_chunk_into_24h_subchunks_aligned(chunk_df, ref_time):
    """
    24시간 주기 맞추기
    """
    subchunks = []

    chunk_start = chunk_df['collect_time'].min()
    chunk_end   = chunk_df['collect_time'].max()

    aligned_start = pd.Timestamp.combine(chunk_start.date(), ref_time)

    step = pd.Timedelta(hours=24)
    current_start = aligned_start

    while current_start + step <= chunk_end:
        current_end = current_start + step - pd.Timedelta(seconds=1)
        mask = (chunk_df['collect_time'] >= current_start) & (chunk_df['collect_time'] <= current_end)
        sub_df = chunk_df.loc[mask]

        if not sub_df.empty:
            subchunks.append(sub_df)

        current_start += step

    return subchunks


def split_and_save_24h_chunks():
    """
    1시간 이상의 다운타임 기준으로 분리하고, 24시간 이상 지속되는 구간만 저장
    """
    downtime_threshold = pd.Timedelta(hours=1)
    min_duration = pd.Timedelta(hours=24)

    results = {}

    for m_id in machine_ids:
        file_path = os.path.join(data_path, f"SPC-filtered-decomposed-{m_id}.parquet")
        df = pd.read_parquet(file_path)
        df = df.sort_values('collect_time').reset_index(drop=True)

        global_start_time = df['collect_time'].iloc[0].time()

        gap = df['collect_time'].diff()
        chunk_id = (gap >= downtime_threshold).cumsum()
        df['chunk_id'] = chunk_id

        long_chunks = []
        chunk_counter = 1

        for chunk_id, chunk_df in df.groupby('chunk_id'):
            duration = chunk_df['collect_time'].iloc[-1] - chunk_df['collect_time'].iloc[0]
            if duration < min_duration:
                continue

            subchunks = split_chunk_into_24h_subchunks_aligned(chunk_df, global_start_time)
            subchunks = [sc for sc in subchunks if not sc.empty]

            for subchunk in subchunks:
                filename = f"SPC-filtered-decomposed-{m_id}-split-{chunk_counter}.parquet"
                save_path = os.path.join(data_path, filename)
                subchunk.to_parquet(save_path, engine='pyarrow', index=False)
                print(f" Saved: {save_path} ({len(subchunk)} rows)")
                chunk_counter += 1

            long_chunks.extend(subchunks)

        results[m_id] = long_chunks

        print(f"\n {m_id}: {len(long_chunks)} chunks saved (each ≈ 24h)")
        for i, chunk in enumerate(long_chunks):
            start = chunk['collect_time'].iloc[0]
            end = chunk['collect_time'].iloc[-1]
            print(f"   └─ Chunk {i + 1}: {start} → {end} (duration: {end - start}, rows: {len(chunk)})")

    return results
# split_and_save_24h_chunks()


def filter_chunks_with_diff_threshold():
    """
    diff >= 0.01인 청크만 반환
    """
    chunks_by_machine = split_and_save_24h_chunks()
    final_results = {}
    min_diff_threshold = 0.01

    for m_id in machine_ids:
        long_chunks = chunks_by_machine[m_id]

        filtered_chunks = []
        print(f"\n {m_id}: Checking {len(long_chunks)} chunks...")

        for i, chunk in enumerate(long_chunks):
            y = chunk['Load_Total_Power_Consumption']
            diffs = y.diff().abs().dropna()

            if (diffs < min_diff_threshold).all():
                start = chunk['collect_time'].iloc[0]
                end = chunk['collect_time'].iloc[-1]
                print(f" Chunk {i+1} skipped (all diffs < {min_diff_threshold}): {start} → {end}")
                continue

            filtered_chunks.append(chunk)

        final_results[m_id] = filtered_chunks

        print(f"\n {m_id}: {len(filtered_chunks)} valid chunks (each ≈ 24h)")
        for i, chunk in enumerate(filtered_chunks):
            start = chunk['collect_time'].iloc[0]
            end = chunk['collect_time'].iloc[-1]
            y = chunk['Load_Total_Power_Consumption']
            y_shifted = y - y.iloc[0]
            max_shifted = y_shifted.max()
            print(f"   └─ Chunk {i+1}: {start} → {end} (rows: {len(chunk)}, max_shifted: {max_shifted:.2f})")

    return final_results
# filter_chunks_with_diff_threshold()


def compare_chunk_slope_first_last():
    """
    첫 번째와 마지막 청크를 비교하여 전체 변화량과 전체 변화율 그리고 변화가 0이 아닌 구간만 고려한 평균 기울기 출력
    """
    all_chunks = filter_chunks_with_diff_threshold()

    for m_id, chunks in all_chunks.items():
        chunk_first = chunks[0]
        chunk_last = chunks[-1]

        t0 = chunk_first['collect_time']
        y0 = chunk_first['Load_Total_Power_Consumption']
        y0_shifted = y0 - y0.iloc[0]
        dt0_hours = (t0 - t0.iloc[0]).dt.total_seconds() / 3600
        mask0 = y0_shifted != 0
        slope0 = (y0_shifted[mask0].diff() / dt0_hours[mask0].diff()).mean()

        dy0 = y0.iloc[-1] - y0.iloc[0]
        total_time0 = (t0.iloc[-1] - t0.iloc[0]).total_seconds() / 3600
        rate0 = dy0 / total_time0

        t_last = chunk_last['collect_time']
        y_last = chunk_last['Load_Total_Power_Consumption']
        y_last_shifted = y_last - y_last.iloc[0]
        dt_last_hours = (t_last - t_last.iloc[0]).dt.total_seconds() / 3600
        mask_last = y_last_shifted != 0
        slope_last = (y_last_shifted[mask_last].diff() / dt_last_hours[mask_last].diff()).mean()

        dy_last = y_last.iloc[-1] - y_last.iloc[0]
        total_time_last = (t_last.iloc[-1] - t_last.iloc[0]).total_seconds() / 3600
        rate_last = dy_last / total_time_last

        print(f"{m_id}")
        print(f"   • Chunk [0] 전체 변화량:     {dy0:.2f} kW")
        print(f"   • Chunk [0] 전체 변화율:     {rate0:.4f} kW/시간")
        print(f"   • Chunk [0] 0 제외 변화율:   {slope0:.4f} kW/시간\n")

        print(f"   • Chunk [-1] 전체 변화량:    {dy_last:.2f} kW")
        print(f"   • Chunk [-1] 전체 변화율:    {rate_last:.4f} kW/시간")
        print(f"   • Chunk [-1] 0 제외 변화율:  {slope_last:.4f} kW/시간")
# compare_chunk_slope_first_last()


def split_chunk_by_flat_intervals(chunk, max_flat_hold_hours=0.1):
    """
    flat 구간이 0.1시간 이상 유지되면 해당 지점 전까지 subchunk로 분할
    """
    t = chunk['collect_time'].reset_index(drop=True)
    y = chunk['Load_Total_Power_Consumption'].reset_index(drop=True)
    delta_y = y.diff().abs()
    n = len(y)

    subchunks = []
    start_idx = 0
    cursor = 0

    while cursor < n - 1:
        if delta_y[cursor + 1] >= 0.01:
            cursor += 1
            continue

        hold_start = cursor
        while cursor < n - 1 and delta_y[cursor + 1] < 0.01:
            cursor += 1
        hold_end = cursor

        hold_duration = (t[hold_end] - t[hold_start]).total_seconds() / 3600
        if hold_duration < max_flat_hold_hours:
            cursor += 1
            continue

        if start_idx < hold_start:
            subchunks.append(chunk.iloc[start_idx:hold_start].copy())

        start_idx = hold_end + 1
        cursor = start_idx

    if start_idx < n:
        subchunks.append(chunk.iloc[start_idx:].copy())

    return subchunks


def print_subchunk_slopes_by_flat_hold(max_flat_hold_hours=0.1):
    """
    첫 번째와 마지막 청크를 평탄한 구간을 기준으로 서브청크를 만든 후,
    각 서브청크의 변화량, 변화율과 모든 서브청크들의 가중 평균 변화율 출력
    """
    all_chunks = filter_chunks_with_diff_threshold()

    for m_id, chunks in all_chunks.items():
        print(f"\n Machine: {m_id}")

        for idx in [0, -1]:
            chunk = chunks[idx]

            merged_subchunks = split_chunk_by_flat_intervals(chunk, max_flat_hold_hours=max_flat_hold_hours)

            print(f"\n Chunk [{idx}] - ΔY=0이 ≥ {max_flat_hold_hours}h 지속된 구간 전까지 분할 결과:")

            total_delta_y = 0.0
            total_delta_t = 0.0
            count_valid = 0

            for i, subchunk in enumerate(merged_subchunks, start=1):
                t0 = subchunk['collect_time'].iloc[0]
                t1 = subchunk['collect_time'].iloc[-1]
                y0 = subchunk['Load_Total_Power_Consumption'].iloc[0]
                y1 = subchunk['Load_Total_Power_Consumption'].iloc[-1]
                delta_t = (t1 - t0).total_seconds() / 3600
                delta_y = y1 - y0
                slope = delta_y / delta_t

                total_delta_y += delta_y
                total_delta_t += delta_t
                count_valid += 1

                print(f"   • Subchunk {i}: {t0} → {t1} | ΔY = {delta_y:.1f}, ΔT = {delta_t:.2f}h → slope = {slope:.2f} kW/h")

            weighted_avg_slope = total_delta_y / total_delta_t
            print(f"\n    서브청크 {count_valid}개 → 가중 평균 변화율 = {weighted_avg_slope:.2f} kW/h")
# print_subchunk_slopes_by_flat_hold()


def compute_min_rate_for_interval(d_start, d_end, chunks_by_machine):
    """
    변화율이 가장 낮은 모터 구하기
    """
    min_rate = float('inf')

    for m_id in machine_ids:
        chunks = chunks_by_machine[m_id]
        chunk_dict = {chunk['collect_time'].iloc[0].date(): chunk for chunk in chunks}

        if d_start not in chunk_dict or d_end not in chunk_dict:
            print(f" {m_id} 구간 누락: {d_start} 또는 {d_end}")
            return None

        chunk_start = chunk_dict[d_start]
        chunk_end = chunk_dict[d_end]

        start_val = chunk_start['Load_Total_Power_Consumption'].iloc[0]
        end_val = chunk_end['Load_Total_Power_Consumption'].iloc[-1]
        diff = abs(end_val - start_val)

        duration_sec = (chunk_end['collect_time'].iloc[-1] - chunk_start['collect_time'].iloc[0]).total_seconds()
        if duration_sec < 1e-2:
            print(f" {m_id} duration too short or zero!")
            return None

        rate = diff / (duration_sec / 3600)
        min_rate = min(min_rate, rate)

        print(f"   └─ {m_id}: Δ = {diff:.2f}, Rate = {rate:.2f} per hour")

    return min_rate


def find_max_of_min_rate_interval(min_overlap_days=1):
    """
    최소 변화율이 최대가 되는 구간 찾기
    """
    chunks_by_machine = filter_chunks_with_diff_threshold()

    date_map = {
        m_id: [chunk['collect_time'].iloc[0].date() for chunk in chunks_by_machine.get(m_id, [])]
        for m_id in machine_ids
    }

    common_dates = sorted(set.intersection(*map(set, date_map.values())))
    if not common_dates:
        print("# 공통 구간 없음 #")
        return None

    valid_intervals = [
        (d_start, d_end)
        for d_start, d_end in combinations(common_dates, 2)
        if (d_end - d_start).days >= min_overlap_days
    ]

    best_start, best_end = None, None
    best_min_rate_among_machines = -1
    interval_count = 0

    for d_start, d_end in valid_intervals:
        interval_count += 1
        print(f"\n# Checking interval # : {d_start} → {d_end}")

        min_rate = compute_min_rate_for_interval(d_start, d_end, chunks_by_machine)

        if min_rate is not None and min_rate > best_min_rate_among_machines:
            best_min_rate_among_machines = min_rate
            best_start, best_end = d_start, d_end
            print(f"# New best interval #: {best_start} → {best_end} (min rate = {best_min_rate_among_machines:.2f}/h)")

    print(f"\n 총 비교한 구간 수: {interval_count}개")
    if best_start and best_end:
        print(f"# 최종 선택된 구간 #: {best_start} → {best_end} (min rate = {best_min_rate_among_machines:.2f}/h)")
        return best_start, best_end, chunks_by_machine
    else:
        print("# 유효 구간 없음 #")
        return None
# find_max_of_min_rate_interval()