# %%
import os

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from functools import partial
from multiprocessing import Pool
from scipy.optimize import brentq

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel as C, RBF, Matern
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import StratifiedKFold

plt.rcParams['font.family'] = 'Times New Roman'

module_path = os.path.realpath(__file__)
base_dir = os.path.dirname(module_path)
print("base_dir:", base_dir)

GRID_DENSITY = 100


def _compute_intersection_for_stroke(args):
    """
    주어진 stroke에서 GPR 모델과 시스템 곡선의 교점을 계산
    """
    stroke, model, poly_func, f_min, f_max, num_points = args

    intersections = []

    def diff_func(flowrate):
        head_gpr = model.predict(np.array([[flowrate, stroke]])).item()
        head_sys = poly_func(flowrate)
        return head_gpr - head_sys

    f_root = brentq(diff_func, f_min, f_max)
    h_root = model.predict(np.array([[f_root, stroke]])).item()
    intersections.append((f_root, stroke, h_root))

    return intersections

# %%
class FluidMachinery:
    def __init__(self, name, category, control_param='Stroke', auto_draw=True):
        self._machine_name = name
        self._control = control_param
        self._rated = (1185, 2560, 0)[category]
        self._rho = (1.225, 997, 0)[category]
        self._g = 9.81

        module_path = os.path.realpath(__file__)
        self.base_dir = os.path.dirname(module_path)
        self.save_npy()
        self.df = self.load_npy
        self.save_system_npy()
        df_system = self.load_system_npy

        self.flowrate = self.df['Flowrate']
        self.dP = self.df['dP']
        if 'Head' in self.df.dtype.names:
            self.Head = self.df['Head']
        else:
            self.pressure = self.df['Pressure']
        self.control = self.df[self._control]
        self.power = self.df['Power']
        self.efficiency = self.df['Efficiency']
        self.flowrate_system = df_system['Flowrate']
        self.Head_system = df_system['Head']


    def settings(self):
        print(f"{self._machine_name}가 기본 장비입니다")


    def save_npy(self):
        """CSV 데이터를 읽어 npy 파일로 저장"""
        csv_path = os.path.join(self.base_dir, f'{self._machine_name}_data.csv')
        npy_path = os.path.join(self.base_dir, f'{self._machine_name}_data.npy')
        df = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8-sig')
        np.save(npy_path, df)


    def save_system_npy(self):
        """시스템 CSV 데이터를 읽어 npy 파일로 저장"""
        csv_path = os.path.join(self.base_dir, f'{self._machine_name}_system.csv')
        npy_path = os.path.join(self.base_dir, f'{self._machine_name}_system.npy')
        df_system = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8-sig')
        np.save(npy_path, df_system)


    @property
    def load_npy(self):
        """저장된 npy 데이터를 로드"""
        npy_path = os.path.join(self.base_dir, f'{self._machine_name}_data.npy')
        df = np.load(npy_path, allow_pickle=True)
        return df


    @property
    def load_system_npy(self):
        """저장된 시스템 npy 데이터를 로드"""
        npy_path = os.path.join(self.base_dir, f'{self._machine_name}_system.npy')
        df_system = np.load(npy_path, allow_pickle=True)
        return df_system


    def cal_efficiency(self):
        pass

class Blower(FluidMachinery):
    def __init__(self, name):
        self._machine_name = name


    def settings(self):
        print(f"{self._machine_name}는 압축기입니다")

class Pump(FluidMachinery):
    def __init__(self, name, category, auto_draw=False):
        super().__init__(name, category=category, auto_draw=auto_draw)


    def fit_poly_through_zero(self, x, y):
        """y절편 없이 2차 다항식(a*x^2 + b*x) 피팅"""
        X = np.column_stack((x ** 2, x))
        a, b = np.linalg.inv(X.T @ X) @ X.T @ y
        return a, b


    def _poly_func(self, x_val, a, b):
        """주어진 계수 a, b로 2차 다항식 값 계산"""
        return a * x_val ** 2 + b * x_val


    def _fit_system_curve(self):
        """시스템 곡선(flowrate vs head) 계수 a, b 계산"""
        q_system = self.flowrate_system
        pressure_system = self.Head_system
        a, b = self.fit_poly_through_zero(q_system, pressure_system)
        return a, b


    def cal_efficiency(self, q, h, p):
        """유량(q), 양정(h), 소비전력(p)으로 효율 계산"""
        eff = 100 * ((q / 60) * (h * self._rho * self._g) / 1000) / p
        return eff


    def _fit_surface(self, x1, x2, y, cv_splits=5):
        """
        GaussianProcessRegressor로 곡면을 학습
        내부 optimizer로 하이퍼파라미터 최적화
        교차 검증을 통해 모델 성능 평가
        """
        X = np.c_[x1, x2]

        kernel = (
                         C(1.0, (1e-2, 1e2)) *
                         RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e4))
                         + C(1.0, (1e-2, 1e2)) *
                         Matern(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e4), nu=1.5)
                 ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))

        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=42
        )
        model.fit(X, y.ravel())

        print("\n[Hyperparameter Optimization Result]")
        print(f"Best Kernel: {model.kernel_}")

        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = []

        unique_vals, stratify_labels = np.unique(self.control, return_inverse=True)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, stratify_labels)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_model = GaussianProcessRegressor(
                kernel=model.kernel_,
                alpha=1e-6,
                optimizer=None,
                normalize_y=True
            )
            fold_model.fit(X_train, y_train.ravel())
            y_pred = fold_model.predict(X_test)
            fold_r2 = r2(y_test, y_pred)
            scores.append(fold_r2)

            unique_vals_test, counts = np.unique(stratify_labels[test_idx], return_counts=True)
            for label, count in zip(unique_vals_test, counts):
                print(f"Fold {fold}: Stratify label {label} -> {count} samples")
            print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            print(f"  Fold {fold} R² Score: {fold_r2:.4f}")

        mean_cv_score = np.mean(scores)
        print(f"\nMean cross-validated R²: {mean_cv_score:.4f}")

        final_model = GaussianProcessRegressor(
            kernel=model.kernel_,
            alpha=1e-8,
            optimizer=None,
            normalize_y=True
        )
        final_model.fit(X, y.ravel())
        y_pred = final_model.predict(X)
        final_r2_score = r2(y, y_pred)
        print(f"Final model R² score on full data: {final_r2_score:.4f}")

        return final_model, y_pred


    def fit_head_and_power_surface(self, flowrate, stroke, head, power, cv_splits=5, use_bayes=True):
        """
        Flowrate와 Stroke 기반으로 Head 곡면 학습 후, Head 예측값으로 Power 곡면 학습
        """
        print("\n=== Head Surface 학습 시작 ===")
        model_head, head_pred = self._fit_surface(
            x1=flowrate,
            x2=stroke,
            y=head,
            cv_splits=cv_splits,
            # use_bayes=False
        )

        print("\n=== Power Surface 학습 시작 ===")
        fit_power_model = partial(
            self._fit_surface,
            x1=flowrate,
            x2=head_pred,
            cv_splits=cv_splits,
            # use_bayes=False
        )

        power_model, power_pred = fit_power_model(y=power)

        return model_head, power_model, head_pred, power_pred


    def visualize_head_surface_with_model(self):
        """
        Head 곡면과 실제/예측값을 3D로 시각화
        """
        x1 = self.flowrate
        x2 = self.control
        y = self.Head

        model, y_pred = self._fit_surface(x1, x2, y)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        grid_f = np.linspace(x1.min(), x1.max(), GRID_DENSITY)
        grid_s = np.linspace(x2.min(), x2.max(), GRID_DENSITY)
        F_grid, S_grid = np.meshgrid(grid_f, grid_s)
        X_grid = np.c_[F_grid.ravel(), S_grid.ravel()]
        H_pred_grid = model.predict(X_grid).reshape(F_grid.shape)

        ax.plot_surface(F_grid, S_grid, H_pred_grid, cmap='viridis', alpha=0.6)

        ax.scatter(x1, x2, y, c='blue', marker='^', label='Actual', s=40)

        ax.scatter(x1, x2, y_pred, c='green', marker='o', label='Predicted', s=40)

        for xi, si, yi, ypi in zip(x1, x2, y, y_pred):
            ax.plot([xi, xi], [si, si], [yi, ypi], color='red', linestyle='-', linewidth=3, alpha=1.0)

        ax.set_xlabel('Flowrate', fontsize=20)
        ax.set_ylabel('Stroke', fontsize=20)
        ax.set_zlabel('Head', fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='z', labelsize=20)
        ax.set_title('GPR Head Surface with Actual & Predicted', fontsize=30)
        ax.legend(fontsize=20)
        plt.tight_layout()
        plt.show()

        return model


    def fit_head_and_efficiency_surface(self, flowrate, stroke, head, power, cv_splits=5):
        """
        Flowrate와 Stroke로 Head 곡면 학습 후, Head 기반으로 Efficiency 곡면 학습
        """
        print("\n=== Head Surface 학습 시작 ===")
        model_head, head_pred = self._fit_surface(
            x1=flowrate,
            x2=stroke,
            y=head,
            cv_splits=cv_splits,
        )

        eff_true = self.cal_efficiency(flowrate, head, power)

        print("\n=== Efficiency Surface 학습 시작 ===")
        fit_eff_model = partial(
            self._fit_surface,
            x1=flowrate,
            x2=head_pred,
            cv_splits=cv_splits,
        )
        eff_model, eff_pred = fit_eff_model(y=eff_true)

        return model_head, eff_model, head_pred, eff_pred


    def find_and_plot_intersections_2d_and_3d(self, num_points=100, use_multiprocessing=True):
        """
        Flowrate와 Stroke로 학습된 Head 곡면과 시스템 곡선의 2D/3D 교점 계산 및 시각화
        """
        x1 = self.flowrate
        x2 = self.control
        y = self.Head

        model, _ = self._fit_surface(x1, x2, y)
        # model, _ = self._fit_surface(x1, x2, y, use_bayes=False)
        a, b = self._fit_system_curve()

        self.poly_func = partial(self._poly_func, a=a, b=b)

        f_min, f_max = x1.min(), x1.max()
        s_min, s_max = x2.min(), x2.max()
        f_grid = np.linspace(f_min, f_max, num_points)
        s_grid = np.linspace(s_min, s_max, num_points)
        F, S = np.meshgrid(f_grid, s_grid)
        points = np.c_[F.ravel(), S.ravel()]

        target_strokes = np.unique(x2)
        plt.figure(figsize=(10, 6))
        all_intersections = []

        for stroke in target_strokes:
            args = (stroke, model, self.poly_func, f_min, f_max, num_points)
            intersections = _compute_intersection_for_stroke(args)

            gpr_vals = [model.predict(np.array([[f, stroke]])).item() for f in f_grid]
            sys_vals = self.poly_func(f_grid)

            intersections = np.array(intersections)
            all_intersections.extend(intersections)

            plt.plot(f_grid, gpr_vals, label=f'GPR (stroke={stroke:.1f})')
            plt.plot(f_grid, sys_vals, linestyle='--', label='System Curve' if stroke == target_strokes[0] else None)
            plt.scatter(intersections[:, 0], intersections[:, 2], color='red', s=40, label=f'Intersections (stroke={stroke:.1f})')

            print(f"\nStroke = {stroke:.4f} 교점 좌표:")
            for pt in intersections:
                print(f"  Flowrate: {pt[0]:.6f}, Stroke: {pt[1]:.6f}, Head: {pt[2]:.6f}")

        plt.xlabel('Flowrate')
        plt.ylabel('Head')
        plt.title('Flowrate vs Head with Intersections for Specified Strokes')
        plt.legend()
        plt.grid(True)
        plt.show()

        H_gpr = np.array([model.predict(np.array([pt])).item() for pt in points]).reshape(F.shape)
        H_sys = self.poly_func(F)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(F, S, H_gpr, alpha=0.6, cmap='viridis')
        ax.plot_surface(F, S, H_sys, alpha=0.3, color='gray')

        all_intersections = np.array(all_intersections)
        ax.scatter(all_intersections[:, 0], all_intersections[:, 1], all_intersections[:, 2], color='red', s=50, label='Intersections')

        print(" 전체 교차점 좌표 (Flowrate, Stroke, Head):")
        for i, (f, s, h) in enumerate(all_intersections):
            print(f"   {i + 1:>2d}) Flowrate = {f:.3f}, Stroke = {s:.3f}, Head = {h:.3f}")

        ax.set_xlabel('Flowrate')
        ax.set_ylabel('Stroke')
        ax.set_zlabel('Head')
        ax.set_title('3D GPR vs System Curve with Intersections')
        ax.legend()
        plt.show()

        args_list = [(stroke, model, self.poly_func, f_min, f_max, num_points) for stroke in s_grid]
        if use_multiprocessing:
            pool = Pool(10)
            results = pool.map(_compute_intersection_for_stroke, args_list)
            pool.close()
            pool.join()
        else:
            results = [_compute_intersection_for_stroke(args) for args in args_list]

        intersection_points = [pt for sublist in results for pt in sublist]
        intersection_points = np.array(intersection_points)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(F, S, H_gpr, cmap='viridis', alpha=0.6, edgecolor='none')
        ax.plot_surface(F, S, H_sys, color='gray', alpha=0.4)
        ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], color='red', s=40, label='Exact Intersections')

        print(f'전체 교점 좌표 : (Flowrate, Stroke, Head)\n', intersection_points)
        print(f'전체 교점 좌표 수 :', intersection_points.shape)

        ax.set_xlabel('Flowrate')
        ax.set_ylabel('Stroke')
        ax.set_zlabel('Head')
        ax.set_title('Exact Intersections: GPR Surface & System Curve (3D)')
        ax.legend()
        plt.show()
        return intersection_points


    def get_predicted_power(self, model_power, flowrate, head):
        """
        학습된 Power 모델을 사용해 (flowrate, head)에서의 Power 예측
        """
        X_power = np.array([[flowrate, head]])
        return model_power.predict(X_power).item()


    def get_predicted_efficiency(self, model_eff, flowrate, head):
        """
        학습된 Efficiency 모델을 사용해 (flowrate, head)에서의 Efficiency 예측
        """
        X_eff = np.array([[flowrate, head]])
        return model_eff.predict(X_eff).item()


    def find_top_efficiency_points(self, model_head, model_eff, model_power=None, top_n=100):
        """
        GPR Head 곡면과 System Curve 교점 계산, 전체 grid에서 Top N 효율점 탐색 및 출력
        """
        intersection_points = pump.find_and_plot_intersections_2d_and_3d(use_multiprocessing=True)
        intersection_results = []
        for f, s, h in intersection_points:
            p_pred = self.get_predicted_power(model_power, f, h) if model_power else None
            eff_pred = self.get_predicted_efficiency(model_eff, f, h)
            intersection_results.append((f, s, h, p_pred, eff_pred))

        if intersection_results:
            print(f"\n=== GPR 곡면과 System Curve 교차점 ===")
            for i, (f, s, h, p, eff) in enumerate(intersection_results, start=1):
                if model_power:
                    print(f"{i}) Flowrate={f:.3f}, Stroke={s:.3f}, Head={h:.3f}, "f"Efficiency={eff:.2f}%, Power={p:.3f}")
                else:
                    print(f"{i}) Flowrate={f:.3f}, Stroke={s:.3f}, Head={h:.3f}, Efficiency={eff:.2f}%")

        flow_grid = np.linspace(self.flowrate.min(), self.flowrate.max(), 100)
        stroke_grid = np.linspace(self.control.min(), self.control.max(), 100)
        F, S = np.meshgrid(flow_grid, stroke_grid)
        X_grid = np.c_[F.ravel(), S.ravel()]

        H_pred = model_head.predict(X_grid)
        Eff_pred = model_eff.predict(np.c_[X_grid[:, 0], H_pred])

        top_idx = np.argsort(Eff_pred)[-top_n:][::-1]

        print(f"\n=== Top {top_n} Efficiency Points ===")
        top_points = []
        for i, idx in enumerate(top_idx, start=1):
            f, s, h, eff = X_grid[idx, 0], X_grid[idx, 1], H_pred[idx], Eff_pred[idx]
            if model_power:
                p = self.get_predicted_power(model_power, f, h)
                print(f"{i}) Flowrate={f:.3f}, Stroke={s:.3f}, Head={h:.3f}, Efficiency={eff:.2f}%, Power={p:.3f}")
            else:
                print(f"{i}) Flowrate={f:.3f}, Stroke={s:.3f}, Head={h:.3f}, Efficiency={eff:.2f}%")
            top_points.append((f, s, h, eff))

        return top_points


class Compressor(FluidMachinery):
    def __init__(self, name):
        self._machine_name = name


    def settings(self):
        print(f"{self._machine_name}는 압축기입니다")


def create_machine(category, name, auto_draw=True):
    machines = [Blower, Pump, Compressor]
    return machines[category](name, category, auto_draw=auto_draw)


if __name__ == "__main__":
    pump = create_machine(1, "pump", auto_draw=True)

    model_head, model_power, head_pred, power_pred = pump.fit_head_and_power_surface(pump.flowrate, pump.control, pump.Head, pump.power)
    pump.visualize_head_surface_with_model()
    model_head_for_eff, model_eff, head_pred_for_eff, eff_pred = pump.fit_head_and_efficiency_surface(pump.flowrate, pump.control, pump.Head, pump.power)
    pump.find_top_efficiency_points(model_head_for_eff, model_eff, model_power=model_power)