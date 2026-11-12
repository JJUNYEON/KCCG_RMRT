"""
===============================================================================
Filename    : PSA_P80_Calculation.py
Description : CSV에서 체 분석 결과를 읽어 누적 통과율, P80 및 입도분포 계산
Author      : 김준연 SW
Created     : 2026-09-07
Version     : 2.0.0
===============================================================================

CSV 형식(행=체 구간, 열=시료):

    sieve_mm,Sample 1,Sample 2
    +19,0,0
    16,2.2,0
    ...
    -0.106,14.3,9.2

첫 번째 열의 음수 값 또는 "pan"은 팬(pan)으로 처리
나머지 양수 값은 실제 체 눈 크기(mm)로 처리
ex) "+19"는 19 mm 체 위 잔류분을 뜻하며, 19 mm 통과율 계산에 포함됨
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_sieve_label(label: str) -> float | None:
    """체 표기를 mm 단위 숫자로 변환하고, 팬은 None으로 반환"""

    cleaned = label.strip().lower().replace("mm", "").strip()

    if cleaned in {"pan", "팬"}:
        return None

    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"인식할 수 없는 체 크기 표기: {label!r}") from exc

    if value < 0:
        return None
    if value == 0:
        raise ValueError(
            "체 크기는 0보다 커야 합니다. 팬은 음수 또는 'pan'으로 표시하세요."
        )

    return value


def read_sieve_csv(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    CSV에서 체 크기와 시료별 잔류량/잔류율을 읽음

    반환값:
        sieve_sizes: 내림차순 체 크기(mm), 팬 제외
        retained: 시료별 잔류 데이터(행=시료, 열=체 구간+팬)
        sample_names: 시료명
        original_totals: 정규화 전 시료별 합계
    """

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    if len(rows) < 3:
        raise ValueError("CSV에는 헤더, 최소 1개의 체 구간, 팬 행이 필요합니다.")

    header = [cell.strip() for cell in rows[0]]
    if len(header) < 2:
        raise ValueError("CSV에는 체 크기 열과 최소 1개의 시료 열이 필요합니다.")

    sample_names = header[1:]
    if any(not name for name in sample_names):
        raise ValueError("모든 시료 열에 이름을 입력하세요.")
    if len(set(sample_names)) != len(sample_names):
        raise ValueError("시료명은 서로 달라야 합니다.")

    sieve_rows: list[tuple[float, list[float]]] = []
    pan_rows: list[list[float]] = []

    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(
                f"CSV {line_number}행의 열 개수가 헤더와 다릅니다: "
                f"{len(row)}개(필요: {len(header)}개)"
            )

        size = parse_sieve_label(row[0])

        try:
            values = [float(cell.strip()) for cell in row[1:]]
        except ValueError as exc:
            raise ValueError(
                f"CSV {line_number}행에 숫자가 아닌 잔류값이 있습니다."
            ) from exc

        if not np.all(np.isfinite(values)):
            raise ValueError(f"CSV {line_number}행에 유한하지 않은 값이 있습니다.")
        if np.any(np.asarray(values) < 0):
            raise ValueError(f"CSV {line_number}행에 음수 잔류값이 있습니다.")

        if size is None:
            pan_rows.append(values)
        else:
            sieve_rows.append((size, values))

    if len(pan_rows) != 1:
        raise ValueError(
            "팬 행은 정확히 1개여야 하며, 음수 체 크기 또는 'pan'으로 표시해야 합니다."
        )
    if not sieve_rows:
        raise ValueError("양수로 표시된 실제 체 크기가 하나 이상 필요합니다.")

    sieve_rows.sort(key=lambda item: item[0], reverse=True)
    sieve_sizes = np.asarray([item[0] for item in sieve_rows], dtype=float)

    if len(np.unique(sieve_sizes)) != len(sieve_sizes):
        raise ValueError("중복된 체 크기가 있습니다.")

    # 행=시료, 열=큰 체부터 작은 체, 마지막 열=팬
    retained_by_fraction = np.asarray(
        [item[1] for item in sieve_rows] + pan_rows, dtype=float
    )
    retained = retained_by_fraction.T
    original_totals = retained.sum(axis=1)

    if np.any(original_totals <= 0):
        bad_samples = [
            sample_names[i] for i, total in enumerate(original_totals) if total <= 0
        ]
        raise ValueError(f"잔류값 합계가 0인 시료가 있습니다: {', '.join(bad_samples)}")

    return sieve_sizes, retained, sample_names, original_totals


def calculate_cumulative_passing(retained_data: np.ndarray) -> np.ndarray:
    """체별 잔류량 또는 잔류율을 100% 기준 누적 통과율로 변환"""

    total = retained_data.sum(axis=1, keepdims=True)
    normalized_retained = retained_data / total * 100.0

    # 마지막 열(팬)은 실제 체가 아니므로 누적 잔류 계산에서 제외합니다.
    cumulative_retained = np.cumsum(normalized_retained[:, :-1], axis=1)
    return 100.0 - cumulative_retained


def calculate_px(
    sieve_sizes: np.ndarray,
    passing: np.ndarray,
    target_passing: float = 80.0,
) -> float:
    """log10(입도)-누적 통과율 선형 보간으로 Px를 계산"""

    if not 0 < target_passing < 100:
        raise ValueError("목표 통과율은 0보다 크고 100보다 작아야 합니다.")

    if target_passing < passing.min() or target_passing > passing.max():
        return float("nan")

    order = np.argsort(passing, kind="stable")
    passing_sorted = passing[order]
    log_sieve_sorted = np.log10(sieve_sizes[order])
    log_px = np.interp(target_passing, passing_sorted, log_sieve_sorted)
    return float(10**log_px)


def plot_psd(
    sieve_sizes: np.ndarray,
    cumulative_passing: np.ndarray,
    sample_names: list[str],
    p80_values: list[float],
    output_path: Path,
) -> None:
    """로그 입도축으로 누적 입도분포를 그리고 PNG로 저장"""

    fig, ax = plt.subplots(figsize=(11, 7))

    # x축을 작은 입도에서 큰 입도 순서로 표시합니다.
    plot_order = np.argsort(sieve_sizes)

    for index, (name, passing, p80) in enumerate(
        zip(sample_names, cumulative_passing, p80_values)
    ):
        p80_text = f"{p80:.3f} mm" if np.isfinite(p80) else "측정 범위 밖"
        (line,) = ax.plot(
            sieve_sizes[plot_order],
            passing[plot_order],
            marker="o",
            markersize=7,
            linewidth=2,
            label=f"{name} (P80 = {p80_text})",
        )

        if np.isfinite(p80):
            color = line.get_color()
            ax.scatter(p80, 80, color=color, s=80, zorder=5)
            ax.vlines(p80, 0, 80, color=color, linestyle=":", linewidth=2)
            # 여러 시료의 P80이 가까울 때 라벨이 겹치지 않도록 높이를 교대로 배치합니다.
            y_offset = 10 + (index % 3) * 18
            ax.annotate(
                f"{name}: {p80:.3f} mm",
                xy=(p80, 80),
                xytext=(-6, y_offset),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=10,
            )

    ax.axhline(y=80, color="black", linestyle="--", linewidth=1.5, label="80% passing")
    ax.set_xscale("log")

    ticks = np.sort(sieve_sizes)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{value:g}" for value in ticks])
    ax.set_xlim(ticks[0] / 1.3, ticks[-1] * 1.3)
    ax.set_ylim(0, 105)

    ax.set_xlabel("Sieve size (mm)", fontsize=12)
    ax.set_ylabel("Cumulative passing (%)", fontsize=12)
    ax.set_title("Particle Size Distribution and P80", fontsize=15)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSV 체 분석 결과에서 누적 통과율과 P80을 계산합니다."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("PSA_input.csv"),
        help="입력 CSV 경로(기본값: 스크립트와 같은 폴더의 PSA_input.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="그래프 PNG 저장 경로(기본값: 입력파일명_PSD_P80.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="그래프 창을 열지 않고 PNG만 저장",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    csv_path = args.csv_file.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else csv_path.with_name(f"{csv_path.stem}_PSD_P80.png")
    )

    sieve_sizes, retained, sample_names, original_totals = read_sieve_csv(csv_path)
    cumulative_passing = calculate_cumulative_passing(retained)
    p80_values = [
        calculate_px(sieve_sizes, passing, 80.0) for passing in cumulative_passing
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_psd(sieve_sizes, cumulative_passing, sample_names, p80_values, output_path)

    print("P80 calculation results")
    print("-" * 50)
    for name, total, p80 in zip(sample_names, original_totals, p80_values):
        if np.isfinite(p80):
            print(f"{name}: P80 = {p80:.3f} mm  (input total = {total:.3f})")
        else:
            print(f"{name}: P80 = N/A (80% passing is outside measured range)")
    print(f"Graph saved: {output_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
