import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    outer: np.ndarray
    inner: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect square/rectangular hollow frames (口) from a Raspberry Pi camera."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--min-area", type=int, default=2500, help="Minimum outer contour area.")
    parser.add_argument(
        "--aspect-tolerance",
        type=float,
        default=0.35,
        help="Allowed width/height ratio difference from 1.0.",
    )
    parser.add_argument(
        "--approx-epsilon",
        type=float,
        default=0.02,
        help="Polygon approximation factor.",
    )
    parser.add_argument(
        "--threshold-block-size",
        type=int,
        default=31,
        help="Adaptive threshold block size, must be odd.",
    )
    parser.add_argument(
        "--threshold-c",
        type=int,
        default=8,
        help="Adaptive threshold constant.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Optional image path for single-image test mode.",
    )
    return parser.parse_args()


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(4, 2).astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def is_quadrilateral(contour: np.ndarray, epsilon_factor: float) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None
    return approx


def aspect_ratio_ok(bbox: tuple[int, int, int, int], tolerance: float) -> bool:
    _, _, width, height = bbox
    if height == 0:
        return False
    ratio = width / float(height)
    return abs(ratio - 1.0) <= tolerance


def centered_ratio(outer_bbox: tuple[int, int, int, int], inner_bbox: tuple[int, int, int, int]) -> float:
    ox, oy, ow, oh = outer_bbox
    ix, iy, iw, ih = inner_bbox
    outer_center = np.array([ox + ow / 2.0, oy + oh / 2.0])
    inner_center = np.array([ix + iw / 2.0, iy + ih / 2.0])
    distance = np.linalg.norm(outer_center - inner_center)
    scale = max(ow, oh)
    if scale == 0:
        return 0.0
    return 1.0 - min(distance / scale, 1.0)


def iter_descendants(hierarchy: np.ndarray, start_index: int):
    child_index = hierarchy[start_index][2]
    while child_index != -1:
        yield child_index
        yield from iter_descendants(hierarchy, child_index)
        child_index = hierarchy[child_index][0]


def preprocess(frame: np.ndarray, block_size: int, threshold_c: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        threshold_c,
    )
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned


def detect_square_frames(
    frame: np.ndarray,
    min_area: int,
    aspect_tolerance: float,
    epsilon_factor: float,
    block_size: int,
    threshold_c: int,
) -> tuple[list[Detection], np.ndarray]:
    mask = preprocess(frame, block_size, threshold_c)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    if hierarchy is None:
        return detections, mask

    hierarchy = hierarchy[0]
    contour_count = len(contours)

    for index in range(contour_count):
        contour = contours[index]
        if hierarchy[index][3] != -1:
            continue

        outer_area = cv2.contourArea(contour)
        if outer_area < min_area:
            continue

        outer_quad = is_quadrilateral(contour, epsilon_factor)
        if outer_quad is None:
            continue

        outer_bbox = cv2.boundingRect(outer_quad)
        if not aspect_ratio_ok(outer_bbox, aspect_tolerance):
            continue

        best_match: Detection | None = None

        for child_index in iter_descendants(hierarchy, index):
            child_contour = contours[child_index]
            inner_area = cv2.contourArea(child_contour)
            if inner_area > 0:
                inner_quad = is_quadrilateral(child_contour, epsilon_factor)
                if inner_quad is not None:
                    inner_bbox = cv2.boundingRect(inner_quad)
                    if aspect_ratio_ok(inner_bbox, aspect_tolerance):
                        ox, oy, ow, oh = outer_bbox
                        ix, iy, iw, ih = inner_bbox
                        area_ratio = inner_area / outer_area
                        inside = ix > ox and iy > oy and ix + iw < ox + ow and iy + ih < oy + oh
                        if inside and 0.08 <= area_ratio <= 0.8:
                            centering = centered_ratio(outer_bbox, inner_bbox)
                            shape_score = 1.0 - abs((ow / float(oh)) - 1.0)
                            score = (area_ratio * 0.5) + (centering * 0.3) + (shape_score * 0.2)
                            candidate = Detection(
                                outer=order_quad(outer_quad),
                                inner=order_quad(inner_quad),
                                bbox=outer_bbox,
                                score=score,
                            )
                            if best_match is None or candidate.score > best_match.score:
                                best_match = candidate

        if best_match is not None:
            detections.append(best_match)

    detections.sort(key=lambda item: item.score, reverse=True)
    return detections, mask


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = frame.copy()
    for detection in detections:
        outer = detection.outer.astype(np.int32)
        inner = detection.inner.astype(np.int32)
        cv2.polylines(output, [outer], True, (0, 255, 0), 3)
        cv2.polylines(output, [inner], True, (255, 128, 0), 2)
        x, y, w, h = detection.bbox
        label = f"square-frame {detection.score:.2f}"
        cv2.putText(output, label, (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 180, 255), 2)
    return output


def run_image_mode(args: argparse.Namespace) -> int:
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Cannot read image: {args.image}")
        return 1

    detections, mask = detect_square_frames(
        frame,
        min_area=args.min_area,
        aspect_tolerance=args.aspect_tolerance,
        epsilon_factor=args.approx_epsilon,
        block_size=args.threshold_block_size,
        threshold_c=args.threshold_c,
    )
    annotated = draw_detections(frame, detections)
    cv2.imshow("input", annotated)
    cv2.imshow("mask", mask)
    print(f"Detected {len(detections)} square frame(s). Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def run_camera_mode(args: argparse.Namespace) -> int:
    capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not capture.isOpened():
        print("Cannot open camera. Check camera connection and camera index.")
        return 1

    prev_time = time.time()

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Failed to grab frame from camera.")
            break

        detections, mask = detect_square_frames(
            frame,
            min_area=args.min_area,
            aspect_tolerance=args.aspect_tolerance,
            epsilon_factor=args.approx_epsilon,
            block_size=args.threshold_block_size,
            threshold_c=args.threshold_c,
        )
        annotated = draw_detections(frame, detections)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(
            annotated,
            "Press q to quit",
            (10, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("square-frame-detector", annotated)
        cv2.imshow("square-frame-mask", mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    if args.image:
        return run_image_mode(args)
    return run_camera_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
