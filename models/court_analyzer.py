import numpy as np
import cv2


def extract_geometric_points(polygon_points):
    """從多邊形頂點中提取幾何點，但不進行繪製"""
    polygon_points = np.array(polygon_points)

    # 計算平均像素位置
    mean_x = np.mean(polygon_points[:, 0])
    mean_y = np.mean(polygon_points[:, 1])

    # Top Right 搜尋邏輯
    top_right_points = polygon_points[(polygon_points[:, 0] > mean_x) &
                                      (polygon_points[:, 1] < mean_y)]
    if len(top_right_points) > 0:
        top_right = top_right_points[np.argsort(top_right_points[:, 1])[:3]]
        top_right = top_right[np.argmax(top_right[:, 0])]
    else:
        top_right = polygon_points[np.argmin(polygon_points[:, 0] - polygon_points[:, 1])]

    # 基本點位提取
    top_left = polygon_points[np.argmin(polygon_points[:, 0] + polygon_points[:, 1])]
    bottom_left = polygon_points[np.argmax(polygon_points[:, 1] - polygon_points[:, 0])]
    bottom_right = polygon_points[np.argmax(polygon_points[:, 1] + polygon_points[:, 0])]

    return {
        "RA": {
            "points": [top_left, bottom_left, top_right, bottom_right],
            "labels": ["top_left", "bottom_left", "top_right", "bottom_right"]
        }
    }


def extract_geometric_points_ra(polygon_points):
    """僅提取禁區點，但不進行繪製"""
    points = extract_geometric_points(polygon_points)
    return {"RA": points["RA"]}