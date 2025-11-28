import numpy as np
import matplotlib.pyplot as plt


def normalize_coordinates(coordinates):
    """归一化坐标到[0, 1]范围内"""
    min_x, min_y = np.min(coordinates, axis=0)
    max_x, max_y = np.max(coordinates, axis=0)
    normalized_coords = np.zeros_like(coordinates)
    if max_x - min_x != 0:
        normalized_coords[:, 0] = (coordinates[:, 0] - min_x) / (max_x - min_x)
        normalized_coords[:, 1] = (coordinates[:, 1] - min_y) / (max_y - min_y)
    return normalized_coords


def linear_map(x, M):
    """线性映射函数，将[0, 1]范围内的值映射到[0, M-1]范围内"""
    return int(x * (M - 1))


def bresenham_line(x0, y0, x1, y1, grid):
    """Bresenham线段算法，绘制线段"""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy  # error value e_xy
    while True:
        # Set the pixel
        grid[y0, x0] = 1
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def migrate_points(N, M, coordinates, adjacency_matrix, R):
    # 归一化坐标
    normalized_coordinates = normalize_coordinates(coordinates)

    # 初始化M×M的网格
    grid = np.zeros((M, M), dtype=int)

    # 将坐标映射到M×M的网格中
    mapped_coordinates = [(linear_map(x, M), linear_map(y, M)) for x, y in normalized_coordinates]

    # 设置点周围的像素值
    for (x, y) in mapped_coordinates:
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < M and 0 <= ny < M:
                    grid[ny, nx] = 1

    # 设置连接点的线
    for i in range(N):
        for j in range(i + 1, N):  # 避免重复绘制线段
            if adjacency_matrix[i][j] == 1 and mapped_coordinates[i]!=(0,0) and mapped_coordinates[j]!=(0,0):
                x0, y0 = mapped_coordinates[i]
                x1, y1 = mapped_coordinates[j]
                bresenham_line(x0, y0, x1, y1, grid)

    return grid


def migrate_batch_points(batch_input, A, M, R=1):
    """

    Args:
        batch_input:  T, N, C
        A:  N, N
        M: the size of grid

    Returns:

    """
    T, N, C = batch_input.shape
    Grid = []
    for i in range(T):
        cor = batch_input[i]
        grid_single = migrate_points(N, M, cor, A, R)
        Grid.append(grid_single)
        # if ~(np.any(cor==0)):
        # # 使用matplotlib可视化并保存图片
        # plt.imshow(grid_single, cmap='Greys', interpolation='none')
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()

    return np.array(Grid)


if __name__ == "__main__":
    # 示例数据
    N = 5  # 点的数量
    M = 100  # 新网格的大小
    coordinates = np.random.rand(N, 2) * 10  # 随机生成N个点的坐标，范围为[0, 10)
    adjacency_matrix = np.random.randint(0, 2, (N, N))  # 随机生成邻接矩阵
    R = 2  # 半径

    # 迁移点
    grid = migrate_points(N, M, coordinates, adjacency_matrix, R)

    # 使用matplotlib可视化并保存图片
    plt.imshow(grid, cmap='Greys', interpolation='none')
    plt.axis('off')  # 关闭坐标轴
    plt.savefig('migrated_grid.png', bbox_inches='tight', pad_inches=0)
    plt.show()

