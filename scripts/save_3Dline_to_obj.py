import sys
import os

#!/usr/bin/env python3

def read_lines(txt_file):
    """
    从 txt 文件中读取线段，每行应包含6个数字，数字之间以空格分隔。
    每行的前3个数字为起点，后3个数字为终点。
    可以跳过空行或以 '#' 开头的注释行。
    """
    vertices = []
    segments = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 6:
                continue
            try:
                nums = list(map(float, parts))
            except ValueError:
                continue
            # 记录两个顶点的位置，下标为后面生成顶点的序号(OBJ中编号从1开始)
            start_idx = len(vertices) + 1
            vertices.append((nums[0], nums[1], nums[2]))
            vertices.append((nums[3], nums[4], nums[5]))
            segments.append((start_idx, start_idx + 1))
    return vertices, segments

def write_obj(obj_file, vertices, segments):
    """
    将顶点和线段写入 OBJ 文件中:
    每个点作为一个顶点 "v x y z"；
    每条线段为一行 "l start_idx end_idx"。
    """
    with open(obj_file, 'w') as f:
        for pt in vertices:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(pt[0], pt[1], pt[2]))
        for seg in segments:
            f.write("l {} {}\n".format(seg[0], seg[1]))

def read_obj(obj_file):
    """
    从 OBJ 文件中读取顶点和线段：
    每行以 "v " 开头的记录顶点数据；
    每行以 "l " 开头的记录线段，顶点编号从1开始。
    """
    vertices = []
    segments = []
    with open(obj_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                vertices.append((x, y, z))
            elif line.startswith('l '):
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    idx1, idx2 = int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                segments.append((idx1, idx2))
    return vertices, segments

def write_txt(txt_file, vertices, segments):
    """
    将线段数据写入 TXT 文件中，每行6个数字，前3个为起点，后3个为终点。
    OBJ中顶点编号从1开始，故 vertices 列表索引为编号-1。
    """
    with open(txt_file, 'w') as f:
        for seg in segments:
            idx1, idx2 = seg
            # 检查索引合法性
            if idx1-1 < 0 or idx1-1 >= len(vertices) or idx2-1 < 0 or idx2-1 >= len(vertices):
                continue
            v1 = vertices[idx1-1]
            v2 = vertices[idx2-1]
            f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]
            ))

def main():
    if len(sys.argv) != 3:
        print("Usage: {} input_file output_file".format(sys.argv[0]))
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    _, in_ext = os.path.splitext(input_file)
    _, out_ext = os.path.splitext(output_file)
    in_ext = in_ext.lower()
    out_ext = out_ext.lower()

    if in_ext == ".txt" and out_ext == ".obj":
        vertices, segments = read_lines(input_file)
        if not vertices:
            print("没有读取到有效的线段数据。")
            sys.exit(1)
        write_obj(output_file, vertices, segments)
        print("OBJ文件已保存到", output_file)
    elif in_ext == ".obj" and out_ext == ".txt":
        vertices, segments = read_obj(input_file)
        if not vertices:
            print("OBJ文件中没有读取到有效数据。")
            sys.exit(1)
        write_txt(output_file, vertices, segments)
        print("TXT文件已保存到", output_file)
    else:
        print("不支持的转换方式，请确保输入输出文件格式为 '.txt' -> '.obj' 或 '.obj' -> '.txt'")
        sys.exit(1)

if __name__ == '__main__':
    main()