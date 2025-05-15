#!/usr/bin/env python3
"""
line_visualization_steps_fixed_with_jitter.py

Adds small random jitter to avoid complete overlap during visualization.
"""
import numpy as np
import open3d as o3d
import random

def load_lines(path):
    """Load 3D line segments from a text file and reshape into an (N, 2, 3) array."""
    return np.loadtxt(path).reshape(-1, 2, 3)

def apply_transform(lines, T):
    """
    Apply a homogeneous transformation T (4×4) to each 3D line.
    lines: (N, 2, 3) array of segment endpoints.
    Returns transformed (N, 2, 3) array.
    """
    # Convert to homogeneous coordinates
    lines_h = np.concatenate([lines, np.ones((len(lines), 2, 1))], axis=2)
    # Apply transform and drop homogeneous coordinate
    out = (T @ lines_h.transpose(0, 2, 1)).transpose(0, 2, 1)
    return out[..., :3]

def ransac_line_registration(src, tgt, iters=500, thresh=0.02):
    """
    Estimate the best rigid transform aligning src→tgt using RANSAC + Kabsch.
    src, tgt: (N, 2, 3) arrays of corresponding lines.
    iters: number of RANSAC iterations.
    thresh: inlier distance threshold.
    Returns a 4×4 transformation matrix.
    """
    def dirs_mids(lines):
        """Compute direction unit vectors and midpoints for each line."""
        d = lines[:,1] - lines[:,0]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        m = (lines[:,0] + lines[:,1]) / 2
        return d, m

    def estimate_kabsch(A, B):
        """
        Compute rotation R that best aligns A→B via the Kabsch algorithm.
        A, B: (k, 3) arrays of direction vectors.
        """
        H = A.T @ B
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Ensure a proper rotation (determinant = +1)
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        return R

    sd, sm = dirs_mids(src)
    td, tm = dirs_mids(tgt)
    best_T = np.eye(4)
    best_inliers = []
    N = len(src)

    for _ in range(iters):
        # Randomly sample 3 line correspondences
        idx = random.sample(range(N), 3)
        R = estimate_kabsch(sd[idx], td[idx])
        t = tm[idx].mean(0) - (R @ sm[idx].T).mean(1)
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = t

        # Count inliers
        inliers = []
        for i in range(N):
            p0, p1 = src[i]
            p0t = R @ p0 + t
            p1t = R @ p1 + t
            d_norm = (p1t - p0t) / np.linalg.norm(p1t - p0t)
            dist = np.linalg.norm(np.cross(d_norm, tgt[i][0] - p0t))
            if dist < thresh:
                inliers.append(i)

        # Update best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_T = T

    # Re-estimate using all inliers
    if best_inliers:
        Rf = estimate_kabsch(sd[best_inliers], td[best_inliers])
        tf = tm[best_inliers].mean(0) - (Rf @ sm[best_inliers].T).mean(1)
        Tf = np.eye(4); Tf[:3,:3] = Rf; Tf[:3,3] = tf
        best_T = Tf

    return best_T

def jitter(lines, sigma=0.005):
    """
    Add Gaussian noise to each point.
    lines: (N, 2, 3) array
    sigma: standard deviation of noise
    """
    return lines + np.random.normal(scale=sigma, size=lines.shape)

def create_line_set(lines, color):
    """
    Build an Open3D LineSet from an (N,2,3) array.
    color: [r, g, b] in [0,1]
    """
    pts = lines.reshape(-1, 3)
    idx = [[i*2, i*2+1] for i in range(len(lines))]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(idx)
    )
    ls.paint_uniform_color(color)
    return ls

if __name__ == '__main__':
    # Load original and target line sets
    txt1 = load_lines('line_3d.txt')
    txt2 = load_lines('line_3d_cut.txt')

    # Step 1: jitter both sets and visualize
    print("Step 1: Displaying jittered original (gray) and target (green) lines")
    o3d.visualization.draw_geometries([
        create_line_set(jitter(txt1), [0.5, 0.5, 0.5]),
        create_line_set(jitter(txt2), [0, 1, 0])
    ], window_name='Step 1', width=800, height=600)

    # Randomly transform txt2 → txt3
    ang = np.deg2rad(random.uniform(-45, 45))
    R_rand = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, ang))
    t_rand = np.array([random.uniform(-0.2, 0.2) for _ in range(3)])
    T_rand = np.eye(4); T_rand[:3, :3] = R_rand; T_rand[:3, 3] = t_rand
    txt3 = apply_transform(txt2, T_rand)

    # Step 2: visualize txt3 (red) against txt1 (gray)
    print("Step 2: Displaying jittered original (gray) and randomly transformed (red) lines")
    o3d.visualization.draw_geometries([
        create_line_set(jitter(txt1), [0.5, 0.5, 0.5]),
        create_line_set(jitter(txt3), [1, 0, 0])
    ], window_name='Step 2', width=800, height=600)

    # Step 3: estimate registration, generate txt4, visualize
    T_est = ransac_line_registration(txt3, txt1)
    txt4 = apply_transform(txt2, T_est)
    print("Step 3: Displaying registration result—gray (orig), red (transf), blue (estimated)")
    o3d.visualization.draw_geometries([
        create_line_set(jitter(txt1), [0.5, 0.5, 0.5]),
        create_line_set(jitter(txt3), [1, 0, 0]),
        create_line_set(jitter(txt4), [0, 0, 1])
    ], window_name='Step 3', width=800, height=600)

    # Step 4: add back original txt2 for final comparison
    print("Step 4: Displaying final comparison—gray, red, blue, green")
    o3d.visualization.draw_geometries([
        create_line_set(jitter(txt1), [0.5, 0.5, 0.5]),
        create_line_set(jitter(txt3), [1, 0, 0]),
        create_line_set(jitter(txt4), [0, 0, 1]),
        create_line_set(jitter(txt2), [0, 1, 0])
    ], window_name='Step 4', width=800, height=600)
