use math::matrix::Vector3d;

use crate::skeleton::inst::*;
use crate::skeleton::tree::*;

/*
Fits a skeleton's joints to its markers' tracked positions, once per frame,
starting from the last frame's joints.

A small dense Levenberg-Marquardt of its own, in place of vision::solver's
NonLinearSolver, which on this problem (riftcv1-mocap's rift_solve_bench, a
walking skeleton seen by 12 cameras) took a median 3.9 ms and drifted 20-50 mm
off its markers: its damping scales only the diagonal of J^T J, so when a joint
direction is momentarily unobservable (a straight elbow hides shoulder twist)
the system went singular, steps were refused and all 100 iterations ran
without moving; it also kept iterating after converging.

Here:
- the residual is Dennis's: each marker's target position less its position
  from forward kinematics (the same transforms as SkeletonTree's, done here
  without allocation);
- the Jacobian is analytic: a joint's rotation component i moves every marker
  below it by (A J(w) e_i) x (x - c), A the joint's parent rotation, J the
  rotation's left Jacobian, c the joint's pivot;
- the damping adds lambda * max(diag, floor), so singular directions stay
  well conditioned and simply don't move;
- a parameter at a joint limit that the step would push further out is held
  for that step (the limits are clamped after each step, as before), and an
  unconstrained axis (-pi to pi) is left unbounded;
- it stops when the fit is within TARGET_RMS, when an accepted step gains
  less than MIN_RELATIVE_DECREASE, or after MAX_ITERATIONS Jacobians.
*/

const MAX_ITERATIONS: usize = 20;

/// Refused steps (more damping) allowed per Jacobian.
const MAX_RETRIES: usize = 10;

/// Stop once every marker coordinate fits to about this (m, RMS).
const TARGET_RMS: f64 = 2e-5;

const MIN_RELATIVE_DECREASE: f64 = 1e-4;

const LAMBDA_START: f64 = 1e-3;
const LAMBDA_MIN: f64 = 1e-9;
const LAMBDA_MAX: f64 = 1e9;

/// Floor on the damping's diagonal, relative to J^T J's largest diagonal entry.
const DAMPING_FLOOR: f64 = 1e-6;

// TODO: Ideally require some minimum error and confidence to consider the match to be successful.
pub fn solve_skeleton_joints_state(
    skeleton: &Skeleton,
    _skeleton_tree: &SkeletonTree,
    initial_state: &SkeletonJointsState,
    markers: &[(usize, Vector3d)]
) -> SkeletonJointsState {
    if markers.is_empty() {
        return initial_state.clone();
    }

    let mut fk = Kinematics::new(skeleton, markers);
    let n = fk.num_params();
    let m = markers.len() * 3;

    let mut p = initial_state.serialize(skeleton);
    fk.clamp(&mut p);

    let mut r = vec![0.0; m];
    let mut cost = fk.residuals(&p, &mut r);
    let target_cost = (m as f64) * TARGET_RMS * TARGET_RMS;

    let mut jac = vec![0.0; m * n];     // row-major: jac[i * n + k] = d(position i) / d(param k)
    let mut a = vec![0.0; n * n];
    let mut g = vec![0.0; n];
    let mut idx = Vec::with_capacity(n);
    let mut chol = vec![0.0; n * n];
    let mut step = vec![0.0; n];
    let mut p_new = vec![0.0; n];
    let mut r_new = vec![0.0; m];
    let mut lambda = LAMBDA_START;

    for _ in 0..MAX_ITERATIONS {
        if cost <= target_cost {
            break;
        }

        fk.jacobian(&p, &mut jac);

        // g = J^T r (the direction that lowers r), and which parameters may move.
        idx.clear();
        for k in 0..n {
            let mut s = 0.0;
            for i in 0..m {
                s += jac[i * n + k] * r[i];
            }
            g[k] = s;
            let (lo, hi) = fk.bounds[k];
            if !((p[k] <= lo && s < 0.0) || (p[k] >= hi && s > 0.0)) {
                idx.push(k);
            }
        }
        let nf = idx.len();
        if nf == 0 {
            break;
        }

        // A = J^T J over the free parameters.
        let mut max_diag: f64 = 0.0;
        for u in 0..nf {
            let ku = idx[u];
            for v in 0..=u {
                let kv = idx[v];
                let mut s = 0.0;
                for i in 0..m {
                    s += jac[i * n + ku] * jac[i * n + kv];
                }
                a[u * nf + v] = s;
                a[v * nf + u] = s;
            }
            max_diag = max_diag.max(a[u * nf + u]);
        }
        let floor = (max_diag * DAMPING_FLOOR).max(1e-12);

        let mut accepted = false;
        let mut converged = false;
        for _ in 0..MAX_RETRIES {
            // (A + lambda * D) step = g
            chol[..nf * nf].copy_from_slice(&a[..nf * nf]);
            for u in 0..nf {
                chol[u * nf + u] += lambda * a[u * nf + u].max(floor);
            }
            if !cholesky_solve(&mut chol[..nf * nf], nf, &idx, &g, &mut step[..nf]) {
                lambda *= 4.0;
                if lambda > LAMBDA_MAX {
                    break;
                }
                continue;
            }

            p_new.copy_from_slice(&p);
            let mut max_step: f64 = 0.0;
            for (u, &k) in idx.iter().enumerate() {
                p_new[k] += step[u];
                max_step = max_step.max(step[u].abs());
            }
            fk.clamp(&mut p_new);

            let new_cost = fk.residuals(&p_new, &mut r_new);
            if new_cost < cost {
                let gain = (cost - new_cost) / cost;
                core::mem::swap(&mut p, &mut p_new);
                core::mem::swap(&mut r, &mut r_new);
                cost = new_cost;
                lambda = (lambda / 3.0).max(LAMBDA_MIN);
                accepted = true;
                converged = gain < MIN_RELATIVE_DECREASE || max_step < 1e-9;
                break;
            }

            lambda *= 4.0;
            if lambda > LAMBDA_MAX {
                break;
            }
        }

        if !accepted || converged {
            break;
        }
    }

    // Not SkeletonJointsState::clamp: that would put unconstrained axes back
    // inside +-pi, and the next frame would start from a different rotation.
    SkeletonJointsState::parse(&p, skeleton)
}

/// Solves (L L^T) x = b in place, b gathered from g by idx. False if the
/// matrix isn't positive definite.
fn cholesky_solve(a: &mut [f64], n: usize, idx: &[usize], g: &[f64], x: &mut [f64]) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !(d > 0.0) {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    for i in 0..n {
        let mut s = g[idx[i]];
        for k in 0..i {
            s -= a[i * n + k] * x[k];
        }
        x[i] = s / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = x[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * x[k];
        }
        x[i] = s / a[i * n + i];
    }
    true
}

type V3 = [f64; 3];
type M3 = [[f64; 3]; 3];

fn mul(a: &M3, b: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn apply(a: &M3, v: &V3) -> V3 {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn cross(a: &V3, b: &V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// I + c1 [w]x + c2 [w]x^2.
fn skew_poly(w: &V3, c1: f64, c2: f64) -> M3 {
    let k = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]];
    let k2 = mul(&k, &k);
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = if i == j { 1.0 } else { 0.0 } + c1 * k[i][j] + c2 * k2[i][j];
        }
    }
    out
}

/// The rotation for axis-angle w (Rodrigues, as math's from_axis_angle).
fn rotation(w: &V3) -> M3 {
    let t2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    if t2 < 1e-12 {
        return skew_poly(w, 1.0, 0.5);
    }
    let t = t2.sqrt();
    skew_poly(w, t.sin() / t, (1.0 - t.cos()) / t2)
}

/// The left Jacobian of the rotation: d(R(w)) / dw_i R(w)^T = [J e_i]x.
fn left_jacobian(w: &V3) -> M3 {
    let t2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    if t2 < 1e-12 {
        return skew_poly(w, 0.5, 1.0 / 6.0);
    }
    let t = t2.sqrt();
    skew_poly(w, (1.0 - t.cos()) / t2, (t - t.sin()) / (t2 * t))
}

#[derive(Clone, Copy)]
enum Param {
    Translation(usize),
    Rotation(usize),
    Bone(usize, usize),
}

/// Forward kinematics and its Jacobian on a flat parameter vector
/// (SkeletonJointsState's serialized form), with no allocation per evaluation.
/// Bone b's rotation turns everything below it about its end point; the
/// markers on bone b move with b's parent (as in SkeletonTree).
struct Kinematics<'a> {
    targets: &'a [(usize, Vector3d)],
    params: Vec<Param>,
    bounds: Vec<(f64, f64)>,
    /// Bones with every parent before its children.
    order: Vec<usize>,
    parent: Vec<Option<usize>>,
    end: Vec<V3>,
    /// For each target: its bone and the marker's model position.
    target_bone: Vec<usize>,
    target_model: Vec<V3>,
    /// For each bone, whether it lies strictly above each target's bone.
    above: Vec<Vec<bool>>,

    root_t: V3,
    root_w: V3,
    bone_w: Vec<V3>,
    /// Each bone's parent transform (the one its markers and pivot use).
    parent_rot: Vec<M3>,
    parent_t: Vec<V3>,
    positions: Vec<V3>,
}

impl<'a> Kinematics<'a> {
    fn new(skeleton: &Skeleton, targets: &'a [(usize, Vector3d)]) -> Self {
        let nb = skeleton.bones.len();
        let mut params = vec![
            Param::Translation(0), Param::Translation(1), Param::Translation(2),
            Param::Rotation(0), Param::Rotation(1), Param::Rotation(2),
        ];
        let mut bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 6];
        for (b, bone) in skeleton.bones.iter().enumerate() {
            for i in 0..3 {
                let c = &bone.rotation_constraints[i];
                if c.is_fixed() {
                    continue;
                }
                params.push(Param::Bone(b, i));
                // An unconstrained axis (-pi to pi) is left unbounded: a component
                // clamped at +-pi traps the fit when it wanders onto the other
                // branch of an ambiguous joint, and any value is a valid rotation.
                if c.min <= -std::f64::consts::PI && c.max >= std::f64::consts::PI {
                    bounds.push((f64::NEG_INFINITY, f64::INFINITY));
                } else {
                    bounds.push((c.min, c.max));
                }
            }
        }

        let parent: Vec<Option<usize>> = skeleton.bones.iter().map(|b| b.parent).collect();
        let mut order = vec![];
        let mut placed = vec![false; nb];
        while order.len() < nb {
            for b in 0..nb {
                if !placed[b] && parent[b].map_or(true, |p| placed[p]) {
                    placed[b] = true;
                    order.push(b);
                }
            }
        }

        let target_bone: Vec<usize> = targets.iter().map(|(i, _)| skeleton.markers[*i].bone_index).collect();
        let target_model: Vec<V3> = targets.iter().map(|(i, _)| {
            let p = &skeleton.markers[*i].position;
            [p[0], p[1], p[2]]
        }).collect();
        let mut above = vec![vec![false; targets.len()]; nb];
        for (j, &tb) in target_bone.iter().enumerate() {
            let mut b = parent[tb];
            while let Some(x) = b {
                above[x][j] = true;
                b = parent[x];
            }
        }

        Self {
            targets,
            params,
            bounds,
            order,
            parent,
            end: skeleton.bones.iter().map(|b| [b.end_position[0], b.end_position[1], b.end_position[2]]).collect(),
            target_bone,
            target_model,
            above,
            root_t: [0.0; 3],
            root_w: [0.0; 3],
            bone_w: vec![[0.0; 3]; nb],
            parent_rot: vec![[[0.0; 3]; 3]; nb],
            parent_t: vec![[0.0; 3]; nb],
            positions: vec![[0.0; 3]; targets.len()],
        }
    }

    fn num_params(&self) -> usize {
        self.params.len()
    }

    fn clamp(&self, p: &mut [f64]) {
        for (v, (lo, hi)) in p.iter_mut().zip(&self.bounds) {
            *v = v.max(*lo).min(*hi);
        }
    }

    fn forward(&mut self, p: &[f64]) {
        for (k, param) in self.params.iter().enumerate() {
            match *param {
                Param::Translation(i) => self.root_t[i] = p[k],
                Param::Rotation(i) => self.root_w[i] = p[k],
                Param::Bone(b, i) => self.bone_w[b][i] = p[k],
            }
        }

        let root_rot = rotation(&self.root_w);
        for &b in &self.order {
            let (pr, pt) = match self.parent[b] {
                None => (root_rot, self.root_t),
                Some(q) => {
                    // q's children transform: q's parent transform, then q's
                    // rotation about q's end point.
                    let rq = rotation(&self.bone_w[q]);
                    let e = &self.end[q];
                    let re = apply(&rq, e);
                    let local_t = [e[0] - re[0], e[1] - re[1], e[2] - re[2]];
                    let a = &self.parent_rot[q];
                    let t = apply(a, &local_t);
                    (mul(a, &rq), [t[0] + self.parent_t[q][0], t[1] + self.parent_t[q][1], t[2] + self.parent_t[q][2]])
                }
            };
            self.parent_rot[b] = pr;
            self.parent_t[b] = pt;
        }

        for j in 0..self.target_model.len() {
            let b = self.target_bone[j];
            let x = apply(&self.parent_rot[b], &self.target_model[j]);
            let t = &self.parent_t[b];
            self.positions[j] = [x[0] + t[0], x[1] + t[1], x[2] + t[2]];
        }
    }

    /// r = target - position for every marker; returns the sum of squares.
    fn residuals(&mut self, p: &[f64], r: &mut [f64]) -> f64 {
        self.forward(p);
        let mut sum = 0.0;
        for (j, (_, target)) in self.targets.iter().enumerate() {
            for i in 0..3 {
                let e = target[i] - self.positions[j][i];
                r[j * 3 + i] = e;
                sum += e * e;
            }
        }
        sum
    }

    /// jac[(3j + i) * n + k] = d(position j, coordinate i) / d(param k), at p.
    fn jacobian(&mut self, p: &[f64], jac: &mut [f64]) {
        self.forward(p);
        let n = self.params.len();
        let nt = self.positions.len();
        jac.iter_mut().for_each(|v| *v = 0.0);

        let root_j = left_jacobian(&self.root_w);
        let mut bone_j: Option<(usize, M3)> = None;

        for (k, param) in self.params.iter().enumerate() {
            match *param {
                Param::Translation(i) => {
                    for j in 0..nt {
                        jac[(3 * j + i) * n + k] = 1.0;
                    }
                }
                Param::Rotation(i) => {
                    let axis = [root_j[0][i], root_j[1][i], root_j[2][i]];
                    for j in 0..nt {
                        let x = &self.positions[j];
                        let d = cross(&axis, &[x[0] - self.root_t[0], x[1] - self.root_t[1], x[2] - self.root_t[2]]);
                        for c in 0..3 {
                            jac[(3 * j + c) * n + k] = d[c];
                        }
                    }
                }
                Param::Bone(b, i) => {
                    // A J e_i, with the pivot at b's end point in the world.
                    let jb = match &bone_j {
                        Some((bb, m)) if *bb == b => *m,
                        _ => {
                            let m = mul(&self.parent_rot[b], &left_jacobian(&self.bone_w[b]));
                            bone_j = Some((b, m));
                            m
                        }
                    };
                    let axis = [jb[0][i], jb[1][i], jb[2][i]];
                    let e = apply(&self.parent_rot[b], &self.end[b]);
                    let c = [e[0] + self.parent_t[b][0], e[1] + self.parent_t[b][1], e[2] + self.parent_t[b][2]];
                    for j in 0..nt {
                        if !self.above[b][j] {
                            continue;
                        }
                        let x = &self.positions[j];
                        let d = cross(&axis, &[x[0] - c[0], x[1] - c[1], x[2] - c[2]]);
                        for cc in 0..3 {
                            jac[(3 * j + cc) * n + k] = d[cc];
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::standard_skeleton;

    fn some_state(skeleton: &Skeleton, seed: f64) -> SkeletonJointsState {
        let mut s = SkeletonJointsState::zero(skeleton);
        s.translation = Vector3d::from_slice(&[0.3 * seed.sin(), -0.2, 0.95]);
        s.rotation = Vector3d::from_slice(&[0.3, -0.5 * seed, 2.0 + seed]);
        for (b, bone) in skeleton.bones.iter().enumerate() {
            for i in 0..3 {
                let c = &bone.rotation_constraints[i];
                if !c.is_fixed() {
                    let mid = 0.5 * (c.min.max(-1.5) + c.max.min(1.5));
                    s.bone_rotations[b][i] = mid + 0.4 * ((b * 3 + i) as f64 + seed).sin();
                }
            }
        }
        s
    }

    #[test]
    fn forward_matches_the_tree_and_jacobian_matches_differences() {
        let skeleton = standard_skeleton();
        let tree = SkeletonTree::new(&skeleton);
        for seed in [0.0, 0.7, 2.9] {
            let state = some_state(&skeleton, seed);
            let mut truth = vec![Vector3d::zero(); skeleton.markers.len()];
            tree.forward_kinematics(&state, &mut [], &mut truth);

            let targets: Vec<(usize, Vector3d)> = truth.iter().cloned().enumerate().collect();
            let mut fk = Kinematics::new(&skeleton, &targets);
            let p = state.serialize(&skeleton);
            let m = targets.len() * 3;
            let mut r = vec![0.0; m];
            assert!(fk.residuals(&p, &mut r) < 1e-20);

            let n = fk.num_params();
            let mut jac = vec![0.0; m * n];
            fk.jacobian(&p, &mut jac);
            let h = 1e-6;
            let mut r2 = vec![0.0; m];
            for k in 0..n {
                let mut q = p.clone();
                q[k] += h;
                fk.residuals(&q, &mut r2);
                q[k] -= 2.0 * h;
                let mut r3 = vec![0.0; m];
                fk.residuals(&q, &mut r3);
                for i in 0..m {
                    let numeric = (r3[i] - r2[i]) / (2.0 * h);
                    assert!((numeric - jac[i * n + k]).abs() < 1e-6,
                            "param {} row {}: {} against {}", k, i, jac[i * n + k], numeric);
                }
            }
        }
    }
}
