const PI = 3.14159265358979323846;
const PHI = 0x9e3779b9u;

const SAMPLES = 1000u;
const T = 1000u;
const T_early = 200u;
const tau = 1e-4;

const N = 50u;
const WORKGROUP_SIZE = 50u;
const dm = 1e-3 / f32(N);
// const dm = 1e-4;
const d = 12u;
const L = 6.0;
const B = 0.0;

const q = 1.0;
const A = 0.5;
const max_dF = 2.0;

@group(0) @binding(0) var<storage, read_write> output_x: array<f32, SAMPLES * d>;
@group(0) @binding(1) var<storage, read_write> output_f: array<f32, SAMPLES>;
@group(0) @binding(2) var<storage, read_write> output_n: array<u32, SAMPLES>;

var<workgroup> inactive: array<u32, N>;
var<workgroup> ff: array<f32, N>;
var<workgroup> g_val: f32;
var<workgroup> f_avg: f32;
var<workgroup> best_x: array<f32, d>;
var<workgroup> best_f: f32;
var<workgroup> best_n: u32;

fn rastrigin(x: array<f32, d>) -> f32 {
    var F_value = 10.0 * f32(d);
    for (var i = 0u; i < d; i++) {
        F_value += x[i] * x[i] - 10.0 * cos(2.0 * PI * x[i]);
    }
    return F_value;
}

fn dRastrigin(x: array<f32, d>) -> array<f32, d> {
    var dF_value: array<f32, d>;
    for (var i = 0u; i < d; i++) {
        dF_value[i] = 2.0 * x[i] + 20.0 * PI * sin(2.0 * PI * x[i]);
    }
    return dF_value;
}

fn norm_sq(x: array<f32, d>) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < d; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

fn ackley(x: array<f32, d>) -> f32 {
    var a = 0.0;
    for (var k = 0u; k < d; k++) {
        a += cos(2.0 * PI * x[k]);
    }
    var F_value = 20.0 + exp(1.0) - 20.0 * exp(-0.2 * sqrt(norm_sq(x) / f32(d))) - exp(a / f32(d));
    return F_value;
}

fn dAckley(x: array<f32, d>) -> array<f32, d> {
    var dF_value: array<f32, d>;
    var a = 0.0;
    for (var k = 0u; k < d; k++) {
        a += cos(2.0 * PI * x[k]);
    }
    var b = sqrt(norm_sq(x) / f32(d));
    for (var k = 0u; k < d; k++) {
        dF_value[k] = (4.0/f32(d)) * x[k] * exp(-0.2 * b) / b + (2.0 * PI / f32(d)) * exp(a / f32(d)) * sin(2.0 * PI * x[k]);
    }
    return dF_value;
}

fn sphere(x: array<f32, d>) -> f32 {
    var F_value = 0.0;
    for (var i = 0u; i < d; i++) {
        F_value += x[i] * x[i];
    }
    return F_value;
}

fn dSphere(x: array<f32, d>) -> array<f32, d> {
    var dF_value: array<f32, d>;
    for (var i = 0u; i < d; i++) {
        dF_value[i] = 2.0 * x[i];
    }
    return dF_value;
}

fn F(x: array<f32, d>) -> f32 {
    var shifted_x: array<f32, d>;
    for (var i = 0u; i < d; i++) {
        shifted_x[i] = x[i] + B;
    }
    return ackley(shifted_x);
}

fn dF(x: array<f32, d>) -> array<f32, d> {
    var shifted_x: array<f32, d>;
    for (var i = 0u; i < d; i++) {
        shifted_x[i] = x[i] + B;
    }
    return dAckley(shifted_x);
}

fn rand(x: u32, min_x: f32, max_x: f32) -> f32 {
    // PCG (Permuted Congruential Generator)
    var state = x;
    
    state = state ^ (state >> 15u);
    state = state * 0x85ebca6bu;
    state = state ^ (state >> 13u);
    state = state * 0xc2b2ae35u;
    state = state ^ (state >> 16u);
    
    state = state * 0x853c49e3u + 0xda3e39cb;
    
    let rot = state >> 28u;
    state = ((state >> 22u) ^ state) >> ((rot & 0x3u));
    
    let normalized = f32(state) / 4294967296.0;
    
    return min_x + (max_x - min_x) * normalized;
}

fn gaussian(seed: u32, mean: f32, variance: f32) -> f32 {
    let u1 = rand(seed, 0.0, 1.0);
    let u2 = rand(seed * PHI, 0.0, 1.0);
    
    let r = sqrt(-2.0 * log(max(1e-6, u1)));
    let theta = 2.0 * PI * u2;
    
    let standard = r * cos(theta);
    return standard * sqrt(variance) + mean;
}

fn random_direction(seed: u32) -> array<f32, d> {
    var dir: array<f32, d>;
    var sum = 0.0;
    for (var k = 0u; k < d; k++) {
        dir[k] = gaussian(seed ^ (k * 16777619u), 0.0, 1.0);
        sum += dir[k] * dir[k];
    }
    if (sum > 0.0) {
        sum = sqrt(sum);
        for (var k = 0u; k < d; k++) {
            dir[k] /= sum;
        }
    }
    return dir;
}

fn sigma2(m: f32) -> f32 {
    return max(0.0, (1.0 / pow(m, q)) - 1.0);
}

fn omega(m: f32, seed: u32) -> array<f32, d> {
    var omega_value: array<f32, d>;
    let dir = random_direction(seed);
    let scale_seed = (seed << 13u) | (seed >> 19u);
    let scale = A * max_dF * gaussian(scale_seed, 0.0, sigma2(m));
    for (var k = 0u; k < d; k++) {
        omega_value[k] = dir[k] * scale;
    }
    return omega_value;
}

fn f(m: f32) -> f32 {
    return sqrt(m) / (1 + sqrt(m));
}

fn get_x(i: u32, sample_id: u32) -> array<f32, d> {
    var x: array<f32, d>;
    
    var base_seed = sample_id * PHI + i * 0x85ebca77u + 0xc2b2ae3du;
    
    base_seed = base_seed ^ (base_seed >> 15u);
    base_seed = base_seed * 0x85ebca6bu;
    base_seed = base_seed ^ (base_seed >> 13u);
    
    for (var k = 0u; k < d; k++) {
        let dim_seed = base_seed + k * PHI;
        x[k] = rand(dim_seed, -L/2.0, L/2.0);
    }
    return x;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn cs(
    @builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let sample_id = workgroup_id.x;
    let i = local_id.x;
    
    var x: array<f32, d> = get_x(i, sample_id);
    var m = 1.0 / f32(N);
    // var m = 0.1;
    var f = F(x);
    var df = dF(x);
    inactive[i] = u32(i >= N);

    if (i == 0u) {
        best_f = f;
        best_n = 0u;
        for (var k = 0u; k < d; k++) {
            best_x[k] = x[k];
        }
    }

    for (var n = 0u; n < T; n++) {
        if (inactive[i] != 1u) {
            if (n > 0u) {
                let omega_seed = n * PHI + i * 0x85ebca77u + sample_id * 0xc2b2ae3du;
                let mixed_seed = omega_seed ^ (omega_seed >> 16u);
                
                let omega_value = omega(m, mixed_seed);
                for (var k = 0u; k < d; k++) {
                    if (m > dm) {
                        x[k] = x[k] + (tau/m) * (-df[k] + omega_value[k]);
                    }
                }
                f = F(x);
                df = dF(x);
            }
            ff[i] = f;
        }

        workgroupBarrier();

        if (i == 0u) {
            var f_max = f;
            var f_min = f;
            var f_sum = 0.0;
            var N_active = 0u;
            for (var j = 0u; j < N; j++) {
                if (inactive[j] != 1u) {
                    let fj = ff[j];
                    f_max = max(f_max, fj);
                    f_min = min(f_min, fj);
                    f_sum += fj;
                    N_active += 1u;
                }
            }
            if (f_min < best_f) {
                best_f = f_min;
                best_n = n;
            }
            if (N_active > 0u) {
                if (f_max - f_min > 1e-6) {
                    g_val = f32(N_active) / (f_max - f_min);
                } else {
                    g_val = 0.0;
                }
                f_avg = f_sum / f32(N_active);
            }
        }

        workgroupBarrier();

        if (inactive[i] != 1u) {
            if ((best_n == n) && (best_f == f)) {
                for (var k = 0u; k < d; k++) {
                    best_x[k] = x[k];
                }
            } else if ((n - best_n) > T_early) {
                inactive[i] = 1u;
                // break;
            }

            var delta = f(m) * g_val * (f - f_avg);
            m = m - delta * tau;

            if (m < dm) {
                inactive[i] = 1u;
            }
        }
    }

    if (i == 0u) {
        for (var k = 0u; k < d; k++) {
            output_x[sample_id * d + k] = best_x[k] + B;
        }
        output_f[sample_id] = best_f;
        output_n[sample_id] = best_n;
    }
}
