const PI = 3.14159265358979323846;

const SAMPLES = 100u;
const T = 2000u;
const T_early = 200u;
const tau = 0.0001;
const dm = 0.001 / f32(N);

const N = 10u;
const d = 2u;
const L = 6.0;

const q = 1.0;
const A = 0.5;
const max_dF = 66.0;

@group(0) @binding(0) var<storage, read_write> output_x: array<f32, SAMPLES * d>;
@group(0) @binding(1) var<storage, read_write> output_f: array<f32, SAMPLES>;
@group(0) @binding(2) var<storage, read_write> output_n: array<u32, SAMPLES>;

var<workgroup> inactive: array<u32, N>;
var<workgroup> ff: array<f32, N + 2>;
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
    return rastrigin(x);
}

fn dF(x: array<f32, d>) -> array<f32, d> {
    return dRastrigin(x);
}

fn rand(x: f32, min_x: f32, max_x: f32) -> f32 {
    var r = fract(x * 0.1031);
    r = r * (r + 33.33);
    r = r * (r + r);
    return fract(r) * (max_x - min_x) + min_x;
}

fn gaussian(seed: f32, mean: f32, variance: f32) -> f32 {
    let u1 = rand(seed, 0.0, 1.0);
    let u2 = rand(seed + 2.0214, 0.0, 1.0);

    let r = sqrt(-2.0 * log(u1));
    let theta = 2.0 * PI * u2;

    let standard = r * cos(theta);
    return standard * sqrt(variance) + mean;
}

fn random_direction(seed: f32) -> array<f32, d> {
    var dir: array<f32, d>;
    var sum = 0.0;
    for (var k = 0u; k < d; k++) {
        dir[k] = rand(seed + 2.13 * f32(k) + sum, -1.0, 1.0);
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

fn omega(m: f32, seed: f32) -> array<f32, d> {
    var omega_value: array<f32, d>;
    let dir = random_direction(seed);
    let scale = A * gaussian(seed + 1.22, 0.0, sigma2(m)) * max_dF;
    for (var k = 0u; k < d; k++) {
        omega_value[k] = dir[k] * scale;
    }
    return omega_value;
}

fn get_x(i: u32, sample_id: u32) -> array<f32, d> {
    var x: array<f32, d>;
    let base_seed = 33.0 + f32(sample_id) * 1000.0 + f32(i) * 10.0;
    
    for (var k = 0u; k < d; k++) {
        x[k] = rand(base_seed + f32(k) * 7.89, -L/2.0, L/2.0);
    }
    return x;
}

@compute @workgroup_size(N)
fn cs(
    @builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let sample_id = workgroup_id.x;
    let i = local_id.x;
    
    var x: array<f32, d> = get_x(i, sample_id);
    var m = 1.0 / f32(N);
    var f = F(x);
    var df = dF(x);
    inactive[i] = 0u;

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
                let omega_value = omega(m, m + f32(d)*f32(n + 13) + x[0]*x[d-1] + 5.43 * f32(sample_id));
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
            var g = 0.0;
            if (f_max - f_min > 1e-6) {
                g = f32(N_active) / (f_max - f_min);
            }
            ff[N] = g;
            ff[N + 1] = f_sum / f32(N_active);
        }

        workgroupBarrier();

        if (inactive[i] != 1u) {
            if (best_n == n && best_f == f) {
                for (var k = 0u; k < d; k++) {
                    best_x[k] = x[k];
                }
            } else if ((n - best_n) > T_early) {
                // inactive[i] = 1u;
                break;
            }

            let g = ff[N];
            let f_avg = ff[N + 1];

            var delta = m * g * (f - f_avg);
            m = m - delta * tau;

            if (m < dm) {
                inactive[i] = 1u;
            }
        }
    }

    if (i == 0u) {
        for (var k = 0u; k < d; k++) {
            output_x[sample_id * d + k] = best_x[k];
        }
        output_f[sample_id] = best_f;
        output_n[sample_id] = best_n;
    }
}
