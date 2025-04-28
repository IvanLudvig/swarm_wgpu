const N = 100u;
const d = 2u;
const tau = 0.0001;
const PI = 3.14159265358979323846;
const T = 2000u;
const T_early = 200u;
const dm = 0.001 / f32(N);

@group(0) @binding(0) var<storage, read> xx: array<f32, N * d>;
@group(0) @binding(1) var<storage, read_write> inactive: array<u32, N>;
@group(0) @binding(2) var<storage, read_write> ff: array<f32, N + 2>;

@group(1) @binding(0) var<storage, read_write> best_x: array<f32, d>;
@group(1) @binding(1) var<storage, read_write> best_f: f32;
@group(1) @binding(2) var<storage, read_write> best_n: u32;

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

fn rand(x: f32) -> f32 {
    let sin_val = sin(x * 12.9898);
    return fract(sin_val * 43758.5453);
}

fn box_muller(u1: f32, u2: f32) -> f32 {
    let r = sqrt(-2.0 * log(u1));
    let theta = 2.0 * PI * u2;
    return r * cos(theta);
}

fn gaussian(seed: f32, mean: f32, variance: f32) -> f32 {
    let u1 = rand(seed);
    let u2 = rand(seed + 2.0214);
    let standard = box_muller(u1, u2);
    return standard * sqrt(variance) + mean;
}

fn random_direction(seed: f32) -> array<f32, d> {
    var direction: array<f32, d>;
    var sum = 0.0;
    for (var i = 0u; i < d; i++) {
        direction[i] = rand(seed + f32(i));
        sum += direction[i] * direction[i];
    }
    for (var i = 0u; i < d; i++) {
        direction[i] /= sqrt(sum);
    }
    return direction;
}

fn sigma2(m: f32) -> f32 {
    return max(0.0, (1.0 / m) - 1.0);
}

fn omega(m: f32, seed: f32) -> array<f32, d> {
    var omega_value: array<f32, d>;
    let direction = random_direction(seed);
    // let scale = 33.0 * gaussian(seed + 1.22, 0.0, sigma2(m));
    let scale = 0.0;
    for (var i = 0u; i < d; i++) {
        omega_value[i] = direction[i] * scale;
    }
    return omega_value;
}

fn get_x(i: u32) -> array<f32, d> {
    var x: array<f32, d>;
    for (var j = 0u; j < d; j++) {
        x[j] = xx[i * d + j];
    }
    return x;
}

@compute @workgroup_size(N)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;

    var x: array<f32, d> = get_x(i);
    var m = 1.0 / f32(N);
    var f = F(x);
    var df = dF(x);

    for (var n = 0u; n < T; n++) {
        if (inactive[i] != 1u) {
            if (n > 0u) {
                let omega_value = omega(m, m + f32(d)*f32(n + 13) + x[0]*x[d-1]);
                for (var k = 0u; k < d; k++) {
                    if (m > dm) {
                        x[k] = x[k] - (tau/m) * (df[k] + omega_value[k]);
                    }
                }
                f = F(x);
                df = dF(x);
            }
            ff[i] = f;
        }

        workgroupBarrier();
        storageBarrier();

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
        storageBarrier();

        if (inactive[i] != 1u) {
            if (best_n == n && best_f == f) {
                for (var k = 0u; k < d; k++) {
                    best_x[k] = x[k];
                }
            } else if (best_n - n > T_early) {
                break;
            }

            let g = ff[N];
            let f_avg = ff[N + 1];

            var delta = g * (f - f_avg);
            m = m - delta * tau;

            if (m < dm) {
                inactive[i] = 1u;
            }
        }
    }
}
