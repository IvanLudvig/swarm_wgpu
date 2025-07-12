use std::{borrow::Cow, fs};
use wgpu::util::DeviceExt;

const D: u32 = 3;
const SAMPLES: u32 = 1000;
const SUCCESS_THRESHOLD: f32 = 0.1;
const SOLUTION: f32 = 0.0;

// noise parameter: q for Gauss or stability index for Levy
const PARAM1_VALUES: &[f32] = &[0.5, 1.0, 2.0];

// noise scale
const PARAM2_VALUES: &[f32] = &[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2];

// refined search
// const COARSE_C: f32 = 1e-1;
// const PARAM2_VALUES: &[f32] = &[
//     0.25 * COARSE_C,
//     0.5 * COARSE_C,
//     COARSE_C,
//     2.0 * COARSE_C,
//     4.0 * COARSE_C,
//     6.0 * COARSE_C,
//     8.0 * COARSE_C,
// ];
const GRID_SIZE: u32 = (PARAM1_VALUES.len() * PARAM2_VALUES.len()) as u32;

async fn run() {
    let start_time = std::time::Instant::now();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let mut limits = wgpu::Limits::default();
    limits.max_buffer_size = 256 * 4 * 1024 * 1024;
    limits.max_compute_workgroup_storage_size = 16384 * 4;
    limits.max_compute_invocations_per_workgroup = 256 * 4;
    limits.max_compute_workgroup_size_x = 256 * 4;
    limits.max_compute_workgroup_size_y = 256 * 4;
    limits.max_compute_workgroup_size_z = 64 * 4;
    limits.max_storage_buffer_binding_size = 128 * 4 * 1024 * 1024;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            label: None,
            memory_hints: Default::default(),
            trace: Default::default(),
        })
        .await
        .unwrap();

    let shader_src =
        fs::read_to_string("src/shaders/compute.glsl").expect("Failed to read compute shader file");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_src)),
    });

    let best_position = vec![-1.0f32; (GRID_SIZE * SAMPLES * D) as usize];
    let best_fitness = vec![f32::MAX; (GRID_SIZE * SAMPLES) as usize];
    let n = vec![0u32; (GRID_SIZE * SAMPLES) as usize];

    let best_position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Best Position Buffer"),
        contents: bytemuck::cast_slice(&best_position),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let best_fitness_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Best Fitness Buffer"),
        contents: bytemuck::cast_slice(&best_fitness),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let n_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("N Buffer"),
        contents: bytemuck::cast_slice(&n),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    let staging_best_position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Best Position Buffer"),
        size: (GRID_SIZE * SAMPLES * D) as u64 * std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_best_fitness_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Best Fitness Buffer"),
        size: (GRID_SIZE * SAMPLES) as u64 * std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_best_n_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Best N Buffer"),
        size: (GRID_SIZE * SAMPLES) as u64 * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut param_sets = Vec::with_capacity((GRID_SIZE * 4) as usize);
    for &p1 in PARAM1_VALUES {
        for &p2 in PARAM2_VALUES {
            param_sets.push(p1);
            param_sets.push(p2);
            param_sets.push(0.0f32);
            param_sets.push(0.0f32);
        }
    }

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Parameter Buffer"),
        contents: bytemuck::cast_slice(&param_sets),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let output_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("Output Bind Group Layout"),
        });

    let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &output_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: best_position_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: best_fitness_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: n_buffer.as_entire_binding(),
            },
        ],
        label: Some("Output Bind Group"),
    });

    let param_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("Parameter Bind Group Layout"),
        });

    let param_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &param_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: param_buffer.as_entire_binding(),
        }],
        label: Some("Parameter Bind Group"),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&output_bind_group_layout, &param_bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &output_bind_group, &[]);
        compute_pass.set_bind_group(1, &param_bind_group, &[]);
        compute_pass.dispatch_workgroups(SAMPLES, GRID_SIZE, 1);
    }

    encoder.copy_buffer_to_buffer(
        &best_position_buffer,
        0,
        &staging_best_position_buffer,
        0,
        (GRID_SIZE * SAMPLES * D) as u64 * std::mem::size_of::<f32>() as u64,
    );

    encoder.copy_buffer_to_buffer(
        &best_fitness_buffer,
        0,
        &staging_best_fitness_buffer,
        0,
        (GRID_SIZE * SAMPLES) as u64 * std::mem::size_of::<f32>() as u64,
    );

    encoder.copy_buffer_to_buffer(
        &n_buffer,
        0,
        &staging_best_n_buffer,
        0,
        (GRID_SIZE * SAMPLES) as u64 * std::mem::size_of::<u32>() as u64,
    );

    queue.submit(Some(encoder.finish()));

    let best_position_slice = staging_best_position_buffer.slice(..);
    let best_fitness_slice = staging_best_fitness_buffer.slice(..);
    let best_n_slice = staging_best_n_buffer.slice(..);

    let (best_position_sender, best_position_receiver) =
        futures_intrusive::channel::shared::oneshot_channel();
    let (best_fitness_sender, best_fitness_receiver) =
        futures_intrusive::channel::shared::oneshot_channel();
    let (best_n_sender, best_n_receiver) = futures_intrusive::channel::shared::oneshot_channel();

    best_position_slice.map_async(wgpu::MapMode::Read, move |v| {
        best_position_sender.send(v).unwrap()
    });
    best_fitness_slice.map_async(wgpu::MapMode::Read, move |v| {
        best_fitness_sender.send(v).unwrap()
    });
    best_n_slice.map_async(wgpu::MapMode::Read, move |v| best_n_sender.send(v).unwrap());

    let _ = device.poll(wgpu::MaintainBase::Wait);

    if best_position_receiver.receive().await.unwrap().is_ok()
        && best_fitness_receiver.receive().await.unwrap().is_ok()
        && best_n_receiver.receive().await.unwrap().is_ok()
    {
        let position_data = best_position_slice.get_mapped_range();
        let fitness_data = best_fitness_slice.get_mapped_range();
        let n_data = best_n_slice.get_mapped_range();

        let best_positions: &[f32] = bytemuck::cast_slice(&position_data);
        let best_n: &[u32] = bytemuck::cast_slice(&n_data);
        println!("Params \t \t Iterations \t Success");

        for param_id in 0..GRID_SIZE {
            let p1 = param_sets[param_id as usize * 4];
            let p2 = param_sets[param_id as usize * 4 + 1];

            let mut successful_count = 0;
            let mut total_iterations = 0;

            for sample in 0..SAMPLES as usize {
                let output_index = (param_id * SAMPLES) as usize + sample;
                let start_idx = output_index * D as usize;
                let position = &best_positions[start_idx..start_idx + D as usize];

                let distance = position
                    .iter()
                    .map(|&x| (x - SOLUTION) * (x - SOLUTION))
                    .sum::<f32>();
                let success = distance < SUCCESS_THRESHOLD;

                if success {
                    successful_count += 1;
                }

                total_iterations += best_n[output_index];
            }

            let success_rate = successful_count as f32 / SAMPLES as f32;
            let avg_iterations = total_iterations as f32 / SAMPLES as f32;

            println!(
                "({:.1}, {:.})\t {:.1}\t\t {:.1}%",
                p1,
                p2,
                avg_iterations,
                success_rate * 100.0
            );
        }

        drop(position_data);
        drop(fitness_data);
        drop(n_data);
        staging_best_position_buffer.unmap();
        staging_best_fitness_buffer.unmap();
        staging_best_n_buffer.unmap();
    }

    let execution_time = start_time.elapsed();
    println!("Execution time: {} ms", execution_time.as_millis());
}

fn main() {
    futures::executor::block_on(run());
}
