#[macro_use]
mod macros;

mod buffer;
mod shader;
mod util;

use self::buffer::Buffer;
use crate::linear::Vec3;
use anyhow::anyhow;
use anyhow::Context as _;
use std::collections::HashSet;
use std::sync::Arc;

pub(crate) struct Context {
    window: Arc<winit::window::Window>,

    gpu: GpuContext,
    swap_chain: wgpu::SwapChain,
    output_size: crate::Size,

    pipeline: wgpu::RenderPipeline,
    voxel_pipeline: wgpu::ComputePipeline,
    denoise_pipeline: wgpu::ComputePipeline,

    bindings: Bindings,
    bind_groups: BindGroups,

    start: std::time::Instant,
    fps_counter: FpsCounter,
    stopwatch: Stopwatch,
    shader_watcher: DirectoryWatcher,

    pressed_keys: HashSet<winit::event::VirtualKeyCode>,
    cursor_grabbed: bool,

    camera: crate::camera::Camera,
    pitch: f32,
    yaw: f32,
}

struct DirectoryWatcher {
    _watcher: notify::RecommendedWatcher,
    events: std::sync::mpsc::Receiver<notify::DebouncedEvent>,
}

pub(crate) struct GpuContext {
    pub instance: wgpu::Instance,
    pub surface: wgpu::Surface,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: wgpu::Queue,
}

struct Bindings {
    old_uniform_buffer: Buffer<Uniforms>,
    uniform_buffer: Buffer<Uniforms>,
    uniforms: Uniforms,
    octree_buffer: Buffer<i32>,
    randomness_buffer: Buffer<f32>,

    old_g_buffer: GBuffer,
    new_g_buffer: GBuffer,

    denoised_color: wgpu::Texture,
}

struct GBuffer {
    size: crate::Size,
    color: wgpu::Texture,
    position: wgpu::Texture,
    normal: wgpu::Texture,
}

struct BindGroups {
    render: BindGroup,
    voxel: BindGroup,
    denoise: BindGroup,
}

struct BindGroup {
    layout: wgpu::BindGroupLayout,
    bindings: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vec3A {
    vector: Vec3,
    _padding: f32,
}

impl From<Vec3> for Vec3A {
    fn from(vector: Vec3) -> Self {
        Vec3A {
            vector,
            _padding: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    camera_origin: Vec3A,
    camera_right: Vec3A,
    camera_up: Vec3A,
    camera_forward: Vec3A,
    light: PointLight,
    time: f32,
    still_sample: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PointLight {
    position: Vec3,
    brightness: f32,
}

struct Stopwatch {
    prev_time: std::time::Instant,
}

impl Stopwatch {
    pub fn new() -> Stopwatch {
        Stopwatch {
            prev_time: std::time::Instant::now(),
        }
    }

    pub fn tick(&mut self) -> std::time::Duration {
        let now = std::time::Instant::now();
        let duration = now.saturating_duration_since(self.prev_time);
        self.prev_time = now;
        duration
    }
}

struct FpsCounter {
    prev_time: std::time::Instant,
    frames: u32,
}

impl FpsCounter {
    pub fn new() -> Self {
        FpsCounter {
            prev_time: std::time::Instant::now(),
            frames: 0,
        }
    }

    pub fn tick(&mut self) -> Option<f32> {
        let now = std::time::Instant::now();
        let elapsed = (now - self.prev_time).as_secs_f32();
        self.frames += 1;
        if elapsed > 0.25 {
            let fps = self.frames as f32 / elapsed;
            self.prev_time = now;
            self.frames = 0;
            Some(fps)
        } else {
            None
        }
    }
}

// Context creation
impl Context {
    pub async fn new(window: Arc<winit::window::Window>) -> anyhow::Result<Context> {
        let gpu = Self::create_gpu_context(&window).await?;

        // poll the device in a background thread: enables mapping of buffers
        std::thread::spawn({
            let device = gpu.device.clone();
            move || loop {
                device.poll(wgpu::Maintain::Wait)
            }
        });

        let output_size = window.inner_size();
        let swap_chain = Self::create_swap_chain(&gpu, output_size);
        let bindings = Self::create_bindings(&gpu, output_size);

        let bind_groups = Self::create_bind_groups(&gpu, &bindings);

        let pipeline = Self::create_render_pipeline(&bind_groups.render, &gpu)?;
        let voxel_pipeline = Self::create_voxel_pipeline(&bind_groups.voxel, &gpu)?;
        let denoise_pipeline = Self::create_denoise_pipeline(&bind_groups.denoise, &gpu)?;

        let camera = crate::camera::Camera {
            position: Vec3::new(0.0, 0.0, -2.0),
            direction: Vec3::new(0.0, 0.0, 1.0),
            fov: 70.0f32.to_radians(),
        };

        let shader_watcher = DirectoryWatcher::new("shaders/")?;

        Ok(Context {
            window,

            gpu,
            swap_chain,
            output_size,

            bindings,
            bind_groups,

            pipeline,
            voxel_pipeline,
            denoise_pipeline,

            start: std::time::Instant::now(),
            stopwatch: Stopwatch::new(),
            fps_counter: FpsCounter::new(),
            shader_watcher,

            pressed_keys: HashSet::new(),
            cursor_grabbed: true,

            camera,
            pitch: 0.0,
            yaw: 0.0,
        })
    }

    pub const SWAP_CHAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

    async fn create_gpu_context(window: &winit::window::Window) -> anyhow::Result<GpuContext> {
        let backends = wgpu::BackendBit::PRIMARY;
        let instance = wgpu::Instance::new(backends);
        let surface = Self::create_surface(&instance, &window);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
            })
            .await
            .context("failed to find compatible graphics adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .context("failed to find device")?;

        Ok(GpuContext {
            instance,
            surface,
            adapter,
            device: Arc::new(device),
            queue,
        })
    }

    fn create_surface(instance: &wgpu::Instance, window: &winit::window::Window) -> wgpu::Surface {
        unsafe { instance.create_surface(window) }
    }

    fn create_swap_chain(gpu: &GpuContext, size: crate::Size) -> wgpu::SwapChain {
        let descriptor = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            format: Self::SWAP_CHAIN_FORMAT,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate,
        };

        gpu.device.create_swap_chain(&gpu.surface, &descriptor)
    }

    fn create_octree_nodes(voxels: Vec<([i16; 3], [u8; 4])>) -> Vec<i32> {
        fn alloc_node(nodes: &mut Vec<i32>) -> usize {
            let index = nodes.len();
            nodes.extend_from_slice(&[0; 8]);
            index
        }

        fn insert_node(
            nodes: &mut Vec<i32>,
            mut current: usize,
            mut center: [i16; 3],
            mut extent: u16,
            pos: [i16; 3],
            color: [u8; 4],
        ) {
            loop {
                let dx = (center[0] <= pos[0]) as usize;
                let dy = (center[1] <= pos[1]) as usize;
                let dz = (center[2] <= pos[2]) as usize;
                let octant = 4 * dx + 2 * dy + dz;

                if extent == 1 {
                    let [m, r, g, b] = color;
                    let [m, r, g, b] = [m as i32, r as i32, g as i32, b as i32];
                    nodes[current + octant] =
                        (1 << 31) | ((m & 0x7f) << 24) | (r << 16) | (g << 8) | b;
                    return;
                } else {
                    let value = nodes[current + octant];
                    let child = match value {
                        0 => {
                            let node = alloc_node(nodes);
                            nodes[current + octant] = node as i32;
                            node
                        }
                        _ if value > 0 => value as usize,
                        _ => todo!("split leaf into multiple nodes"),
                    };

                    let child_center = [
                        center[0] - extent as i16 / 2 + dx as i16 * extent as i16,
                        center[1] - extent as i16 / 2 + dy as i16 * extent as i16,
                        center[2] - extent as i16 / 2 + dz as i16 * extent as i16,
                    ];

                    current = child;
                    center = child_center;
                    extent /= 2;
                }
            }
        }

        let mut extent = 0u16;
        for ([x, y, z], _) in voxels.iter() {
            extent = extent
                .max(x.abs() as u16)
                .max(y.abs() as u16)
                .max(z.abs() as u16);
        }
        let extent = (1 + extent).next_power_of_two();

        let mut nodes = Vec::new();
        let root = alloc_node(&mut nodes);

        for (pos, color) in voxels {
            insert_node(&mut nodes, root, [0; 3], extent, pos, color);
        }

        nodes
    }

    fn create_voxels() -> Vec<([i16; 3], [u8; 4])> {
        let mut voxels = Vec::new();

        let radius = 16i32;

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut color = |p: f32, _x: i32, _y: i32, _z: i32| {
            let red = rng.gen_range(50..=255);
            let green = rng.gen_range(50..=255);
            let blue = rng.gen_range(50..=255);

            let emmisive = rng.gen_bool(p as f64);
            let material = (emmisive as u8) << 6;

            [material, red, green, blue]
        };

        let width = 2 * (radius + 1) as usize;
        let mut heights = vec![None; width.pow(2)];

        for x in -radius..=radius {
            for z in -radius..=radius {
                let xi = (x + radius) as usize;
                let zi = (z + radius) as usize;

                if x.pow(2) + z.pow(2) <= radius.pow(2) {
                    let y =
                        -(radius.pow(2) as f32 - x.pow(2) as f32 - z.pow(2) as f32).sqrt() as i32;
                    heights[xi + zi * width] = Some(y);
                } else {
                    heights[xi + zi * width] = Some(0);
                }
            }
        }

        let get_height = |x: i32, z: i32| {
            if x < -radius || x > radius || z < -radius || z > radius {
                None
            } else {
                let xi = (x + radius) as usize;
                let zi = (z + radius) as usize;
                heights[xi + zi * width]
            }
        };

        for x in -radius..=radius {
            for z in -radius..=radius {
                if let Some(curr) = get_height(x, z) {
                    let low = curr
                        .min(get_height(x - 1, z).unwrap_or(curr))
                        .min(get_height(x + 1, z).unwrap_or(curr))
                        .min(get_height(x, z - 1).unwrap_or(curr))
                        .min(get_height(x, z + 1).unwrap_or(curr));
                    for y in low..=curr {
                        voxels.push(([x as i16, y as i16, z as i16], color(0.01, x, y, z)));
                    }
                }
            }
        }

        for x in -radius..=radius {
            voxels.push(([x as i16, -10, 0], [0x40, 255, 255, 255]));
        }

        voxels
    }

    fn voxels_from_vox(vox: &crate::vox::Vox) -> Vec<([i16; 3], [u8; 4])> {
        let mut voxels = Vec::new();

        let model = &vox.models[0];
        for voxel in &model.voxels {
            let [r, g, b] = vox.get_color_rgb(voxel.color);
            let material = vox.materials.get(&(voxel.color as u32)).unwrap();

            let mut mat = 0;
            if matches!(material.kind, crate::vox::MaterialKind::Emit) {
                mat |= 1 << 6;
            }

            voxels.push((
                [voxel.x as i16, voxel.z as i16, voxel.y as i16],
                [mat, r, g, b],
            ));
        }

        voxels
    }

    fn create_octree(voxels: Vec<([i16; 3], [u8; 4])>) -> Vec<i32> {
        let mut octree = vec![
            // center
            f32::to_bits(0.0) as i32,
            f32::to_bits(0.0) as i32,
            f32::to_bits(0.0) as i32,
            // size
            f32::to_bits(2.0) as i32,
        ];

        let nodes = Self::create_octree_nodes(voxels);
        octree.extend(nodes);
        octree
    }

    fn create_bindings(gpu: &GpuContext, output_size: crate::Size) -> Bindings {
        use wgpu::BufferUsage as Usage;

        let mut uniforms = <Uniforms as bytemuck::Zeroable>::zeroed();
        uniforms.light = PointLight {
            position: Vec3::new(0.4, -0.4, 0.02),
            brightness: 0.05,
        };

        let uniform_buffer = Buffer::new(gpu, Usage::UNIFORM | Usage::COPY_DST, &[uniforms]);
        let old_uniform_buffer = Buffer::new(gpu, Usage::UNIFORM | Usage::COPY_DST, &[uniforms]);

        let octree = Self::create_octree(Self::create_voxels());
        let octree_buffer = Buffer::new(gpu, Usage::STORAGE | Usage::COPY_DST, &octree);

        let old_g_buffer = GBuffer::new(gpu, output_size);
        let new_g_buffer = GBuffer::new(gpu, output_size);

        let denoised_color =
            GBuffer::create_storage_texture(output_size, GBuffer::COLOR_FORMAT, gpu);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let randomness = (0..(1 << 16))
            .map(|_| rng.gen_range(0.0..1.0))
            .collect::<Vec<_>>();
        let randomness_buffer = Buffer::new(gpu, Usage::STORAGE | Usage::COPY_DST, &randomness);

        Bindings {
            old_uniform_buffer,
            uniform_buffer,
            uniforms,
            octree_buffer,
            randomness_buffer,

            old_g_buffer,
            new_g_buffer,

            denoised_color,
        }
    }

    fn create_bind_groups(gpu: &GpuContext, bindings: &Bindings) -> BindGroups {
        BindGroups {
            render: Self::create_render_bind_group(gpu, bindings),
            voxel: Self::create_voxel_bind_group(gpu, bindings),
            denoise: Self::create_denoise_bind_group(gpu, bindings),
        }
    }

    fn create_render_bind_group(gpu: &GpuContext, bindings: &Bindings) -> BindGroup {
        let color = util::view(&bindings.denoised_color);
        let (layout, bindings) = bind_group![
            UniformImage(0 => (&color, ReadOnly, GBuffer::COLOR_FORMAT, D2) in FRAGMENT),
            // Uniform(0 => (bindings.uniform_buffer) in wgpu::ShaderStage::FRAGMENT),
        ];

        BindGroup::from_entries(&layout, &bindings, gpu)
    }

    fn create_voxel_bind_group(gpu: &GpuContext, bindings: &Bindings) -> BindGroup {
        let old_images = &bindings.old_g_buffer;
        let new_images = &bindings.new_g_buffer;

        let old_color = util::view(&old_images.color);
        let new_color = util::view(&new_images.color);

        let old_position = util::view(&old_images.position);
        let new_position = util::view(&new_images.position);

        let (layout, bindings) = bind_group![
            UniformImage(0 => (&old_color, ReadOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
            UniformImage(1 => (&new_color, WriteOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
            UniformImage(2 => (&old_position, ReadOnly, GBuffer::POSITION_FORMAT, D2) in COMPUTE),
            UniformImage(3 => (&new_position, WriteOnly, GBuffer::POSITION_FORMAT, D2) in COMPUTE),
            Uniform(6 => (&bindings.uniform_buffer) in COMPUTE),
            Uniform(7 => (&bindings.old_uniform_buffer) in COMPUTE),
            Storage(8 => (&bindings.octree_buffer, read_only: true) in COMPUTE),
            Storage(9 => (&bindings.randomness_buffer, read_only: true) in COMPUTE),
        ];

        BindGroup::from_entries(&layout, &bindings, gpu)
    }

    fn create_denoise_bind_group(gpu: &GpuContext, bindings: &Bindings) -> BindGroup {
        let new_color = util::view(&bindings.new_g_buffer.color);
        let denoise_color = util::view(&bindings.denoised_color);

        let (layout, bindings) = bind_group![
            UniformImage(0 => (&denoise_color, WriteOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
            UniformImage(1 => (&new_color, ReadOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
        ];

        BindGroup::from_entries(&layout, &bindings, gpu)
    }

    fn create_compute_pipeline(
        bind_group: &BindGroup,
        shader: impl AsRef<std::path::Path>,
        gpu: &GpuContext,
    ) -> anyhow::Result<wgpu::ComputePipeline> {
        let module = shader::create_shader_module(gpu, shader.as_ref())?;

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group.layout],
                push_constant_ranges: &[],
            });

        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&layout),
                module: &module,
                entry_point: "main",
            });

        Ok(pipeline)
    }

    fn create_voxel_pipeline(
        bind_group: &BindGroup,
        gpu: &GpuContext,
    ) -> anyhow::Result<wgpu::ComputePipeline> {
        Self::create_compute_pipeline(bind_group, "shaders/voxels.comp", gpu)
    }

    fn create_denoise_pipeline(
        bind_group: &BindGroup,
        gpu: &GpuContext,
    ) -> anyhow::Result<wgpu::ComputePipeline> {
        Self::create_compute_pipeline(bind_group, "shaders/denoise.comp", gpu)
    }

    fn create_render_pipeline(
        bind_group: &BindGroup,
        gpu: &GpuContext,
    ) -> anyhow::Result<wgpu::RenderPipeline> {
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group.layout],
                push_constant_ranges: &[],
            });

        let vertex_module = shader::create_shader_module(gpu, "shaders/basic.vert")?;
        let fragment_module = shader::create_shader_module(gpu, "shaders/display.frag")?;

        let vertex = wgpu::VertexState {
            module: &vertex_module,
            entry_point: "main",
            buffers: &[],
        };

        let fragment = wgpu::FragmentState {
            module: &fragment_module,
            entry_point: "main",
            targets: &[wgpu::ColorTargetState {
                format: Self::SWAP_CHAIN_FORMAT,
                alpha_blend: wgpu::BlendState::REPLACE,
                color_blend: wgpu::BlendState::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
        };

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                vertex,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: wgpu::CullMode::None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(fragment),
            });

        Ok(pipeline)
    }

    fn recreate_pipeline(&mut self) -> anyhow::Result<()> {
        info!("recreating render pipeline");
        self.pipeline = Self::create_render_pipeline(&self.bind_groups.render, &self.gpu)?;

        info!("recreating voxel pipeline");
        self.voxel_pipeline = Self::create_voxel_pipeline(&self.bind_groups.voxel, &self.gpu)?;

        info!("recreating denoise pipeline");
        self.denoise_pipeline =
            Self::create_denoise_pipeline(&self.bind_groups.denoise, &self.gpu)?;

        self.bindings.uniforms.still_sample = 0;
        Ok(())
    }

    fn resize(&mut self, new_size: crate::Size) -> anyhow::Result<()> {
        self.output_size = new_size;
        self.recreate_swap_chain();
        self.bindings.old_g_buffer = GBuffer::new(&self.gpu, self.output_size);
        self.bindings.new_g_buffer = GBuffer::new(&self.gpu, self.output_size);
        self.bindings.denoised_color =
            GBuffer::create_storage_texture(new_size, GBuffer::COLOR_FORMAT, &self.gpu);
        self.bind_groups = Self::create_bind_groups(&self.gpu, &self.bindings);
        self.recreate_pipeline()
    }
}

// Event handling
impl Context {
    pub fn handle_event(
        &mut self,
        event: winit::event::Event<()>,
        flow: &mut winit::event_loop::ControlFlow,
    ) -> anyhow::Result<()> {
        use winit::event::{DeviceEvent, Event, WindowEvent};
        use winit::event_loop::ControlFlow;

        #[allow(clippy::single_match, clippy::collapsible_match)]
        match event {
            Event::MainEventsCleared => {
                while let Ok(event) = self.shader_watcher.events.try_recv() {
                    match event {
                        notify::DebouncedEvent::Create(_)
                        | notify::DebouncedEvent::Write(_)
                        | notify::DebouncedEvent::Chmod(_)
                        | notify::DebouncedEvent::Remove(_)
                        | notify::DebouncedEvent::Rename(_, _) => self
                            .recreate_pipeline()
                            .context("failed to recreate pipeline")?,
                        notify::DebouncedEvent::Rescan
                        | notify::DebouncedEvent::NoticeWrite(_)
                        | notify::DebouncedEvent::NoticeRemove(_) => { /* ignore */ }
                        notify::DebouncedEvent::Error(error, path) => {
                            error!(?path, "while watching shader directory: {}", error);
                        }
                    }
                }

                if let Some(fps) = self.fps_counter.tick() {
                    self.window.set_title(&format!("voxels @ {:.1}", fps));
                }

                let dt = self.stopwatch.tick().as_secs_f32();
                self.update(dt);
                pollster::block_on(self.render()).context("failed to render frame")?;
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *flow = ControlFlow::Exit;
                }
                WindowEvent::Resized(new_size) => {
                    self.resize(new_size).context("failed to resize")?;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    use winit::event::{ElementState::Pressed, VirtualKeyCode as Key};

                    if let Some(key) = input.virtual_keycode {
                        match input.state {
                            winit::event::ElementState::Pressed => {
                                self.pressed_keys.insert(key);
                            }
                            winit::event::ElementState::Released => {
                                self.pressed_keys.remove(&key);
                            }
                        }

                        match key {
                            Key::Escape => *flow = ControlFlow::Exit,
                            Key::Space if input.state == Pressed => {
                                self.bindings.uniforms.light.position = self.camera.position
                            }
                            Key::Tab if input.state == Pressed => {
                                self.cursor_grabbed = !self.cursor_grabbed;
                                if self.window.set_cursor_grab(self.cursor_grabbed).is_ok() {
                                    self.window.set_cursor_visible(!self.cursor_grabbed);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                WindowEvent::DroppedFile(path) => {
                    info!(path = %path.display(), "loading vox file");
                    match crate::vox::load(&path) {
                        Err(e) => error!("failed to load vox: {:?}", e),
                        Ok(vox) => {
                            let octree = Self::create_octree(Self::voxels_from_vox(&vox));
                            use wgpu::BufferUsage as Usage;
                            self.bindings.octree_buffer =
                                Buffer::new(&self.gpu, Usage::STORAGE | Usage::COPY_DST, &octree);
                            self.bind_groups = Self::create_bind_groups(&self.gpu, &self.bindings);
                            self.recreate_pipeline()
                                .context("failed to recreate pipeline")?;
                        }
                    }
                }
                _ => {}
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                    if self.cursor_grabbed {
                        self.yaw += 0.001 * dx as f32;
                        self.pitch -= 0.001 * dy as f32;
                        self.bindings.uniforms.still_sample = 0;
                    }
                }
                _ => {}
            },
            _ => {}
        }

        Ok(())
    }
}

// Rendering
impl Context {
    pub fn update(&mut self, dt: f32) {
        use winit::event::VirtualKeyCode as Key;

        self.camera.direction = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        );

        let pressed = |keys: &[Key]| keys.iter().any(|key| self.pressed_keys.contains(key));

        let [right, _, forward] = self.camera.axis();
        let mut movement = Vec3::zero();

        // I'm primarily a Dvorak user, so excuse the key bindings ;)
        if pressed(&[Key::Up, Key::Comma]) {
            movement += forward;
        }
        if pressed(&[Key::Down, Key::O]) {
            movement -= forward;
        }
        if pressed(&[Key::Right, Key::E]) {
            movement += right;
        }
        if pressed(&[Key::Left, Key::A]) {
            movement -= right;
        }
        if pressed(&[Key::PageUp, Key::Period]) {
            movement.y += 1.0;
        }
        if pressed(&[Key::PageDown, Key::Apostrophe]) {
            movement.y -= 1.0;
        }
        if movement != Vec3::zero() {
            let speed = if pressed(&[Key::LControl, Key::RControl]) {
                0.02
            } else if pressed(&[Key::LShift, Key::RShift]) {
                1.0
            } else {
                0.2
            };
            self.camera.position += speed * dt * movement.norm();
            self.bindings.uniforms.still_sample = 0;
        }
    }

    pub async fn render(&mut self) -> anyhow::Result<()> {
        let output = self.get_next_frame()?;

        self.update_bindings();

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

            let local_x = 16;
            let local_y = 16;
            let groups_x = (self.output_size.width + local_x - 1) / local_x;
            let groups_y = (self.output_size.height + local_y - 1) / local_y;

            cpass.set_pipeline(&self.voxel_pipeline);
            cpass.set_bind_group(0, &self.bind_groups.voxel.bindings, &[]);
            cpass.dispatch(groups_x, groups_y, 1);

            cpass.set_pipeline(&self.denoise_pipeline);
            cpass.set_bind_group(0, &self.bind_groups.denoise.bindings, &[]);
            cpass.dispatch(groups_x, groups_y, 1);
        }

        self.bindings
            .old_g_buffer
            .copy_from(&self.bindings.new_g_buffer, &mut encoder);

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &output.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(color(0.2, 0.2, 0.2, 1.0)),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_groups.render.bindings, &[]);
            rpass.draw(0..6, 0..1);
        }

        let command_buffer = encoder.finish();
        self.gpu.queue.submit(Some(command_buffer));

        Ok(())
    }

    fn get_next_frame(&mut self) -> anyhow::Result<wgpu::SwapChainTexture> {
        for _attempt in 0..8 {
            match self.swap_chain.get_current_frame() {
                Err(wgpu::SwapChainError::Outdated) => {
                    let _ = info_span!("swap chain outdated");
                    self.recreate_swap_chain();
                }
                Ok(wgpu::SwapChainFrame {
                    suboptimal: true, ..
                })
                | Err(wgpu::SwapChainError::Lost) => {
                    let _ = info_span!("swap chain lost");
                    info!("recreating swap chain");
                    self.swap_chain = Self::create_swap_chain(&self.gpu, self.output_size);
                }
                Ok(frame) => return Ok(frame.output),
                Err(e) => return Err(e).context("could not get next frame in swap chain"),
            }
        }

        Err(anyhow!("failed to fetch next frame in swap chain"))
    }

    fn recreate_swap_chain(&mut self) {
        info!("recreating surface");
        self.gpu.surface = Self::create_surface(&self.gpu.instance, &self.window);
        info!("recreating swap chain");
        self.swap_chain = Self::create_swap_chain(&self.gpu, self.output_size);
    }

    fn update_bindings(&mut self) {
        let time = self.start.elapsed().as_secs_f32();

        let [camera_right, camera_up, camera_forward] = self.camera.axis_scaled(self.output_size);

        self.bindings
            .old_uniform_buffer
            .write(&self.gpu, 0, &[self.bindings.uniforms]);

        self.bindings.uniforms = Uniforms {
            camera_origin: self.camera.position.into(),
            camera_right: Vec3A::from(camera_right),
            camera_up: Vec3A::from(camera_up),
            camera_forward: Vec3A::from(camera_forward),
            time,
            still_sample: self.bindings.uniforms.still_sample + 1,
            ..self.bindings.uniforms
        };

        self.bindings
            .uniform_buffer
            .write(&self.gpu, 0, &[self.bindings.uniforms]);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let randomness = (0..(1 << 16))
            .map(|_| rng.gen_range(0.0..1.0))
            .collect::<Vec<_>>();
        self.bindings
            .randomness_buffer
            .write(&self.gpu, 0, &randomness);
    }
}

fn color(r: f64, g: f64, b: f64, a: f64) -> wgpu::Color {
    wgpu::Color { r, g, b, a }
}

impl BindGroup {
    pub fn from_entries(
        layout: &[wgpu::BindGroupLayoutEntry],
        bindings: &[wgpu::BindGroupEntry],
        gpu: &GpuContext,
    ) -> BindGroup {
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: layout,
            });

        let bindings = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: bindings,
        });

        BindGroup { layout, bindings }
    }
}

impl DirectoryWatcher {
    fn new(path: impl AsRef<std::path::Path>) -> anyhow::Result<DirectoryWatcher> {
        use notify::Watcher;

        let path = path.as_ref();

        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = notify::watcher(tx, std::time::Duration::from_millis(500))
            .context("failed to create file system watcher")?;
        watcher
            .watch(path, notify::RecursiveMode::Recursive)
            .with_context(|| format!("failed to watch over directory `{}`", path.display()))?;

        Ok(DirectoryWatcher {
            _watcher: watcher,
            events: rx,
        })
    }
}

impl GBuffer {
    pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
    pub const POSITION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
    pub const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

    fn new(gpu: &GpuContext, size: crate::Size) -> GBuffer {
        let color = Self::create_storage_texture(size, GBuffer::COLOR_FORMAT, gpu);
        let position = Self::create_storage_texture(size, GBuffer::POSITION_FORMAT, gpu);
        let normal = Self::create_storage_texture(size, GBuffer::NORMAL_FORMAT, gpu);

        GBuffer {
            size,
            color,
            position,
            normal,
        }
    }

    fn create_storage_texture(
        size: crate::Size,
        format: wgpu::TextureFormat,
        gpu: &GpuContext,
    ) -> wgpu::Texture {
        use wgpu::TextureUsage as Usage;
        util::create_texture(
            size,
            format,
            Usage::COPY_SRC | Usage::COPY_DST | Usage::STORAGE,
            gpu,
        )
    }

    fn copy_from(&self, other: &GBuffer, encoder: &mut wgpu::CommandEncoder) {
        util::copy_entire_texture_to_texture(&other.color, &self.color, self.size, encoder);
        util::copy_entire_texture_to_texture(&other.position, &self.position, self.size, encoder);
        util::copy_entire_texture_to_texture(&other.normal, &self.normal, self.size, encoder);
    }
}
