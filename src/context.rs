#[macro_use]
mod macros;

mod buffer;
mod shader;

use self::buffer::Buffer;
use crate::linear::Vec3;
use anyhow::Context as _;
use std::sync::Arc;

pub(crate) struct Context {
    gpu: Arc<GpuContext>,
    swap_chain: wgpu::SwapChain,
    output_size: crate::Size,
    pipeline: wgpu::RenderPipeline,
    bindings: Bindings,

    start: std::time::Instant,

    camera: crate::camera::Camera,
}

pub(crate) struct GpuContext {
    pub instance: wgpu::Instance,
    pub surface: wgpu::Surface,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

struct Bindings {
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,

    uniform_buffer: Buffer<Uniforms>,
    uniforms: Uniforms,

    octree_buffer: Buffer<i32>,
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
    time: f32,
}

// Context creation
impl Context {
    pub async fn new(window: &winit::window::Window) -> anyhow::Result<Context> {
        let gpu = Self::create_gpu_context(window).await?;

        // poll the device in a background thread: enables mapping of buffers
        std::thread::spawn({
            let gpu = gpu.clone();
            move || loop {
                gpu.device.poll(wgpu::Maintain::Wait)
            }
        });

        let output_size = window.inner_size();
        let swap_chain = Self::create_swap_chain(&gpu, output_size);
        let bindings = Self::create_bindings(&gpu);
        let pipeline = Self::create_render_pipeline(&gpu, &bindings)?;

        let camera = crate::camera::Camera {
            position: Vec3::zero(),
            direction: Vec3::zero(),
            fov: 70.0f32.to_radians(),
        };

        Ok(Context {
            gpu,
            swap_chain,
            output_size,
            pipeline,
            bindings,

            start: std::time::Instant::now(),

            camera,
        })
    }

    pub const SWAP_CHAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

    async fn create_gpu_context(window: &winit::window::Window) -> anyhow::Result<Arc<GpuContext>> {
        let backends = wgpu::BackendBit::PRIMARY;
        let instance = wgpu::Instance::new(backends);
        let surface = unsafe { instance.create_surface(window) };
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

        Ok(Arc::new(GpuContext {
            instance,
            surface,
            adapter,
            device,
            queue,
        }))
    }

    fn create_swap_chain(gpu: &GpuContext, size: crate::Size) -> wgpu::SwapChain {
        let descriptor = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            format: Self::SWAP_CHAIN_FORMAT,
            width: size.width,
            height: size.width,
            present_mode: wgpu::PresentMode::Mailbox,
        };

        gpu.device.create_swap_chain(&gpu.surface, &descriptor)
    }

    fn create_octree_nodes(voxels: Vec<([i16; 3], [u8; 3])>) -> Vec<i32> {
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
            color: [u8; 3],
        ) {
            loop {
                let dx = (center[0] <= pos[0]) as usize;
                let dy = (center[1] <= pos[1]) as usize;
                let dz = (center[2] <= pos[2]) as usize;
                let octant = 4 * dx + 2 * dy + dz;

                if extent == 1 {
                    let [r, g, b] = [color[0] as i32, color[1] as i32, color[2] as i32];
                    nodes[current + octant] = (1 << 31) | (r << 16) | (g << 8) | b;
                    return;
                } else {
                    let value = nodes[current + octant];
                    let child = if value == 0 {
                        let node = alloc_node(nodes);
                        nodes[current + octant] = node as i32;
                        node
                    } else if value > 0 {
                        value as usize
                    } else {
                        todo!("split leaf into multiple nodes");
                    };

                    let child_center = [
                        center[0] - extent as i16 / 2 + dx as i16 * extent as i16,
                        center[1] - extent as i16 / 2 + dy as i16 * extent as i16,
                        center[2] - extent as i16 / 2 + dz as i16 * extent as i16,
                    ];

                    current = child;
                    center = child_center;
                    extent = extent / 2;
                }
            }
        }

        let mut extent = 1u16;
        for ([x, y, z], _) in voxels.iter() {
            extent = extent
                .max(x.abs() as u16)
                .max(y.abs() as u16)
                .max(z.abs() as u16);
        }
        let extent = dbg!(extent.next_power_of_two());

        let mut nodes = Vec::new();
        let root = alloc_node(&mut nodes);

        for (pos, color) in voxels {
            insert_node(&mut nodes, root, [0; 3], extent, pos, color);
        }

        nodes
    }

    fn create_voxels() -> Vec<([i16; 3], [u8; 3])> {
        let mut voxels = Vec::new();

        let r = 32;

        for x in -r..=r {
            for z in -r..=r {
                let d = r*r - (x*x + z*z);
                if d >= 0 {
                    let y = f32::sqrt(d as f32) as i16;
                    voxels.push(([x, y, z], [50, 150, 50]));
                    voxels.push(([x, -y, z], [50, 150, 50]));
                }
            }
        }

        voxels
    }

    fn create_bindings(gpu: &GpuContext) -> Bindings {
        use wgpu::BufferUsage as Usage;

        let uniforms = <Uniforms as bytemuck::Zeroable>::zeroed();
        let uniform_buffer = Buffer::new(gpu, Usage::UNIFORM | Usage::COPY_DST, &[uniforms]);

        let mut octree = vec![
            f32::to_bits(0.0) as i32,
            f32::to_bits(0.0) as i32,
            f32::to_bits(0.0) as i32,
            f32::to_bits(2.0) as i32,
        ];

        let nodes = Self::create_octree_nodes(Self::create_voxels());

        octree.extend(dbg!(nodes));

        let octree_buffer = Buffer::new(gpu, Usage::STORAGE | Usage::COPY_DST, &octree);

        let (layout_entries, binding_entries) = bind_group![
            Uniform(0 => (uniform_buffer) in wgpu::ShaderStage::FRAGMENT),
            Storage(1 => (octree_buffer, read_only: true) in wgpu::ShaderStage::FRAGMENT),
        ];

        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &layout_entries,
                });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &binding_entries,
        });

        Bindings {
            bind_group,
            bind_group_layout,
            uniform_buffer,
            uniforms,
            octree_buffer,
        }
    }

    fn create_render_pipeline(
        gpu: &GpuContext,
        bindings: &Bindings,
    ) -> anyhow::Result<wgpu::RenderPipeline> {
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bindings.bind_group_layout],
                push_constant_ranges: &[],
            });

        let vertex_module = shader::create_shader_module(gpu, "shaders/basic.vert")?;
        let fragment_module = shader::create_shader_module(gpu, "shaders/basic.frag")?;

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
}

// Event handling
impl Context {
    pub fn handle_event(
        &mut self,
        event: winit::event::Event<()>,
        flow: &mut winit::event_loop::ControlFlow,
    ) -> anyhow::Result<()> {
        use winit::event::{Event, WindowEvent};
        use winit::event_loop::ControlFlow;

        match event {
            Event::MainEventsCleared => {
                pollster::block_on(self.render()).context("failed to render frame")?;
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *flow = ControlFlow::Exit;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    use winit::event::VirtualKeyCode as Key;

                    let pressed = |key| {
                        input.state == winit::event::ElementState::Pressed
                            && input.virtual_keycode == Some(key)
                    };

                    if pressed(Key::Escape) {
                        *flow = ControlFlow::Exit;
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
    pub async fn render(&mut self) -> anyhow::Result<()> {
        let output = self.get_next_frame()?;

        self.update_bindings();

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

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
            rpass.set_bind_group(0, &self.bindings.bind_group, &[]);
            rpass.draw(0..6, 0..1);
        }

        let command_buffer = encoder.finish();
        self.gpu.queue.submit(Some(command_buffer));

        Ok(())
    }

    fn get_next_frame(&mut self) -> anyhow::Result<wgpu::SwapChainTexture> {
        loop {
            let frame = self
                .swap_chain
                .get_current_frame()
                .context("could not get next frame in swap chain")?;

            if frame.suboptimal {
                self.swap_chain = Self::create_swap_chain(&self.gpu, self.output_size);
            } else {
                return Ok(frame.output);
            }
        }
    }

    fn update_bindings(&mut self) {
        let time = 0.1 * self.start.elapsed().as_secs_f32();

        self.camera.position = Vec3::new(
            -1.5 * (0.4 * time).cos(),
            1.5 - 1.0 * (0.3 * time).cos(),
            -1.5 * (0.4 * time).sin(),
        );
        self.camera.direction = -self.camera.position.norm();

        let [camera_right, camera_up, camera_forward] = self.camera.axis();
        let fov_scale = (self.camera.fov / 2.0).tan();
        let aspect = self.output_size.width as f32 / self.output_size.height as f32;

        self.bindings.uniforms = Uniforms {
            camera_origin: self.camera.position.into(),
            camera_right: Vec3A::from(aspect * camera_right),
            camera_up: Vec3A::from(camera_up),
            camera_forward: Vec3A::from(camera_forward / fov_scale),
            time,
        };

        self.bindings
            .uniform_buffer
            .write(&self.gpu, 0, &[self.bindings.uniforms]);
    }
}

fn color(r: f64, g: f64, b: f64, a: f64) -> wgpu::Color {
    wgpu::Color { r, g, b, a }
}
