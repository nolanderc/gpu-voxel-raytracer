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

use crate::scancodes::Scancode as KeyCode;

use std::convert::TryFrom;
use std::rc::Rc;

/// Context which contains state for the entire rendering pipeline
pub(crate) struct Context {
    /// Handle to the window.
    window: Arc<winit::window::Window>,

    /// Handle to the GPU device and command queue.
    gpu: GpuContext,

    /// Handle to the front and back-buffer
    swap_chain: wgpu::SwapChain,

    /// Size of the output window
    output_size: crate::Size,

    /// Pipeline for displaying the final output to the screen using a full-window quad
    pipeline: wgpu::RenderPipeline,

    /// Compute pipeline for computing ray-voxel intersections
    voxel_pipeline: wgpu::ComputePipeline,

    /// Compute pipeline for doing temporal blending
    temporal_pipeline: wgpu::ComputePipeline,

    /// Compute pipeline for denoising the final image
    denoise_pipeline: wgpu::ComputePipeline,

    /// Pipeline to render the GUI
    gui_pipeline: wgpu::RenderPipeline,

    /// Stores handles and data of GPU-side objects
    bindings: Bindings,

    /// Specifies which bindings are used for each shader
    bind_groups: BindGroups,

    /// Time when the renderer was started
    start: std::time::Instant,

    /// Counts the frames per specond (fps)
    fps_counter: FpsCounter,

    /// Computes the time between each frame
    stopwatch: Stopwatch,

    /// Notifies the application when shader source has changed
    shader_watcher: DirectoryWatcher,

    /// Stores user keyboard and mouse input
    input: UserInput,

    /// Camera parameters controlled by the user
    camera: crate::camera::Camera,
    pitch: f32,
    yaw: f32,

    /// Stores GUI state
    gui: Gui,
}

struct UserInput {
    /// Set of keys which are currently pressed
    pressed_keys: HashSet<KeyCode>,
    /// Is the cursor grabbed by the window? (i.e. is it visible or locked inside the window)
    cursor_grabbed: bool,

    /// Position of the mouse in logical coordinates (points instead of pixels)
    mouse_position: winit::dpi::LogicalPosition<f64>,

    /// Which modifier keys are currently pressed?
    modifiers: winit::event::ModifiersState,
}

struct Gui {
    /// Context to the egui library
    ctx: egui::CtxRef,

    /// List of events which during the last frame
    events: Vec<egui::Event>,
    /// List of meshes to use for rendering the GUI
    meshes: Vec<GuiMesh>,
    /// List of models on disk
    vox_files: Vec<Rc<std::path::Path>>,
    /// The currently chosen model to use for display
    current_model: Model,
}

#[derive(Debug, Clone, PartialEq)]
enum Model {
    /// Use the default, proceduraly generated, model
    Default,
    /// Use a specific vox file
    Vox(Rc<std::path::Path>),
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::Default => write!(f, "default"),
            Model::Vox(path) => {
                write!(f, "{}", path.display())
            }
        }
    }
}

/// A mesh used in the GUI
struct GuiMesh {
    /// Index buffer, makes up the triangles of the mesh
    indices: Buffer<u32>,
    /// Vertex buffer, specifies the of the triangles
    vertices: Buffer<GuiVertex>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GuiVertex {
    /// Vertex's position in the range [0, size), where size is the size of the window.
    position: [f32; 2],
    /// Texture coordinates in the range [0, 1]
    tex_coord: [f32; 2],
    /// RGBA color information with premultiplied alpha
    color: [u8; 4],
}

impl Gui {
    /// Format used for textures (only stores alpha), colors are per vertex
    pub const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

    /// Create a new GUI context
    pub fn new() -> anyhow::Result<Self> {
        let vox_files =
            Self::list_vox_files("vox").context("failed to gather names of vox files")?;

        let gui = Gui {
            ctx: egui::CtxRef::default(),
            events: Vec::new(),
            meshes: Vec::new(),
            vox_files,
            current_model: Model::Default,
        };

        Ok(gui)
    }

    /// List the vox files found in the specified directory
    fn list_vox_files(directory: &str) -> anyhow::Result<Vec<Rc<std::path::Path>>> {
        let entries = std::fs::read_dir(directory)?;

        let mut files = Vec::new();

        for entry in entries {
            let entry = entry.context("failed to read directory entry")?;
            if entry
                .file_type()
                .context("could not get file type")?
                .is_file()
            {
                let path = entry.path();
                if matches!(path.extension(), Some(ext) if ext == "vox") {
                    files.push(path.into());
                }
            }
        }

        files.sort();

        Ok(files)
    }
}

/// Watches over a directory, notifying on any changes
struct DirectoryWatcher {
    /// The actual implementation
    _watcher: notify::RecommendedWatcher,
    /// Sends event along this channel when something has changed.
    events: std::sync::mpsc::Receiver<notify::DebouncedEvent>,
}

/// A GPU context
pub(crate) struct GpuContext {
    /// Main handle to the wgpu library
    pub instance: wgpu::Instance,

    /// The surface (window) to render to
    pub surface: wgpu::Surface,
    /// Handle to the specific "driver"
    pub adapter: wgpu::Adapter,
    /// Handle to a device the `adapter` owns
    pub device: Arc<wgpu::Device>,
    /// A queue onto which commands can be sent to the GPU
    pub queue: wgpu::Queue,
}

/// GPU related data storage
struct Bindings {
    /// GPU buffer of parameters that were used to render the previous frame
    old_uniform_buffer: Buffer<Uniforms>,
    /// GPU buffer of parameters that will be used to render the next frame
    uniform_buffer: Buffer<Uniforms>,

    /// The parameters that will be used to render the next frame. These are uploaded to the CPU at
    /// the start of every frame.
    uniforms: Uniforms,

    /// GPU representation of the octree datastructure.
    octree_buffer: Buffer<i32>,

    /// A buffer full of samples of blue noise. Used to generate random numbers on the GPU.
    randomness_buffer: Buffer<f32>,

    /// Per-pixel information generated during the previous frame (colors, normals, depth, etc.)
    old_g_buffer: GBuffer,
    /// Per-pixel information generated during the next frame (colors, normals, depth, etc.)
    new_g_buffer: GBuffer,

    /// Output of the indirect lighting collected by the path-tracer
    sampled_color: wgpu::Texture,

    /// Output of the direct lighting collected by the path-tracer
    sampled_albedo: wgpu::Texture,

    /// Output of the specular lighting collected by the path-tracer
    sampled_specular: wgpu::Texture,

    /// GPU-side parameters that control the temporal blending
    temporal_uniforms: UniformBuffer<TemporalUniforms>,

    /// Output from the denoiser
    denoised_color: wgpu::Texture,

    /// Parameters that control the denoiser
    denoise_uniforms: UniformBuffer<DenoiseUniforms>,

    /// Parameters that control the GUI
    gui_uniforms: Buffer<GuiUniforms>,

    /// Texture used for GUI rendering
    gui_texture: GuiTexture,

    /// GPU-side handle to sampler that performs bilinear filtering on textures.
    near_sampler: wgpu::Sampler,
}

/// A buffer that can be used for storing uniforms on the GPU.
struct UniformBuffer<T> {
    /// The CPU-side values.
    uniforms: T,
    /// The GPU-side values.
    buffer: Buffer<T>,
}

impl<T> std::ops::Deref for UniformBuffer<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.uniforms
    }
}

impl<T> std::ops::DerefMut for UniformBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.uniforms
    }
}

impl<T: bytemuck::Pod> UniformBuffer<T> {
    /// Create a new uniform buffer with the given default values.
    pub fn new(gpu: &GpuContext, uniforms: T) -> UniformBuffer<T> {
        use wgpu::BufferUsage as Usage;
        let buffer = Buffer::new(
            gpu,
            Usage::UNIFORM | Usage::COPY_DST,
            std::slice::from_ref(&uniforms),
        );

        UniformBuffer { uniforms, buffer }
    }

    /// Upload the CPU-side data to the GPU
    pub fn upload(&self, gpu: &GpuContext) {
        self.buffer
            .write(gpu, 0, std::slice::from_ref(&self.uniforms));
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DenoiseUniforms {
    /// Size of the area to consider when performing denoising
    radius: u32,
    /// Larger values result in further away pixels having more influence over the current pixel
    sigma_distance: f32,
    /// Larger values result in pixels that are very different being blended more
    sigma_range: f32,
    /// How much to blend indirect lighting with direct lighting (should be 1 for accurate image,
    /// mostly used for debugging)
    albedo_factor: f32,
}

impl Default for DenoiseUniforms {
    fn default() -> Self {
        DenoiseUniforms {
            radius: 0,
            sigma_distance: 2.0,
            sigma_range: 1.5,
            albedo_factor: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GuiUniforms {
    /// Widht of the window, used for correct scaling of GUI
    width: f32,
    /// Height of the window, used for correct scaling of GUI
    height: f32,
}

struct GuiTexture {
    /// Which version of the texture is currently on the GPU (this is basically a unique hash)
    version: Option<u64>,
    /// Size of the texture on the GPU
    size: crate::Size,
    /// Handle to the data on the GPU
    texture: wgpu::Texture,
}

impl GuiTexture {
    /// Create an empty texture
    pub fn empty(gpu: &GpuContext) -> Self {
        let size = [1, 1].into();
        let texture = util::create_texture_with_data(
            size,
            Gui::TEXTURE_FORMAT,
            wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            gpu,
            &[0; 4],
        );

        GuiTexture {
            // We set the version to none so that it always updates when there's a new texture with
            // actual content
            version: None,
            size,
            texture,
        }
    }
}

struct GBuffer {
    /// Size of the images
    size: crate::Size,
    /// Output color image with all effects applied (except denoising)
    color: wgpu::Texture,
    /// RGB: Normal of the first surface hit,
    /// A: distance from the camera to the first surface hit
    normal_depth: wgpu::Texture,
    /// Specular lighting in the previous frame
    specular: wgpu::Texture,
}

struct BindGroups {
    /// Bindings used during final presentation
    render: BindGroup,
    /// Bindings used for ray-voxel intersections
    voxel: BindGroup,
    /// Bindings used for denoising
    denoise: BindGroup,
    /// Bindings used for the GUI
    gui: BindGroup,
    /// Bindings used for temporal blending
    temporal: BindGroup,
}

struct BindGroup {
    /// The layout of the bindings: which bindings correspond to what kinds of data
    layout: wgpu::BindGroupLayout,
    /// Values bound to the actual bindings in the shader
    bindings: wgpu::BindGroup,
}

/// A 3D vector with additional padding to match the layout of GLSL
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

impl From<[f32; 3]> for Vec3A {
    fn from(vector: [f32; 3]) -> Self {
        Vec3A {
            vector: vector.into(),
            _padding: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    /// Position of the camera, in world-space
    camera_origin: Vec3A,
    /// Direction that points to the right of the camera's viewport
    camera_right: Vec3A,
    /// Direction that points upward in the camera's viewport
    camera_up: Vec3A,
    /// Direction that points toward the direction the camera is facing
    camera_forward: Vec3A,

    /// Position of a point light in the scene
    light: PointLight,

    /// Number of seconds since the rendering started.
    time: f32,

    /// Number of frames the camera hasn't moved
    still_sample: u32,

    /// The number of frames that have been rendered
    frame_number: u32,

    /// Luminocity of the voxels that emit light
    emit_strength: f32,
    /// Luminocity of the sun
    sun_strength: f32,

    /// Size of the sun. Larger values result in softer shadows
    sun_size: f32,

    /// Rotation of the sun around the Y-axis (up)
    sun_yaw: f32,

    /// Rotation of the sun along the Y-axis
    sun_pitch: f32,

    /// Color of the sun
    sun_color: Vec3A,

    /// Color of the sky
    sky_color: Vec3A,

    /// How glossy surfaces are (might be split out into individual materials).
    specularity: f32,
}

impl Default for Uniforms {
    fn default() -> Self {
        Uniforms {
            camera_origin: [0.0; 3].into(),
            camera_right: [0.0; 3].into(),
            camera_up: [0.0; 3].into(),
            camera_forward: [0.0; 3].into(),
            light: PointLight {
                position: [0.0; 3].into(),
                brightness: 0.0,
            },
            time: 0.0,
            still_sample: 0,
            frame_number: 0,

            emit_strength: 4.0,
            sun_strength: 1.0,
            sun_size: 0.05,
            sun_yaw: 1.32,
            sun_pitch: 1.0,
            sun_color: [1.0; 3].into(),

            sky_color: [0.45, 0.6, 0.65].into(),

            specularity: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TemporalUniforms {
    /// How quickly to converge towards full blending. Larger values result in better more
    /// blending, and cleaner results, but increase temporal artifacts. Smaller values result in
    /// more up-to-date information reaching the screen, but takes longer to produce nice-looking
    /// results.
    sample_blending: f32,

    /// The maximum amount of blending. Large values result in old samples staying around for
    /// longer, which increases temporal artifacts. Small values result in less total blending.
    maximum_blending: f32,

    /// Don't blend across surfaces further away than this.
    blending_distance_cutoff: f32,
}

impl Default for TemporalUniforms {
    fn default() -> Self {
        TemporalUniforms {
            sample_blending: 0.5,
            maximum_blending: 0.98,
            blending_distance_cutoff: 1e-2,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PointLight {
    /// Position of the light
    position: Vec3,
    /// Brightness of the light
    brightness: f32,
}

struct Stopwatch {
    /// Previous time the stopwatch was ticked.
    prev_time: std::time::Instant,
}

impl Stopwatch {
    /// Create and start a new stopwatch
    pub fn new() -> Stopwatch {
        Stopwatch {
            prev_time: std::time::Instant::now(),
        }
    }

    /// Get the duration since the stopwatch last was "ticked" or created.
    pub fn tick(&mut self) -> std::time::Duration {
        let now = std::time::Instant::now();
        let duration = now.saturating_duration_since(self.prev_time);
        self.prev_time = now;
        duration
    }
}

/// Calculates the fps: frames per second. Updates at an even interval.
struct FpsCounter {
    /// How long since the frame counter was last updated.
    prev_time: std::time::Instant,
    /// How many frames have been rendered since the frame counter was last updated.
    frames: u32,
    /// The current fps
    fps: f32,
}

impl FpsCounter {
    pub fn new() -> Self {
        FpsCounter {
            prev_time: std::time::Instant::now(),
            frames: 0,
            fps: f32::NAN,
        }
    }

    /// Update the counter
    pub fn tick(&mut self) {
        let now = std::time::Instant::now();
        let elapsed = (now - self.prev_time).as_secs_f32();
        self.frames += 1;
        if elapsed > 0.25 {
            self.fps = self.frames as f32 / elapsed;
            self.prev_time = now;
            self.frames = 0;
        }
    }
}

// Context creation
impl Context {
    /// Create a new context with the given window.
    pub async fn new(window: Arc<winit::window::Window>) -> anyhow::Result<Context> {
        let gpu = Self::create_gpu_context(&window).await?;

        // We need to poll the device in the background
        std::thread::spawn({
            let device = gpu.device.clone();
            move || loop {
                device.poll(wgpu::Maintain::Wait)
            }
        });

        let output_size = window.inner_size();
        let swap_chain = Self::create_swap_chain(&gpu, output_size);
        let bindings = Self::create_bindings(&gpu, output_size, window.scale_factor() as f32)?;

        let bind_groups = Self::create_bind_groups(&gpu, &bindings);

        let pipeline = Self::create_render_pipeline(&bind_groups.render, &gpu)?;
        let voxel_pipeline = Self::create_voxel_pipeline(&bind_groups.voxel, &gpu)?;
        let temporal_pipeline = Self::create_temporal_pipeline(&bind_groups.temporal, &gpu)?;
        let denoise_pipeline = Self::create_denoise_pipeline(&bind_groups.denoise, &gpu)?;
        let gui_pipeline = Self::create_gui_pipeline(&bind_groups.gui, &gpu)?;

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
            temporal_pipeline,
            denoise_pipeline,
            gui_pipeline,

            start: std::time::Instant::now(),
            stopwatch: Stopwatch::new(),
            fps_counter: FpsCounter::new(),
            shader_watcher,

            input: UserInput {
                pressed_keys: HashSet::new(),
                cursor_grabbed: false,
                mouse_position: winit::dpi::LogicalPosition::new(-1.0, -1.0),
                modifiers: Default::default(),
            },

            camera,
            pitch: 0.0,
            yaw: 0.0,

            gui: Gui::new().context("failed to create GUI")?,
        })
    }

    /// Format that is used for the output image shown to the user
    pub const SWAP_CHAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

    /// Creates a GPU context
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
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::default(),
                    limits: wgpu::Limits {
                        max_storage_textures_per_shader_stage: 5,
                        ..Default::default()
                    },
                },
                None,
            )
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

    /// Given a set of voxels made up of position and color, construct the GPU representation of
    /// those voxels as an octree.
    fn create_octree_nodes(voxels: Vec<([i16; 3], [u8; 4])>) -> Vec<i32> {
        fn alloc_node(nodes: &mut Vec<i32>) -> usize {
            let index = nodes.len();
            nodes.extend_from_slice(&[0; 8]);
            index / 8
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
                    nodes[8 * current + octant] =
                        (1 << 31) | ((m & 0x7f) << 24) | (r << 16) | (g << 8) | b;
                    return;
                } else {
                    let value = nodes[8 * current + octant];
                    let child = match value {
                        0 => {
                            let node = alloc_node(nodes);
                            nodes[8 * current + octant] = node as i32;
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

        let depth = Self::voxel_depth(voxels.iter().map(|(pos, _)| *pos));
        let extent = 1 << depth;

        let mut nodes = Vec::new();
        let root = alloc_node(&mut nodes);

        for (pos, color) in voxels {
            insert_node(&mut nodes, root, [0; 3], extent, pos, color);
        }

        nodes
    }

    /// Create the GPU-representation of an octree, prepended with some additional information used
    /// for rendering
    fn create_octree(voxels: Vec<([i16; 3], [u8; 4])>) -> Vec<i32> {
        let depth = Self::voxel_depth(voxels.iter().map(|(pos, _)| *pos));
        let root_size = (1 << depth) as f32;
        let child_size = 1.0;

        let mut octree = vec![
            // center
            f32::to_bits(0.0) as i32,
            f32::to_bits(0.0) as i32,
            f32::to_bits(0.0) as i32,
            // root_size
            f32::to_bits(root_size) as i32,
            // child_size
            f32::to_bits(child_size) as i32,
        ];

        let nodes = Self::create_octree_nodes(voxels);
        octree.extend(nodes);
        octree
    }

    /// Update the GPU octree with the given voxels.
    fn recreate_octree(&mut self, voxels: Vec<([i16; 3], [u8; 4])>) -> anyhow::Result<()> {
        use wgpu::BufferUsage as Usage;

        let octree = Self::create_octree(voxels);
        self.bindings.octree_buffer =
            Buffer::new(&self.gpu, Usage::STORAGE | Usage::COPY_DST, &octree);
        self.bind_groups = Self::create_bind_groups(&self.gpu, &self.bindings);
        self.recreate_pipeline()
            .context("failed to recreate pipeline")?;

        Ok(())
    }

    /// Compute the required depth for an octree that contains all the positions
    fn voxel_depth(mut voxels: impl Iterator<Item = [i16; 3]>) -> u16 {
        let mut min;
        let mut max;

        match voxels.next() {
            None => return 0,
            Some([x, y, z]) => {
                min = x.min(y).min(z);
                max = x.max(y).max(z);
            }
        }

        for [x, y, z] in voxels {
            min = min.min(x).min(y).min(z);
            max = max.max(x).max(y).max(z);
        }

        let min_depth = (min.abs() as u16).next_power_of_two().trailing_zeros();
        let max_depth = (max.abs() as u16 + 1).next_power_of_two().trailing_zeros();

        u32::max(min_depth, max_depth) as u16
    }

    /// Randomly generate a set of voxels. This is the default model displayed when starting the
    /// application.
    fn create_voxels() -> Vec<([i16; 3], [u8; 4])> {
        let mut voxels = Vec::new();

        let radius = 32i32;

        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Generates a random color
        let mut color = |p: f32, _x: i32, _y: i32, _z: i32| {
            let red = rng.gen_range(50..=255);
            let green = rng.gen_range(50..=255);
            let blue = rng.gen_range(50..=255);

            let emmisive = rng.gen_bool(p as f64);
            let material = (emmisive as u8) << 6;

            [material, red, green, blue]
        };

        // Create a height-map
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

        // Get the height at the given coordinates
        let get_height = |x: i32, z: i32| {
            if x < -radius || x > radius || z < -radius || z > radius {
                None
            } else {
                let xi = (x + radius) as usize;
                let zi = (z + radius) as usize;
                heights[xi + zi * width]
            }
        };

        // Based on the height-map, create the voxels, making sure to fill in any voids produced by
        // large slopes in the height-map
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

        // Create a strip of light through the middle
        for x in -radius..=radius {
            voxels.push(([x as i16, -10, 0], [0x40, 255, 255, 255]));
        }

        voxels
    }

    /// Create a set of voxels from a MagicaVoxel "vox" model
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

    /// Create buffers for all the GPU resources
    fn create_bindings(
        gpu: &GpuContext,
        output_size: crate::Size,
        scale_factor: f32,
    ) -> anyhow::Result<Bindings> {
        use wgpu::BufferUsage as Usage;

        let mut uniforms = Uniforms::default();
        uniforms.light = PointLight {
            position: Vec3::new(0.4, -0.4, 0.02),
            brightness: 0.05,
        };

        let uniform_buffer = Buffer::new(gpu, Usage::UNIFORM | Usage::COPY_DST, &[uniforms]);
        let old_uniform_buffer = Buffer::new(gpu, Usage::UNIFORM | Usage::COPY_DST, &[uniforms]);

        let temporal_uniforms = UniformBuffer::new(gpu, TemporalUniforms::default());

        let octree = Self::create_octree(Self::create_voxels());
        let octree_buffer = Buffer::new(gpu, Usage::STORAGE | Usage::COPY_DST, &octree);

        let old_g_buffer = GBuffer::new(gpu, output_size);
        let new_g_buffer = GBuffer::new(gpu, output_size);

        let create_color_texture =
            || GBuffer::create_storage_texture(output_size, GBuffer::COLOR_FORMAT, gpu);

        let sampled_color = create_color_texture();
        let albedo = create_color_texture();
        let specular = create_color_texture();
        let denoised_color = create_color_texture();

        let denoise_uniforms = UniformBuffer::new(gpu, DenoiseUniforms::default());

        let (blue_noise_size, blue_noise_pixels) =
            Self::load_blue_noise("resources/blue-noise-128.zip")
                .context("failed to load blue noise")?;

        assert_eq!(
            blue_noise_size, 128,
            "blue noise images must have width and height set to 128"
        );

        let randomness_buffer =
            Buffer::new(gpu, Usage::STORAGE | Usage::COPY_DST, &blue_noise_pixels);

        let gui_uniforms = Buffer::new(
            gpu,
            Usage::UNIFORM | Usage::COPY_DST,
            &[GuiUniforms {
                width: output_size.width as f32 / scale_factor,
                height: output_size.height as f32 / scale_factor,
            }],
        );

        let near_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bindings = Bindings {
            old_uniform_buffer,
            uniform_buffer,
            uniforms,
            octree_buffer,
            randomness_buffer,

            old_g_buffer,
            new_g_buffer,

            near_sampler,

            temporal_uniforms,

            sampled_albedo: albedo,
            sampled_specular: specular,
            sampled_color,
            denoised_color,

            denoise_uniforms,

            gui_uniforms,
            gui_texture: GuiTexture::empty(gpu),
        };

        Ok(bindings)
    }

    /// Load blue noise images from disk, and return their size and contents appended in a single
    /// array
    fn load_blue_noise(path: &str) -> anyhow::Result<(usize, Vec<f32>)> {
        let file = std::fs::File::open(path)?;
        let mut archive = zip::ZipArchive::new(file)?;

        if archive.is_empty() {
            return Err(anyhow!("archive did not contain any files"));
        }

        let mut image_size = None;
        let mut pixels = Vec::new();

        let file_count = archive.len();

        for file_index in 0..file_count {
            let mut file = archive.by_index(file_index)?;
            if !file.is_file() {
                continue;
            }

            let (width, height) = Self::parse_raw_f32img(&mut file, &mut pixels)
                .with_context(|| format!("failed to read image: {}", file.name()))?;

            if width != height {
                return Err(anyhow!("found non-square blue noise image"))
                    .with_context(|| format!("while reading image: {}", file.name()));
            }

            match image_size.as_mut() {
                None => image_size = Some(width),
                Some(size) if *size != width => {
                    return Err(anyhow!(
                        "blue-noise images in archive do not have same size"
                    ))
                }
                _ => {}
            }
        }

        match image_size {
            None => Err(anyhow!("archive did not contain any images")),
            Some(size) => Ok((size, pixels)),
        }
    }

    /// Parse the custom file format used for storing blue-noise textures
    fn parse_raw_f32img(
        r: &mut impl std::io::Read,
        pixels: &mut Vec<f32>,
    ) -> anyhow::Result<(usize, usize)> {
        fn read_u32(r: &mut impl std::io::Read) -> anyhow::Result<u32> {
            let mut buffer = [0u8; 4];
            r.read_exact(&mut buffer)?;
            Ok(u32::from_be_bytes(buffer))
        }

        let width = read_u32(r)? as usize;
        let height = read_u32(r)? as usize;

        let pixel_count = width * height;
        let old_len = pixels.len();
        let new_len = old_len + pixel_count;
        pixels.resize(new_len, 0.0);

        let pixel_buffer = bytemuck::cast_slice_mut(&mut pixels[old_len..new_len]);
        if let Err(e) = r.read_exact(pixel_buffer) {
            pixels.truncate(old_len);
            return Err(e.into());
        }

        for pixel in &mut pixels[old_len..new_len] {
            *pixel = f32::from_bits(u32::from_be(f32::to_bits(*pixel)));
        }

        Ok((width, height))
    }

    /// Create bind groups for all render stages
    fn create_bind_groups(gpu: &GpuContext, bindings: &Bindings) -> BindGroups {
        BindGroups {
            render: Self::create_render_bind_group(gpu, bindings),
            voxel: Self::create_voxel_bind_group(gpu, bindings),
            temporal: Self::create_temporal_bind_group(gpu, bindings),
            denoise: Self::create_denoise_bind_group(gpu, bindings),
            gui: Self::create_gui_bind_group(gpu, bindings),
        }
    }

    /// See `shaders/display.frag` for more details on the different bindings
    fn create_render_bind_group(gpu: &GpuContext, bindings: &Bindings) -> BindGroup {
        let color = util::view(&bindings.denoised_color);
        let (layout, bindings) = bind_group![
            UniformImage(0 => (&color, ReadOnly, GBuffer::COLOR_FORMAT, D2) in FRAGMENT),
        ];

        BindGroup::from_entries(&layout, &bindings, gpu)
    }

    /// See `shaders/voxel.comp` for more details on the different bindings
    fn create_voxel_bind_group(gpu: &GpuContext, bindings: &Bindings) -> BindGroup {
        let sampled_color = util::view(&bindings.sampled_color);
        let albedo = util::view(&bindings.sampled_albedo);
        let specular = util::view(&bindings.sampled_specular);
        let new_normal_depth = util::view(&bindings.new_g_buffer.normal_depth);

        let (layout, bindings) = bind_group![
            UniformImage(0 => (&sampled_color, WriteOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
            UniformImage(1 => (&new_normal_depth, WriteOnly, GBuffer::NORMAL_DEPTH_FORMAT, D2) in COMPUTE),
            UniformImage(2 => (&albedo, WriteOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
            UniformImage(3 => (&specular, WriteOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
            Uniform(4 => (&bindings.uniform_buffer) in COMPUTE),
            Uniform(5 => (&bindings.old_uniform_buffer) in COMPUTE),
            Storage(6 => (&bindings.octree_buffer, read_only: true) in COMPUTE),
            Storage(7 => (&bindings.randomness_buffer, read_only: true) in COMPUTE),
        ];

        BindGroup::from_entries(&layout, &bindings, gpu)
    }

    /// See `shaders/temporal.comp` for more details on the different bindings
    fn create_temporal_bind_group(gpu: &GpuContext, bindings: &Bindings) -> BindGroup {
        let old_images = &bindings.old_g_buffer;
        let old_color = util::view(&old_images.color);
        let old_normal_depth = util::view(&old_images.normal_depth);
        let old_specular = util::view(&old_images.specular);

        let new_images = &bindings.new_g_buffer;
        let new_color = util::view(&new_images.color);
        let new_normal_depth = util::view(&new_images.normal_depth);
        let new_specular = util::view(&new_images.specular);

        let sampled_color = util::view(&bindings.sampled_color);
        let sampled_specular = util::view(&bindings.sampled_specular);

        let (layout, bindings) = bind_group![
            Sampler(0 => (&bindings.near_sampler) in COMPUTE),

            Texture(1 => (&old_color, Float { filterable: true }, D2) in COMPUTE),
            Texture(2 => (&old_specular, Float { filterable: true }, D2) in COMPUTE),

            UniformImage(3 => (&sampled_color, ReadOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
            UniformImage(4 => (&sampled_specular, ReadOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),

            UniformImage(5 => (&new_color, WriteOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
            UniformImage(6 => (&new_specular, WriteOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),

            Texture(7 => (&old_normal_depth, Float { filterable: true }, D2) in COMPUTE),
            UniformImage(8 => (&new_normal_depth, ReadOnly, GBuffer::NORMAL_DEPTH_FORMAT, D2) in COMPUTE),

            Uniform(9 => (&bindings.temporal_uniforms.buffer) in COMPUTE),
            Uniform(10 => (&bindings.uniform_buffer) in COMPUTE),
            Uniform(11 => (&bindings.old_uniform_buffer) in COMPUTE),
        ];

        BindGroup::from_entries(&layout, &bindings, gpu)
    }

    /// See `shaders/denoise.comp` for more details on the different bindings
    fn create_denoise_bind_group(gpu: &GpuContext, bindings: &Bindings) -> BindGroup {
        let denoise_color = util::view(&bindings.denoised_color);
        let new_color = util::view(&bindings.new_g_buffer.color);
        let new_normal_depth = util::view(&bindings.new_g_buffer.normal_depth);
        let albedo = util::view(&bindings.sampled_albedo);
        let specular = util::view(&bindings.new_g_buffer.specular);

        let (layout, bindings) = bind_group![
            UniformImage(0 => (&denoise_color, WriteOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
            UniformImage(1 => (&new_color, ReadOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
            UniformImage(2 => (&new_normal_depth, ReadOnly, GBuffer::NORMAL_DEPTH_FORMAT, D2) in COMPUTE),
            UniformImage(3 => (&albedo, ReadOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
            UniformImage(4 => (&specular, ReadOnly, GBuffer::COLOR_FORMAT, D2) in COMPUTE),
            Uniform(5 => (&bindings.uniform_buffer) in COMPUTE),
            Uniform(6 => (&bindings.denoise_uniforms.buffer) in COMPUTE),
        ];

        BindGroup::from_entries(&layout, &bindings, gpu)
    }

    /// See `shaders/gui.frag` for more details on the different bindings
    fn gui_bindings<'a>(
        bindings: &'a Bindings,
        texture: &'a wgpu::TextureView,
    ) -> (
        [wgpu::BindGroupLayoutEntry; 3],
        [wgpu::BindGroupEntry<'a>; 3],
    ) {
        bind_group![
            Uniform(0 => (&bindings.gui_uniforms) in VERTEX),
            Sampler(1 => (&bindings.near_sampler) in FRAGMENT),
            Texture(2 => (texture, Float { filterable: true }, D2) in FRAGMENT),
        ]
    }

    /// See `shaders/gui.frag` for more details on the different bindings
    fn create_gui_bind_group(gpu: &GpuContext, bindings: &Bindings) -> BindGroup {
        let gui_texture = util::view(&bindings.gui_texture.texture);
        let (layout, bindings) = Self::gui_bindings(bindings, &gui_texture);
        BindGroup::from_entries(&layout, &bindings, gpu)
    }

    /// Update the GUI bindings
    fn update_gui_bind_group(gpu: &GpuContext, bindings: &Bindings, bind_group: &mut BindGroup) {
        let gui_texture = util::view(&bindings.gui_texture.texture);
        let (_layout, bindings) = Self::gui_bindings(bindings, &gui_texture);
        bind_group.update_bindings(&bindings, gpu);
    }

    /// Create a compute pipeline using the given shader
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

    fn create_temporal_pipeline(
        bind_group: &BindGroup,
        gpu: &GpuContext,
    ) -> anyhow::Result<wgpu::ComputePipeline> {
        Self::create_compute_pipeline(bind_group, "shaders/temporal.comp", gpu)
    }

    fn create_denoise_pipeline(
        bind_group: &BindGroup,
        gpu: &GpuContext,
    ) -> anyhow::Result<wgpu::ComputePipeline> {
        Self::create_compute_pipeline(bind_group, "shaders/denoise.comp", gpu)
    }

    /// Create a pipeline for rendering the GUI
    fn create_gui_pipeline(
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

        let vertex_module = shader::create_shader_module(gpu, "shaders/gui.vert")?;
        let fragment_module = shader::create_shader_module(gpu, "shaders/gui.frag")?;

        let vertex = wgpu::VertexState {
            module: &vertex_module,
            entry_point: "main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GuiVertex>() as u64,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![
                    0 => Float2,
                    1 => Float2,
                    2 => Uint,
                ],
            }],
        };

        let fragment = wgpu::FragmentState {
            module: &fragment_module,
            entry_point: "main",
            targets: &[wgpu::ColorTargetState {
                format: Self::SWAP_CHAIN_FORMAT,
                alpha_blend: wgpu::BlendState::REPLACE,
                color_blend: wgpu::BlendState {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
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

    /// Create a pipeline for outputting the result to screen.
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

    /// Recreate all pipelines
    fn recreate_pipeline(&mut self) -> anyhow::Result<()> {
        info!("recreating render pipeline");
        self.pipeline = Self::create_render_pipeline(&self.bind_groups.render, &self.gpu)?;

        info!("recreating voxel pipeline");
        self.voxel_pipeline = Self::create_voxel_pipeline(&self.bind_groups.voxel, &self.gpu)?;

        info!("recreating temporal pipeline");
        self.temporal_pipeline =
            Self::create_temporal_pipeline(&self.bind_groups.temporal, &self.gpu)?;

        info!("recreating denoise pipeline");
        self.denoise_pipeline =
            Self::create_denoise_pipeline(&self.bind_groups.denoise, &self.gpu)?;

        info!("recreating gui pipeline");
        self.gui_pipeline = Self::create_gui_pipeline(&self.bind_groups.gui, &self.gpu)?;

        self.bindings.uniforms.still_sample = 0;

        Ok(())
    }

    /// Called when the window is resized. Recreates the rendering context to match the new size.
    fn resize(&mut self, new_size: crate::Size) -> anyhow::Result<()> {
        if self.output_size == new_size {
            return Ok(());
        }

        info!(?new_size, "window was resized");
        self.output_size = new_size;

        self.recreate_swap_chain();

        self.bindings.old_g_buffer = GBuffer::new(&self.gpu, self.output_size);
        self.bindings.new_g_buffer = GBuffer::new(&self.gpu, self.output_size);

        let create_color_texture =
            |gpu| GBuffer::create_storage_texture(new_size, GBuffer::COLOR_FORMAT, gpu);

        self.bindings.sampled_color = create_color_texture(&self.gpu);
        self.bindings.sampled_albedo = create_color_texture(&self.gpu);
        self.bindings.sampled_specular = create_color_texture(&self.gpu);
        self.bindings.denoised_color = create_color_texture(&self.gpu);

        self.bindings.gui_uniforms.write(
            &self.gpu,
            0,
            &[GuiUniforms {
                width: new_size.width as f32 / self.window.scale_factor() as f32,
                height: new_size.height as f32 / self.window.scale_factor() as f32,
            }],
        );

        self.bind_groups = Self::create_bind_groups(&self.gpu, &self.bindings);
        self.recreate_pipeline()
    }
}

// Event handling
impl Context {
    /// Called when the user interacts with the window, or we should render a new frame
    pub fn handle_event(
        &mut self,
        event: winit::event::Event<()>,
        flow: &mut winit::event_loop::ControlFlow,
    ) -> anyhow::Result<()> {
        use winit::event::{DeviceEvent, Event, WindowEvent};
        use winit::event_loop::ControlFlow;

        #[allow(clippy::single_match, clippy::collapsible_match)]
        match event {
            // All events are handled, so produce the next frame
            Event::MainEventsCleared => {
                self.check_shader_updates()?;

                self.fps_counter.tick();
                let dt = self.stopwatch.tick().as_secs_f32();

                self.update_gui(dt)?;
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
                WindowEvent::ReceivedCharacter(ch) => {
                    if !self.input.cursor_grabbed && !ch.is_control() {
                        self.gui.events.push(egui::Event::Text(ch.into()))
                    }
                }
                WindowEvent::KeyboardInput { input, .. } => self.keyboard_input(input, flow)?,
                WindowEvent::CursorMoved { position, .. } => {
                    let logical = position.to_logical(self.window.scale_factor());
                    self.input.mouse_position = logical;
                    if !self.input.cursor_grabbed {
                        self.gui.events.push(egui::Event::PointerMoved(
                            [logical.x as f32, logical.y as f32].into(),
                        ))
                    }
                }
                WindowEvent::ModifiersChanged(modifiers) => {
                    self.input.modifiers = modifiers;
                }
                WindowEvent::MouseInput { state, button, .. } => self.mouse_input(state, button),
                _ => {}
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                    if self.input.cursor_grabbed {
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

    /// Called when a mouse button was pressed
    fn mouse_input(
        &mut self,
        state: winit::event::ElementState,
        button: winit::event::MouseButton,
    ) {
        let egui_button = match button {
            winit::event::MouseButton::Left => Some(egui::PointerButton::Primary),
            winit::event::MouseButton::Right => Some(egui::PointerButton::Secondary),
            winit::event::MouseButton::Middle => Some(egui::PointerButton::Middle),
            winit::event::MouseButton::Other(_) => None,
        };

        if let Some(button) = egui_button {
            let pos = [
                self.input.mouse_position.x as f32,
                self.input.mouse_position.y as f32,
            ];

            if !self.input.cursor_grabbed {
                self.gui.events.push(egui::Event::PointerButton {
                    pos: pos.into(),
                    button,
                    pressed: matches!(state, winit::event::ElementState::Pressed),
                    modifiers: Self::egui_modifiers(self.input.modifiers),
                })
            }
        }
    }

    /// Called when there was some keyboard input sent to the window
    fn keyboard_input(
        &mut self,
        input: winit::event::KeyboardInput,
        flow: &mut winit::event_loop::ControlFlow,
    ) -> anyhow::Result<()> {
        use winit::event::{ElementState::Pressed, VirtualKeyCode};

        if let Some(key) = input.virtual_keycode {
            if !self.input.cursor_grabbed {
                let egui_key = match key {
                    VirtualKeyCode::Return => Some(egui::Key::Enter),
                    VirtualKeyCode::Back => Some(egui::Key::Backspace),
                    VirtualKeyCode::Delete => Some(egui::Key::Delete),
                    VirtualKeyCode::Left => Some(egui::Key::ArrowLeft),
                    VirtualKeyCode::Right => Some(egui::Key::ArrowRight),
                    VirtualKeyCode::Up => Some(egui::Key::ArrowUp),
                    VirtualKeyCode::Down => Some(egui::Key::ArrowDown),
                    VirtualKeyCode::Home => Some(egui::Key::Home),
                    VirtualKeyCode::End => Some(egui::Key::End),
                    VirtualKeyCode::Tab => Some(egui::Key::Tab),
                    _ => None,
                };

                if let Some(key) = egui_key {
                    self.gui.events.push(egui::Event::Key {
                        key,
                        pressed: matches!(input.state, Pressed),
                        modifiers: Self::egui_modifiers(self.input.modifiers),
                    })
                }
            }
        }

        if let Ok(key) = KeyCode::try_from(input.scancode) {
            match input.state {
                winit::event::ElementState::Pressed => {
                    self.input.pressed_keys.insert(key);
                }
                winit::event::ElementState::Released => {
                    self.input.pressed_keys.remove(&key);
                }
            }

            match key {
                KeyCode::Escape => *flow = winit::event_loop::ControlFlow::Exit,
                KeyCode::Tab if input.state == Pressed => {
                    self.input.cursor_grabbed = !self.input.cursor_grabbed;
                    if self
                        .window
                        .set_cursor_grab(self.input.cursor_grabbed)
                        .is_ok()
                    {
                        self.window.set_cursor_visible(!self.input.cursor_grabbed);
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Convert window modifiers to GUI modifiers
    fn egui_modifiers(modifiers: winit::event::ModifiersState) -> egui::Modifiers {
        egui::Modifiers {
            alt: modifiers.alt(),
            ctrl: modifiers.ctrl(),
            shift: modifiers.shift(),
            mac_cmd: false,
            command: modifiers.ctrl(),
        }
    }

    /// Check if any of the shaders have changed. If so, reload them.
    fn check_shader_updates(&mut self) -> anyhow::Result<()> {
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

        Ok(())
    }

}

// Rendering
impl Context {
    /// Update the gui, updating events, and collecting meshes that need to be drawn.
    pub fn update_gui(&mut self, dt: f32) -> anyhow::Result<()> {
        let screen_size = [
            self.output_size.width as f32,
            self.output_size.height as f32,
        ];
        let screen_rect = egui::Rect::from_min_size([0.0, 0.0].into(), screen_size.into());

        let raw_input = egui::RawInput {
            screen_rect: Some(screen_rect),
            pixels_per_point: Some(self.window.scale_factor() as f32),
            time: None,
            predicted_dt: dt,
            modifiers: egui::Modifiers::default(),
            events: std::mem::take(&mut self.gui.events),
            ..Default::default()
        };

        self.gui.ctx.begin_frame(raw_input);
        self.draw_ui()?;

        let (_output, shapes) = self.gui.ctx.end_frame();
        let gui_meshes = self.gui.ctx.tessellate(shapes);
        self.build_gui_meshes(gui_meshes);
        self.update_gui_textures();

        Ok(())
    }

    /// Draw the UI, but don't render. Meshes are collected later.
    fn draw_ui(&mut self) -> anyhow::Result<()> {
        let style = self.gui.ctx.style();
        let ctx = self.gui.ctx.clone();

        let mut model_changed = false;

        egui::Window::new("debug")
            .frame({
                let mut frame = egui::Frame::window(&style).multiply_with_opacity(0.8);
                frame.shadow = egui::epaint::Shadow::small();
                frame
            })
            .collapsible(true)
            .resizable(true)
            .fixed_pos([5.0, 5.0])
            .show(&ctx, |ui| {
                ui.label(format!("fps: {}", self.fps_counter.fps));

                ui.collapsing("scene", |ui| {
                    let uniforms = &mut self.bindings.uniforms;
                    ui.collapsing("lighting", |ui| {
                        ui.collapsing("sun", |ui| {
                            Self::color_picker_hue(ui, "sun color", &mut uniforms.sun_color);
                            Self::slider(
                                ui,
                                "sun strength",
                                &mut uniforms.sun_strength,
                                0.0..=10.0,
                            );
                            Self::slider(ui, "sun size", &mut uniforms.sun_size, 0.0..=1.0);

                            ui.separator();

                            Self::slider_angle(ui, "sun yaw", &mut uniforms.sun_yaw, 0.0..=360.0);
                            Self::slider_angle(
                                ui,
                                "sun pitch",
                                &mut uniforms.sun_pitch,
                                -90.0..=90.0,
                            );
                        });

                        ui.collapsing("sky", |ui| {
                            Self::color_picker_hue(ui, "sky color", &mut uniforms.sky_color);
                        });

                        ui.collapsing("emmision", |ui| {
                            Self::slider(
                                ui,
                                "emit strength",
                                &mut uniforms.emit_strength,
                                0.0..=40.0,
                            );
                        });
                    });

                    ui.collapsing("materials", |ui| {
                        Self::slider(ui, "specularity", &mut uniforms.specularity, 0.0..=1.0);
                    });

                    ui.collapsing("model", |ui| {
                        let current = &mut self.gui.current_model;
                        let vox_files = &self.gui.vox_files;

                        egui::ComboBox::from_label("current")
                            .selected_text(current.to_string())
                            .show_ui(ui, |ui| {
                                ui.allocate_at_least(
                                    [100.0, 0.0].into(),
                                    egui::Sense::click_and_drag(),
                                );

                                let default =
                                    ui.selectable_value(current, Model::Default, "default");
                                if default.clicked() {
                                    model_changed = true;
                                }

                                for path in vox_files.iter() {
                                    let model = Model::Vox(path.clone());
                                    let name = match path.file_stem() {
                                        Some(stem) => stem.to_string_lossy().to_string(),
                                        None => path.display().to_string(),
                                    };
                                    if ui.selectable_value(current, model, name).clicked() {
                                        model_changed = true;
                                    }
                                }
                            });
                    });
                });

                ui.collapsing("renderer", |ui| {
                    ui.collapsing("temporal blending", |ui| {
                        let uniforms = &mut self.bindings.temporal_uniforms;
                        Self::slider(ui, "factor", &mut uniforms.sample_blending, 0.0..=1.0);
                        Self::slider(ui, "maximum", &mut uniforms.maximum_blending, 0.0..=1.0);
                        Self::slider_log(
                            ui,
                            "distance cutoff",
                            &mut uniforms.blending_distance_cutoff,
                            0.0..=1.0,
                        );
                    });

                    ui.collapsing("denoiser", |ui| {
                        let uniforms = &mut self.bindings.denoise_uniforms;
                        Self::slider(ui, "radius", &mut uniforms.radius, 0..=8);
                        Self::slider(
                            ui,
                            "sigma distance",
                            &mut uniforms.sigma_distance,
                            0.1..=5.0,
                        );
                        Self::slider(ui, "sigma range", &mut uniforms.sigma_range, 0.1..=5.0);
                    });

                    ui.collapsing("composition", |ui| {
                        Self::slider(
                            ui,
                            "albedo",
                            &mut self.bindings.denoise_uniforms.albedo_factor,
                            0.0..=1.0,
                        );
                    })
                });
            });

        if model_changed {
            match &self.gui.current_model {
                Model::Default => {
                    self.recreate_octree(Self::create_voxels())?;
                }
                Model::Vox(path) => match crate::vox::load(path) {
                    Err(e) => error!("failed to load vox: {:?}", e),
                    Ok(vox) => {
                        self.recreate_octree(Self::voxels_from_vox(&vox)).unwrap();
                    }
                },
            }
        }

        Ok(())
    }

    /// A slider over a range of values
    fn slider<T>(ui: &mut egui::Ui, name: &str, value: &mut T, range: std::ops::RangeInclusive<T>)
    where
        T: egui::emath::Numeric,
    {
        ui.add(
            egui::Slider::new(value, range)
                .text(name)
                .clamp_to_range(true),
        );
    }

    /// A slider over a range of values, logarithmic
    fn slider_log<T>(
        ui: &mut egui::Ui,
        name: &str,
        value: &mut T,
        range: std::ops::RangeInclusive<T>,
    ) where
        T: egui::emath::Numeric,
    {
        ui.add(egui::Slider::new(value, range).text(name).logarithmic(true));
    }

    /// A slider of a range of angles
    fn slider_angle(
        ui: &mut egui::Ui,
        name: &str,
        value: &mut f32,
        range: std::ops::RangeInclusive<f32>,
    ) {
        let mut degrees = value.to_degrees();
        ui.add(
            egui::Slider::new(&mut degrees, range)
                .text(name)
                .clamp_to_range(true)
                .suffix("°"),
        );

        *value = degrees.to_radians();
    }

    /// Create a new color picker with a given label.
    fn color_picker_hue(ui: &mut egui::Ui, name: &str, color: &mut Vec3A) {
        ui.horizontal(|ui| {
            let mut hsva = egui::color::Hsva::from_rgb(color.vector.into());
            egui::color_picker::color_edit_button_hsva(
                ui,
                &mut hsva,
                egui::color_picker::Alpha::Opaque,
            );
            ui.label(name);
            *color = hsva.to_rgb().into();
        });
    }

    /// Create the meshes that should be rendered as the GUI
    fn build_gui_meshes(&mut self, meshes: Vec<egui::ClippedMesh>) {
        self.gui.meshes.clear();
        self.gui.meshes.reserve(meshes.len());

        for egui::ClippedMesh(_clip, mesh) in meshes {
            let indices = Buffer::new(&self.gpu, wgpu::BufferUsage::INDEX, &mesh.indices);

            let vertices = mesh
                .vertices
                .into_iter()
                .map(|vertex| GuiVertex {
                    position: vertex.pos.into(),
                    tex_coord: vertex.uv.into(),
                    color: vertex.color.to_array(),
                })
                .collect::<Vec<_>>();
            let vertices = Buffer::new(&self.gpu, wgpu::BufferUsage::VERTEX, vertices.as_slice());

            self.gui.meshes.push(GuiMesh { indices, vertices })
        }
    }

    /// Check if the textures used when rendering the GUI have changed
    fn update_gui_textures(&mut self) {
        let new = self.gui.ctx.texture();
        match &mut self.bindings.gui_texture {
            old if old.version == Some(new.version) => { /* up to date */ }
            old if old.size.width == new.width as u32 && old.size.height == new.height as u32 => {
                // upload new texture data
                self.gpu.queue.write_texture(
                    wgpu::TextureCopyView {
                        texture: &old.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                    },
                    &new.pixels,
                    wgpu::TextureDataLayout {
                        offset: 0,
                        bytes_per_row: old.size.width,
                        rows_per_image: old.size.height,
                    },
                    wgpu::Extent3d {
                        width: old.size.width,
                        height: old.size.height,
                        depth: 1,
                    },
                );

                old.version = Some(new.version);
            }
            old => {
                let new_size = crate::Size::new(new.width as u32, new.height as u32);
                // recreate texture with new size
                let texture = util::create_texture_with_data(
                    new_size,
                    Gui::TEXTURE_FORMAT,
                    wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
                    &self.gpu,
                    &new.pixels,
                );

                *old = GuiTexture {
                    version: Some(new.version),
                    size: new_size,
                    texture,
                };

                Self::update_gui_bind_group(&self.gpu, &self.bindings, &mut self.bind_groups.gui);
            }
        }
    }

    /// Perform per-frame updates of the camera
    pub fn update(&mut self, dt: f32) {
        self.camera.direction = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        );

        let pressed =
            |keys: &[KeyCode]| keys.iter().any(|key| self.input.pressed_keys.contains(key));

        let [right, _, forward] = self.camera.axis();
        let mut movement = Vec3::zero();

        if pressed(&[KeyCode::W]) {
            movement += forward;
        }
        if pressed(&[KeyCode::S]) {
            movement -= forward;
        }
        if pressed(&[KeyCode::D]) {
            movement += right;
        }
        if pressed(&[KeyCode::A]) {
            movement -= right;
        }
        if pressed(&[KeyCode::E]) {
            movement.y += 1.0;
        }
        if pressed(&[KeyCode::Q]) {
            movement.y -= 1.0;
        }
        if movement != Vec3::zero() {
            let speed = if pressed(&[KeyCode::LeftControl]) {
                0.5
            } else if pressed(&[KeyCode::LeftShift]) {
                50.0
            } else {
                5.0
            };
            self.camera.position += speed * dt * movement.norm();
            self.bindings.uniforms.still_sample = 0;
        }
    }

    /// Render everything
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

            let dispatch_screen = |cpass: &mut wgpu::ComputePass, local_x, local_y| {
                let groups_x = (self.output_size.width + local_x - 1) / local_x;
                let groups_y = (self.output_size.height + local_y - 1) / local_y;
                cpass.dispatch(groups_x, groups_y, 1);
            };

            // Path-trace
            cpass.set_pipeline(&self.voxel_pipeline);
            cpass.set_bind_group(0, &self.bind_groups.voxel.bindings, &[]);
            dispatch_screen(&mut cpass, 16, 16);

            // Temporal blending
            cpass.set_pipeline(&self.temporal_pipeline);
            cpass.set_bind_group(0, &self.bind_groups.temporal.bindings, &[]);
            dispatch_screen(&mut cpass, 16, 16);

            // Denoising
            cpass.set_pipeline(&self.denoise_pipeline);
            cpass.set_bind_group(0, &self.bind_groups.denoise.bindings, &[]);
            dispatch_screen(&mut cpass, 16, 16);
        }

        // Copy the newly created buffers to the new ones
        self.bindings
            .old_g_buffer
            .copy_from(&self.bindings.new_g_buffer, &mut encoder);

        // display the output
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

        // draw the gui
        self.render_gui(&mut encoder, &output.view);

        // submit the commands to the GPU
        let command_buffer = encoder.finish();
        self.gpu.queue.submit(Some(command_buffer));

        drop(output);

        Ok(())
    }

    /// Render the GUI using the meshes computed in `update_ui`
    fn render_gui<'a>(
        &'a mut self,
        encoder: &'a mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: output,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        rpass.set_pipeline(&self.gui_pipeline);
        rpass.set_bind_group(0, &self.bind_groups.gui.bindings, &[]);

        for mesh in self.gui.meshes.iter() {
            rpass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            rpass.set_vertex_buffer(0, mesh.vertices.slice(..));
            rpass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }
    }

    /// Get the next frame from the swapchain that will be used to render to.
    fn get_next_frame(&mut self) -> anyhow::Result<wgpu::SwapChainTexture> {
        for _attempt in 0..8 {
            match self.swap_chain.get_current_frame() {
                Err(wgpu::SwapChainError::Outdated) => {
                    let _ = info_span!("swap chain outdated");
                    self.recreate_swap_chain();
                }
                Err(wgpu::SwapChainError::Lost) => {
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

    /// Recreate the swapchain
    fn recreate_swap_chain(&mut self) {
        info!("recreating surface");
        self.gpu.surface = Self::create_surface(&self.gpu.instance, &self.window);
        info!("recreating swap chain");
        self.swap_chain = Self::create_swap_chain(&self.gpu, self.output_size);
    }

    /// Upload all data to the GPU-side bindings that will be needed for the next render
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
            frame_number: self.bindings.uniforms.frame_number.wrapping_add(1),
            ..self.bindings.uniforms
        };

        self.bindings
            .uniform_buffer
            .write(&self.gpu, 0, &[self.bindings.uniforms]);

        self.bindings.temporal_uniforms.upload(&self.gpu);
        self.bindings.denoise_uniforms.upload(&self.gpu);
    }
}

fn color(r: f64, g: f64, b: f64, a: f64) -> wgpu::Color {
    wgpu::Color { r, g, b, a }
}

impl BindGroup {
    /// Create a new bind group using the output of the `bind_group` macro.
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

    /// Update the bind group without changing the layout
    pub fn update_bindings(&mut self, new_bindings: &[wgpu::BindGroupEntry], gpu: &GpuContext) {
        self.bindings = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: new_bindings,
        });
    }
}

impl DirectoryWatcher {
    /// Watch over a directory, notifying on changes.
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
    /// The format used to store colors
    pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

    /// The format used to normals and depths
    pub const NORMAL_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

    /// Create a new buffer with the given size.
    fn new(gpu: &GpuContext, size: crate::Size) -> GBuffer {
        let color = Self::create_storage_texture(size, GBuffer::COLOR_FORMAT, gpu);
        let normal_depth = Self::create_storage_texture(size, GBuffer::NORMAL_DEPTH_FORMAT, gpu);
        let specular = Self::create_storage_texture(size, GBuffer::COLOR_FORMAT, gpu);

        GBuffer {
            size,
            color,
            normal_depth,
            specular,
        }
    }

    /// Create a texture used to store per-pixel information
    fn create_storage_texture(
        size: crate::Size,
        format: wgpu::TextureFormat,
        gpu: &GpuContext,
    ) -> wgpu::Texture {
        use wgpu::TextureUsage as Usage;
        util::create_texture(
            size,
            format,
            Usage::COPY_SRC | Usage::COPY_DST | Usage::STORAGE | Usage::SAMPLED,
            gpu,
        )
    }

    /// Copy the date from `other` into this buffer
    fn copy_from(&self, other: &GBuffer, encoder: &mut wgpu::CommandEncoder) {
        util::copy_entire_texture_to_texture(&other.color, &self.color, self.size, encoder);
        util::copy_entire_texture_to_texture(
            &other.normal_depth,
            &self.normal_depth,
            self.size,
            encoder,
        );
        util::copy_entire_texture_to_texture(&other.specular, &self.specular, self.size, encoder);
    }
}
