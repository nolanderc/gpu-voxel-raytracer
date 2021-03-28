#[macro_use]
extern crate tracing;

#[macro_use]
mod macros;

mod context;
mod camera;
mod color;
mod cpu;
mod linear;

use crate::camera::Camera;
use crate::color::Color;
use crate::linear::Vec3;

use std::sync::Arc;

trait Backend {
    fn from_voxels(voxels: Vec<(Coord, Color)>) -> Self;

    fn render(&self, camera: Camera, size: Size, pixels: &mut [Color], time: f32);
}

type Size = winit::dpi::PhysicalSize<u32>;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Coord {
    x: u16,
    y: u16,
    z: u16,
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct Ray {
    origin: Vec3,
    direction: Vec3,
}

fn main() -> anyhow::Result<()> {
    setup_panic_handler();

    init_log_subscriber();

    let size = Size {
        width: 800,
        height: 600,
    };

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("voxels")
        .with_resizable(false)
        .with_inner_size(size)
        .build(&event_loop)
        .expect("could not open window");
    let window = Arc::new(window);

    let mut context = pollster::block_on(crate::context::Context::new(window))?;
    
    event_loop.run(move |event, _, flow| {
        if let Err(e) = context.handle_event(event, flow) {
            eprintln!("ERROR: {:#?}", e);
        }
    })
}

// In case there's a panic, take down the whole program
fn setup_panic_handler() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        original_hook(panic_info);
        std::process::exit(1);
    }));
}

fn init_log_subscriber() {
    let mut filter = tracing_subscriber::EnvFilter::default()
        // if not specified otherwise, only display warnings
        .add_directive(tracing_subscriber::filter::LevelFilter::WARN.into())
        // print info from this executable
        .add_directive("voxel=info".parse().unwrap());

    if let Ok(text) = std::env::var("RUST_LOG") {
        filter = filter.add_directive(text.parse().unwrap());
    }

    tracing_subscriber::fmt::Subscriber::builder()
        .with_env_filter(filter)
        .pretty()
        .init()
}

fn create_voxels() -> Vec<(Coord, Color)> {
    let mut voxels = Vec::new();

    let size: u16 = 127;
    let center = size / 2;

    let radius = 5;
    let tree_height = 9;

    for x in 0..size {
        for z in 0..size {
            let cx = x as i32 - center as i32;
            let cz = z as i32 - center as i32;
            let y = (cx.abs() + cz.abs()) / 7;
            voxels.push(([x, y as u16, z].into(), [50, 200, 10].into()));
        }
    }

    for y in 1..tree_height + radius {
        voxels.push(([center, y, center].into(), [100, 50, 10].into()));
    }

    for x in center - radius..=center + radius {
        for z in center - radius..=center + radius {
            for y in tree_height - radius..=tree_height + radius {
                let cx: i32 = x as i32 - center as i32;
                let cy: i32 = y as i32 - tree_height as i32;
                let cz: i32 = z as i32 - center as i32;
                if cx.pow(2) + cy.pow(2) + cz.pow(2) < radius.pow(2) as i32 {
                    voxels.push(([x as u16, y as u16, z as u16].into(), [10, 100, 10].into()));
                }
            }
        }
    }

    voxels
}

impl From<[u16; 3]> for Coord {
    fn from([x, y, z]: [u16; 3]) -> Self {
        Coord { x, y, z }
    }
}
