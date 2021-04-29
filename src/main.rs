#[macro_use]
extern crate tracing;

#[macro_use]
mod macros;

mod vox;
mod scancodes;

mod context;
mod camera;
mod linear;

use std::sync::Arc;

type Size = winit::dpi::PhysicalSize<u32>;

fn main() -> anyhow::Result<()> {
    setup_panic_handler();

    init_log_subscriber();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("voxels")
        .with_resizable(true)
        .with_inner_size(winit::dpi::LogicalSize::new(800, 800))
        .build(&event_loop)
        .expect("could not open window");
    let window = Arc::new(window);
    
    let mut context = pollster::block_on(crate::context::Context::new(window))?;
    
    event_loop.run(move |event, _, flow| {
        if let Err(e) = context.handle_event(event, flow) {
            eprintln!("ERROR: {:?}", e);
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
        .add_directive("voxel=info".parse().unwrap())
        .add_directive("gfx_backend_vulkan=off".parse().unwrap());

    if let Ok(text) = std::env::var("RUST_LOG") {
        filter = filter.add_directive(text.parse().unwrap());
    }

    tracing_subscriber::fmt::Subscriber::builder()
        .with_env_filter(filter)
        .pretty()
        .init()
}
