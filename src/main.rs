#[macro_use]
mod macros;

mod cpu;
mod linear;

use crate::linear::Vec3;

trait Backend {
    fn from_voxels(voxels: Vec<(Coord, Color)>) -> Self;

    fn render(&self, camera: Camera, size: Size, pixels: &mut [Color]);
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Size {
    width: u32,
    height: u32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Coord {
    x: u16,
    y: u16,
    z: u16,
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
}

#[derive(Debug, Copy, Clone)]
struct Camera {
    position: Vec3,
    direction: Vec3,
    fov: f32,
}

#[derive(Debug, Copy, Clone)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

fn main() {
    let size = Size {
        width: 512,
        height: 512,
    };

    let opts = minifb::WindowOptions::default();
    let mut window =
        minifb::Window::new("voxel", size.width as usize, size.height as usize, opts).unwrap();

    let mut voxels = Vec::new();

    let s: u16 = 15;
    let c = s / 2;

    let r = 5;
    let h = 9;

    for x in 0..s {
        for z in 0..s {
            voxels.push(([x, 0, z].into(), [50, 200, 0].into()));
        }
    }

    for y in 1..h + r {
        voxels.push(([c, y, c].into(), [100, 50, 0].into()));
    }

    for x in c - r..=c + r {
        for z in c - r..=c + r {
            for y in h - r..=h + r {
                let cx: i32 = x as i32 - c as i32;
                let cy: i32 = y as i32 - h as i32;
                let cz: i32 = z as i32 - c as i32;
                if cx.pow(2) + cy.pow(2) + cz.pow(2) < r.pow(2) as i32 {
                    voxels.push(([x as u16, y as u16, z as u16].into(), [0, 100, 0].into()));
                }
            }
        }
    }

    let backend = cpu::CpuBackend::from_voxels(voxels);

    let start = std::time::Instant::now();

    let mut pixels = vec![Color::BLACK; (size.width * size.height) as usize];
    while window.is_open() && !window.is_key_pressed(minifb::Key::Escape, minifb::KeyRepeat::No) {
        let time = 0.2 * start.elapsed().as_secs_f32();
        let dir = Vec3::new(time.cos(), -0.5, time.sin()).norm();

        let camera = Camera {
            position: Vec3::new(c as f32 + 0.5, 3.0, c as f32 + 0.5) - 20.0 * dir,
            direction: dir,
            fov: 70.0f32.to_radians(),
        };

        backend.render(camera, size, &mut pixels);

        use rayon::prelude::*;
        let buffer = pixels
            .par_iter()
            .map(|color| (color.r as u32) << 16 | (color.g as u32) << 8 | color.b as u32)
            .collect::<Vec<_>>();

        window
            .update_with_buffer(&buffer, size.width as usize, size.height as usize)
            .unwrap();

        if window.is_key_pressed(minifb::Key::Space, minifb::KeyRepeat::No) {
            if let Some((x, y)) = window.get_mouse_pos(minifb::MouseMode::Discard) {
                eprintln!(
                    "mouse_ray : {:#?}",
                    camera.cast_ray_single(x as u32, y as u32, size)
                );
            }
        }
    }
}

impl From<[u16; 3]> for Coord {
    fn from([x, y, z]: [u16; 3]) -> Self {
        Coord { x, y, z }
    }
}

impl From<[u8; 3]> for Color {
    fn from([r, g, b]: [u8; 3]) -> Self {
        Color { r, g, b }
    }
}

impl Color {
    pub const BLACK: Color = Color::gray(0);

    pub const fn gray(gray: u8) -> Color {
        Color::new(gray, gray, gray)
    }

    pub const fn new(r: u8, g: u8, b: u8) -> Color {
        Color { r, g, b }
    }
}

impl std::fmt::Debug for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Color { r, g, b } = *self;
        write!(
            f,
            "\x1b[38;2;{};{};{}m██\x1b[0m Color({}, {}, {})",
            r, g, b, r, g, b
        )
    }
}

impl Camera {
    pub fn cast_rays(
        &self,
        size: Size,
    ) -> impl rayon::iter::IndexedParallelIterator<Item = (Ray, (u32, u32))> {
        use rayon::prelude::*;

        let fov_scale = (self.fov / 2.0).tan();

        let forward = self.direction.norm();
        let right = Vec3::new(0.0, 1.0, 0.0).cross(forward).norm();
        let up = forward.cross(right);

        let w = size.width as f32;
        let h = size.height as f32;
        let forward_ray = (-w / 2.0) * right + (h / 2.0) * up + (h / 2.0) / fov_scale * forward;

        let origin = self.position;

        (0..size.width * size.height).into_par_iter().map(move |index| {
            let row = index / size.width;
            let col = index % size.width;

            let screen_dir = col as f32 * right - row as f32 * up + forward_ray;
            let coord = (col, row);
            let ray = Ray {
                origin,
                direction: screen_dir.norm(),
            };
            (ray, coord)
        })
    }

    fn cast_ray_single(&self, x: u32, y: u32, size: Size) -> Ray {
        let fov_scale = (self.fov / 2.0).tan();

        let forward = self.direction.norm();
        let right = Vec3::new(0.0, 1.0, 0.0).cross(forward).norm();
        let up = forward.cross(right);

        let w = size.width as f32;
        let h = size.height as f32;
        let forward_ray = (-w / 2.0) * right + (h / 2.0) * up + (h / 2.0) / fov_scale * forward;

        let screen_dir = x as f32 * right - y as f32 * up + forward_ray;
        Ray {
            origin: self.position,
            direction: screen_dir.norm(),
        }
    }
}
