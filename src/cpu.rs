mod octree;

use self::octree::Octree;
use crate::linear::Vec3;
use crate::{Backend, Camera, Color, Coord, Ray, Size};

#[derive(Debug)]
pub struct CpuBackend {
    octree: Octree<Color>,
}

impl Backend for CpuBackend {
    fn from_voxels(voxels: Vec<(Coord, Color)>) -> Self {
        let mut max_coord = 0;
        for (coord, _) in voxels.iter() {
            max_coord = max_coord.max(coord.x).max(coord.y).max(coord.z);
        }

        let max_depth = if max_coord == 0 {
            0
        } else {
            (max_coord as u32 + 1).next_power_of_two().trailing_zeros()
        };

        let mut octree = Octree::with_depth(max_depth as u8);
        for (coord, color) in voxels {
            octree.insert(coord, color);
        }
        CpuBackend { octree }
    }

    fn render(&self, camera: Camera, size: Size, pixels: &mut [Color], time: f32) {
        use rayon::prelude::*;

        let c = 127.0 / 2.0;

        let light_pos = Vec3::new(
            c - 10.0 * (0.3 * time).cos(),
            15.0 + 8.0 * (3.0 * time).sin(),
            c - 13.0 * (0.3 * time).sin(),
        );

        camera
            .cast_rays(size)
            .zip_eq(pixels.par_iter_mut())
            .for_each(|((ray, _), pixel)| {
                *pixel = if let Some(hit) = self.octree.cast_ray(ray) {
                    let hit_point = ray.origin + ray.direction * hit.time;
                    let light_delta = light_pos - hit_point;
                    let light_distance = light_delta.length();
                    let light_dir = light_delta / light_distance;

                    let shadow_ray = Ray {
                        origin: hit_point + 0.001 * hit.normal,
                        direction: light_dir,
                    };

                    let in_shadow =
                        matches!(self.octree.cast_ray(shadow_ray),Some(hit)if hit.time < light_distance);
                    let attenuation = 60.0 * light_distance.powf(-2.0);
                    let shadow = 0.4 + 0.6 * !in_shadow as u8 as f32;
                    let brightness = 0.0 + shadow * light_dir.dot(hit.normal).max(0.0) * attenuation;

                    let r = hit.value.r * brightness;
                    let g = hit.value.g * brightness;
                    let b = hit.value.b * brightness;
                    Color { r, g, b }
                } else {
                    Color::BLACK
                };
            });
    }
}
