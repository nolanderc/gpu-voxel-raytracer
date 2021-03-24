mod octree;

use self::octree::Octree;
use crate::{Backend, Camera, Color, Coord, Size};

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

    fn render(&self, camera: Camera, size: Size, pixels: &mut [Color]) {
        use rayon::prelude::*;
        camera.cast_rays(size).zip_eq(pixels.par_iter_mut())
            .for_each(|((ray, _), pixel)| {
                *pixel = self
                    .octree
                    .cast_ray(ray)
                    .copied()
                    .unwrap_or(Color { r: 0, g: 0, b: 0 });
            });
    }
}
