use crate::linear::Vec3;
use crate::{Coord, Ray};

#[derive(Debug)]
pub(crate) struct Octree<T> {
    root: OctreeSlot<T>,
    depth: u8,
}

type OctreeSlot<T> = Option<OctreeNode<T>>;

#[derive(Debug)]
enum OctreeNode<T> {
    Leaf(T),
    Branch(Box<[OctreeSlot<T>; 8]>),
}

#[derive(Debug, Copy, Clone)]
struct RayExt {
    origin: Vec3,
    direction: Vec3,
    inv_direction: Vec3,
}

impl From<Ray> for RayExt {
    fn from(ray: Ray) -> Self {
        RayExt {
            origin: ray.origin,
            direction: ray.direction,
            inv_direction: ray.direction.map(f32::recip),
        }
    }
}

impl<T: std::fmt::Debug> Octree<T> {
    pub fn with_depth(depth: u8) -> Self {
        Octree { root: None, depth }
    }

    pub fn insert(&mut self, coord: Coord, value: T) {
        let mut center = self.root_center();
        let mut current = &mut self.root;
        let mut size = 1 << self.depth;
        while size > 1 {
            let (dx, dy, dz) = Self::octant_offset(center, coord);
            let octant = Self::octant_from_offset(dx, dy, dz);
            current = Self::get_or_insert_octant(current, octant);

            size /= 2;

            center = Self::octant_center(octant, center, size);
        }

        // at max depth, insert the value
        *current = Some(OctreeNode::Leaf(value));
    }

    fn get_or_insert_octant(slot: &mut OctreeSlot<T>, octant: usize) -> &mut OctreeSlot<T> {
        match slot {
            Some(OctreeNode::Leaf(_old)) => panic!("insert into leaf node"),
            Some(OctreeNode::Branch(children)) => &mut children[octant],
            None => {
                *slot = Some(OctreeNode::empty_branch());
                match slot {
                    Some(OctreeNode::Branch(children)) => &mut children[octant],
                    _ => unreachable!(),
                }
            }
        }
    }

    pub fn cast_ray(&self, ray: Ray) -> Option<&T> {
        let center_coord = self.root_center();
        let center = Vec3 {
            x: center_coord.x as f32,
            y: center_coord.y as f32,
            z: center_coord.z as f32,
        };

        let ray = RayExt::from(ray);
        let size = (1 << self.depth) as f32;

        if let None = ray_cube_intersection(ray, center, size) {
            return None;
        }

        match self.root.as_ref()? {
            OctreeNode::Leaf(value) => return Some(value),
            OctreeNode::Branch(children) => {
                Self::cast_ray_children(children, ray, center, size / 2.0)
            }
        }
    }

    fn cast_ray_children(
        children: &[OctreeSlot<T>; 8],
        ray: RayExt,
        center: Vec3,
        child_size: f32,
    ) -> Option<&T> {
        let mut entries = [(0.0, 0); 8];
        let mut count = 0;
        for octant in 0..8 {
            let child_center = Self::child_octant_center(center, child_size, octant);
            if let Some((entry, _)) = ray_cube_intersection(ray, child_center, child_size) {
                entries[count] = (entry, octant);
                count += 1;
            }
        }

        let entries = &mut entries[..count];
        entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));

        for (_, octant) in entries {
            if let Some(value) =
                Self::cast_ray_children_recurse(children, ray, center, child_size, *octant)
            {
                return Some(value);
            }
        }

        None
    }

    #[inline(always)]
    fn cast_ray_children_recurse<'a>(
        parent_children: &'a [OctreeSlot<T>; 8],
        ray: RayExt,
        center: Vec3,
        child_size: f32,
        octant: usize,
    ) -> Option<&'a T> {
        match parent_children[octant].as_ref()? {
            OctreeNode::Leaf(value) => return Some(value),
            OctreeNode::Branch(children) => {
                let child_center = Self::child_octant_center(center, child_size, octant);
                Self::cast_ray_children(children, ray, child_center, child_size / 2.0)
            }
        }
    }

    fn child_octant_center(parent_center: Vec3, child_size: f32, octant: usize) -> Vec3 {
        let bool_sign = |b| if b { -0.5 } else { 0.5 };

        let add_or_sub = Vec3::new(
            bool_sign(octant & 4 == 0),
            bool_sign(octant & 2 == 0),
            bool_sign(octant & 1 == 0),
        );

        parent_center + child_size * add_or_sub
    }

    fn root_center(&self) -> Coord {
        let size = 1 << self.depth;
        Coord {
            x: size / 2,
            y: size / 2,
            z: size / 2,
        }
    }

    fn octant_offset(center: Coord, coord: Coord) -> (bool, bool, bool) {
        let dx = coord.x >= center.x;
        let dy = coord.y >= center.y;
        let dz = coord.z >= center.z;
        (dx, dy, dz)
    }

    fn octant_from_offset(dx: bool, dy: bool, dz: bool) -> usize {
        4 * dx as usize + 2 * dy as usize + dz as usize
    }

    fn octant_center(octant: usize, parent_center: Coord, size: u16) -> Coord {
        fn add_else_sub(cond: bool, value: &mut u16, amount: u16) {
            if cond {
                *value += amount;
            } else {
                *value -= amount
            }
        }

        let mut center = parent_center;
        add_else_sub(octant & 0x4 != 0, &mut center.x, size / 2);
        add_else_sub(octant & 0x2 != 0, &mut center.y, size / 2);
        add_else_sub(octant & 0x1 != 0, &mut center.z, size / 2);
        center
    }
}

impl<T> OctreeNode<T> {
    const SLOT_EMPTY: OctreeSlot<T> = None;

    pub fn empty_branch() -> Self {
        OctreeNode::Branch(Box::new([Self::SLOT_EMPTY; 8]))
    }
}

fn ray_cube_intersection(ray: RayExt, center: Vec3, size: f32) -> Option<(f32, f32)> {
    let half_size = size / 2.0;
    let delta = center - ray.origin;

    let entry_exit = |plane: usize| {
        use std::f32::INFINITY;
        let dir = ray.direction[plane];
        let dist = delta[plane];
        if dir == 0.0 {
            if dist.abs() <= half_size {
                (-INFINITY, INFINITY)
            } else {
                (INFINITY, -INFINITY)
            }
        } else {
            let inv_dir = ray.inv_direction[plane];
            let entry = (dist - half_size.copysign(inv_dir)) * inv_dir;
            let exit = (dist + half_size.copysign(inv_dir)) * inv_dir;
            (entry, exit)
        }
    };

    let (entry_x, exit_x) = entry_exit(0);
    let (entry_y, exit_y) = entry_exit(1);
    let (entry_z, exit_z) = entry_exit(2);

    let max_entry = entry_x.max(entry_y).max(entry_z);
    let min_exit = exit_x.min(exit_y).min(exit_z);

    if min_exit > 0.0 && max_entry < min_exit {
        Some((max_entry, min_exit))
    } else {
        None
    }
}
