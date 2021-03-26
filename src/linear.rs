use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

macro_rules! impl_elementwise_op {
    ($trait:ident, $op:ident, $target:ident { $($field:ident),* }) => {
        impl $trait<$target> for $target {
            type Output = Self;

            fn $op(self, other: Self) -> Self {
                Self {
                    $(
                        $field: $trait::$op(self.$field, other.$field),
                    )*
                }
            }
        }
    }
}

macro_rules! impl_scalar_op {
    ($trait:ident, $op:ident, $target:ident { $($field:ident),* }, $scalar:ident) => {
        impl $trait<$scalar> for $target {
            type Output = Self;

            fn $op(self, other: $scalar) -> Self {
                Self {
                    $(
                        $field: $trait::$op(self.$field, other),
                    )*
                }
            }
        }
    };

    ($trait:ident, $op:ident, $scalar:ident, $target:ident { $($field:ident),* }) => {
        impl $trait<$target> for $scalar {
            type Output = $target;

            fn $op(self, other: $target) -> $target {
                $target {
                    $(
                        $field: $trait::$op(self, other.$field),
                    )*
                }
            }
        }
    };
}

macro_rules! impl_conversion {
    ($vector:ident { $($field:ident),* }) => {
        impl $vector {
            const DIMENSION: usize = count!($($field),*);
        }

        impl From<[f32; $vector::DIMENSION]> for $vector {
            fn from([$($field),*]: [f32; $vector::DIMENSION]) -> $vector {
                $vector { $($field),* }
            }
        }

        impl From<$vector> for [f32; $vector::DIMENSION] {
            fn from(vector: $vector) -> [f32; $vector::DIMENSION] {
                [ $( vector.$field ),* ]
            }
        }
    }
}

macro_rules! impl_vector {
    ($vector:ident { $($field:ident),* }) => {
        impl $vector {
            #[inline(always)]
            pub const fn new($($field:f32),*) -> $vector {
                $vector {
                    $( $field ),*
                }
            }

            #[inline(always)]
            pub const fn zero() -> $vector {
                $vector {
                    $( $field: 0.0 ),*
                }
            }

            #[inline(always)]
            pub const fn one() -> $vector {
                $vector {
                    $( $field: 1.0 ),*
                }
            }

            #[inline(always)]
            pub fn dot(self, other: $vector) -> f32 {
                0.0 $( + self.$field * other.$field)*
            }

            #[inline(always)]
            pub fn length_squared(self) -> f32 {
                self.dot(self)
            }

            #[inline(always)]
            pub fn length(self) -> f32 {
                self.length_squared().sqrt()
            }

            #[inline(always)]
            pub fn norm(self) -> $vector {
                self / self.length()
            }

            #[inline(always)]
            pub fn map(self, mut f: impl FnMut(f32) -> f32) -> $vector {
                $vector {
                    $( $field: f(self.$field) ),*
                }
            }

            #[inline(always)]
            pub fn elementwise(self, other: $vector, mut f: impl FnMut(f32, f32) -> f32) -> $vector {
                $vector {
                    $( $field: f(self.$field, other.$field) ),*
                }
            }

            #[inline(always)]
            pub fn elementwise_product(self, other: $vector) -> $vector {
                self.elementwise(other, |a, b| a * b)
            }

            #[inline(always)]
            pub fn abs(self) -> $vector {
                $vector {
                    $( $field: self.$field.abs() ),*
                }
            }

            #[inline(always)]
            pub fn any(self, mut f: impl FnMut(f32) -> bool) -> bool {
                false $( || f(self.$field) )*
            }

            #[inline(always)]
            pub fn min(self) -> f32 {
                sequence_binary!(f32::min, [$(self.$field),*])
            }

            #[inline(always)]
            pub fn max(self) -> f32 {
                sequence_binary!(f32::max, [$(self.$field),*])
            }

            #[inline(always)]
            pub fn reflect(self, normal: $vector) -> $vector {
                self - 2.0 * self.dot(normal) * normal
            }
        }


        impl Neg for $vector {
            type Output = $vector;

            fn neg(self) -> Self::Output {
                $vector {
                    $( $field: -self.$field ),*
                }
            }
        }

        impl_conversion!($vector { $($field),* });

        impl_elementwise_op!(Add, add, $vector { $( $field ),* });
        impl_elementwise_op!(Sub, sub, $vector { $( $field ),* });

        impl_scalar_op!(Mul, mul, f32, $vector { $( $field ),* });
        impl_scalar_op!(Mul, mul, $vector { $( $field ),* }, f32);
        impl_scalar_op!(Div, div, $vector { $( $field ),* }, f32);

        impl AddAssign<$vector> for $vector {
            fn add_assign(&mut self, rhs: $vector) {
                $( self.$field += rhs.$field; )*
            }
        }

        impl SubAssign<$vector> for $vector {
            fn sub_assign(&mut self, rhs: $vector) {
                $( self.$field -= rhs.$field; )*
            }
        }
    }
}

impl_vector!(Vec3 { x, y, z });

impl Vec3 {
    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3: index {} is out of bounds", index),
        }
    }
}

impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3: index {} is out of bounds", index),
        }
    }
}
