// Declare entries for bind groups
macro_rules! bind_group {
    ( $( $kind:ident $args:tt ),* $(,)? ) => {
        (
            [ $( bind_group!(@layout $kind $args) ),* ],
            [ $( bind_group!(@resource $kind $args) ),* ],
        )
    };

    // Uniforms
    (@layout Uniform($binding:expr => ($buffer:expr) in $visibility:expr)) => {
        wgpu::BindGroupLayoutEntry {
            binding: $binding,
            visibility: bind_group!(@visibility $visibility),
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: $buffer.size(),
            },
            count: None,
        }
    };
    (@resource Uniform($binding:expr => ($buffer:expr) in $visibility:expr)) => {
        wgpu::BindGroupEntry {
            binding: $binding,
            resource: wgpu::BindingResource::Buffer {
                buffer: &$buffer,
                offset: 0,
                size: None,
            },
        }
    };

    // Uniforms
    (@layout UniformImage($binding:expr => ($texture:expr, $access:expr, $format:expr, $dimension:expr) in $visibility:expr)) => {
        #[allow(unused_imports)]
        wgpu::BindGroupLayoutEntry {
            binding: $binding,
            visibility: bind_group!(@visibility $visibility),
            ty: wgpu::BindingType::StorageTexture {
                access: {
                    use wgpu::StorageTextureAccess::*;
                    $access
                },
                format: {
                    use wgpu::TextureFormat::*;
                    $format
                },
                view_dimension: {
                    use wgpu::TextureViewDimension::*;
                    $dimension
                }
            },
            count: None,
        }
    };
    (@resource UniformImage($binding:expr => ($texture:expr, $access:expr, $format:expr, $dimension:expr) in $visibility:expr)) => {
        wgpu::BindGroupEntry {
            binding: $binding,
            resource: wgpu::BindingResource::TextureView($texture),
        }
    };

    // Storage
    (@layout Storage($binding:expr => ($buffer:expr, read_only: $read:expr) in $visibility:expr)) => {
        wgpu::BindGroupLayoutEntry {
            binding: $binding,
            visibility: bind_group!(@visibility $visibility),
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: $read },
                has_dynamic_offset: false,
                min_binding_size: $buffer.size(),
            },
            count: None,
        }
    };
    (@resource Storage($binding:expr => ($buffer:expr, read_only: $read:expr) in $visibility:expr)) => {
        wgpu::BindGroupEntry {
            binding: $binding,
            resource: wgpu::BindingResource::Buffer {
                buffer: &$buffer,
                offset: 0,
                size: None,
            },
        }
    };

    (@visibility $visibility:expr) => {
        {
            #[allow(dead_code)]
            const COMPUTE: wgpu::ShaderStage = wgpu::ShaderStage::COMPUTE;
            #[allow(dead_code)]
            const VERTEX: wgpu::ShaderStage = wgpu::ShaderStage::VERTEX;
            #[allow(dead_code)]
            const FRAGMENT: wgpu::ShaderStage = wgpu::ShaderStage::FRAGMENT;
            $visibility
        }
    };
}
