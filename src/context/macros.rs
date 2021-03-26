
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
            visibility: $visibility,
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

    // Storage
    (@layout Storage($binding:expr => ($buffer:expr, read_only: $read:expr) in $visibility:expr)) => {
        wgpu::BindGroupLayoutEntry {
            binding: $binding,
            visibility: $visibility,
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
}

