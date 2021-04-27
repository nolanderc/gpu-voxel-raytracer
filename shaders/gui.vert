#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 tex_coord;
layout(location = 2) in uint color;

layout(location = 0) out vec2 frag_tex_coord;
layout(location = 1) out vec4 frag_color;

layout(binding = 0) uniform Screen {
    float width, height;
};

void main() {
    gl_Position = vec4(2 * position.x / width - 1, 1 - 2 * position.y / height, 0, 1);

    vec4 color = unpackUnorm4x8(color);

    frag_tex_coord = tex_coord;
    frag_color = color;
}

