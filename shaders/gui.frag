#version 450

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec2 frag_tex_coord;
layout(location = 1) in vec4 frag_color;

layout(binding = 1) uniform sampler near_sampler;
layout(binding = 2) uniform texture2D color_texture;

void main() {
    out_color = frag_color * texture(sampler2D(color_texture, near_sampler), frag_tex_coord * vec2(1, 1)).r;
}
