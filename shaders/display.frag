#version 450

layout(location = 0) in vec2 frag_coord;

layout(rgba32f, binding = 0) uniform readonly image2D image;

layout(location = 0) out vec4 out_color;

float luminance(vec3 color) {
    return length(color);
}

#define SIZE 3
#define RADIUS (SIZE / 2)

void main() {
    ivec2 size = imageSize(image);
    ivec2 pixel = ivec2(size * (0.5 + vec2(0.5, -0.5) * frag_coord));
    out_color = imageLoad(image, pixel);
}

