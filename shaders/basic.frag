#version 450

#define MAX_DEPTH 15

layout(location = 0) in vec2 frag_coord;

layout(binding = 0) uniform uniforms {
    vec4 camera_origin;
    vec4 camera_right;
    vec4 camera_up;
    vec4 camera_forward;
    float time;
};

layout(binding = 1) readonly buffer octree_data {
    vec3 root_center;
    float root_size;
    int nodes[];
};

layout (location = 0) out vec4 out_color;

bool ray_cube_intersection(
    vec3 origin, vec3 inv_dir,
    vec3 center, float half_size,
    out float entry, out float exit, out vec3 normal
) {
    vec3 signum = sign(inv_dir);

    vec3 entry_planes = center - half_size * signum;
    vec3 exit_planes = center + half_size * signum;

    vec3 entries = (entry_planes - origin) * inv_dir;
    vec3 exits = (exit_planes - origin) * inv_dir;

    entry = max(max(entries.x, entries.y), entries.z);
    exit = min(min(exits.x, exits.y), exits.z);

    normal = -vec3(equal(vec3(entry), entries)) * signum;

    return exit >= 0 && entry < exit;
}

vec3 octant_center(vec3 center, float size, int octant) {
    vec3 delta = vec3((octant >> 2) & 1, (octant >> 1) & 1, octant & 1);
    return center + 0.25 * size * sign(delta - 0.5);
}

struct OctantIntersections {
    int count;
    int octants[4];
    vec3 normals[4];
    float entries[5];
};

void octant_intersections(
    vec3 origin, vec3 inv_dir, 
    vec3 center, float size, 
    float entry, float exit, vec3 entry_normal,
    out OctantIntersections intersections
) {
    vec3 delta = center - origin;
    vec3 plane_entry = delta * inv_dir;
    int order[3] = int[3](0, 1, 2);

    if (plane_entry.y < plane_entry.x) {
        order[0] = 1;
        order[1] = 0;
    }
    if (plane_entry.z < plane_entry[order[1]]) {
        order[2] = order[1];
        if (plane_entry.z < plane_entry[order[0]]) {
            order[1] = order[0];
            order[0] = 2;
        } else {
            order[1] = 2;
        }
    }

    int i = 0;
    while (i < 3 && plane_entry[order[i]] < 0.0) { i++; }

    int octant = (delta.x < 0 ? 4 : 0) | (delta.y < 0 ? 2 : 0) | (delta.z < 0 ? 1 : 0);

    while (i < 3 && plane_entry[order[i]] < entry) { 
        octant ^= 4 >> order[i];
        i++;
    }

    intersections.octants[0] = octant;
    intersections.normals[0] = entry_normal;
    intersections.entries[0] = entry;
    intersections.count = 1;

    while (i < 3 && plane_entry[order[i]] < exit) {
        octant ^= 4 >> order[i];

        vec3 normal = vec3(0);
        normal[order[i]] = -sign(inv_dir[order[i]]);

        intersections.octants[intersections.count] = octant;
        intersections.normals[intersections.count] = normal;
        intersections.entries[intersections.count] = plane_entry[order[i]];
        intersections.count++;

        i++;
    }

    intersections.entries[intersections.count] = exit;
}

struct Frame {
    int node;
    int stage;
    vec3 center;
    float size;
    OctantIntersections intersections;
};

void main() {
    vec3 ray_origin = camera_origin.xyz;
    vec3 ray_dir = normalize(
        frag_coord.x * camera_right.xyz 
        + frag_coord.y * camera_up.xyz 
        + camera_forward.xyz
    );

    vec3 ray_inv_dir = 1.0 / ray_dir;

    float entry, exit;
    vec3 normal;
    bool intersect = ray_cube_intersection(ray_origin, ray_inv_dir, root_center, 0.5 * root_size, entry, exit, normal);

    Frame stack[MAX_DEPTH];
    int stack_size = 0;

    if (intersect) {
        OctantIntersections root_intersections;
        octant_intersections(
                ray_origin, ray_inv_dir,
                root_center, root_size,
                entry, exit, normal,
                root_intersections
            );

        stack[0] = Frame(0, 0, root_center, root_size, root_intersections);
        stack_size = 1;
    }


    int value = 0;
    while (stack_size > 0) {
        int c = stack_size - 1;
        int i = stack[c].stage++;
        if (i >= stack[c].intersections.count) {
            stack_size--;
            continue;
        }

        int node = stack[c].node;
        int octant = stack[c].intersections.octants[i];

        value = nodes[node + octant];
        if (value < 0) {
            normal = stack[c].intersections.normals[i];
            break;
        };
        if (value == 0) continue;

        int child = value;
        vec3 child_center = octant_center(stack[c].center, stack[c].size, octant);
        float child_size = 0.5 * stack[c].size;

        OctantIntersections child_intersections;
        octant_intersections(
                ray_origin, ray_inv_dir,
                child_center, child_size,
                stack[c].intersections.entries[i], stack[c].intersections.entries[i+1],
                stack[c].intersections.normals[i], 
                child_intersections
            );

        stack[stack_size] = Frame(child, 0, child_center, child_size, child_intersections);
        stack_size++;
    }

    if (stack_size == 0) {
        out_color = vec4(abs(ray_dir), 1);
    } else {
        const vec3 LIGHT_DIR = normalize(vec3(1, -3, 2));

        int r = (value >> 16) & 0xff;
        int g = (value >> 8) & 0xff;
        int b = value & 0xff;
        vec3 rgb = vec3(r, g, b) / 255.0;

        float brightness = 0.2 + 0.8 * max(0.0, dot(-LIGHT_DIR, normal));

        out_color = vec4(rgb * brightness, 1);
    }
}
