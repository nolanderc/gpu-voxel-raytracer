#version 450

#define MAX_DEPTH 10

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
    out float entry, out float exit
) {
    vec3 signum = sign(inv_dir);

    vec3 entry_planes = center - half_size * signum;
    vec3 exit_planes = center + half_size * signum;

    vec3 entries = (entry_planes - origin) * inv_dir;
    vec3 exits = (exit_planes - origin) * inv_dir;

    entry = max(max(entries.x, entries.y), entries.z);
    exit = min(min(exits.x, exits.y), exits.z);

    return exit >= 0 && entry < exit;
}

vec3 octant_center(vec3 center, float size, uint octant) {
    vec3 delta = vec3((octant >> 2) & 1, (octant >> 1) & 1, octant & 1);
    return center + 0.25 * size * sign(delta - 0.5);
}

struct OctantIntersections {
    uint octants_and_count;
    float entries[5];
};

void insert_octant(inout uint octants, uint octant, int octant_index) {
    octants = bitfieldInsert(octants, octant, 3 * (octant_index + 1), 3);
}

uint get_octant(uint octants, int octant_index) {
    return bitfieldExtract(octants, 3 * (octant_index + 1), 3);
}

// (1 <= count <= 4, so we store 0 <= count - 1 <= 3 using 2 bits)
void insert_count(inout uint octants, uint count) {
    octants |= count - 1;
}

uint get_count(uint octants) {
    return (octants & 0x3) + 1;
}

void octant_intersections(
    vec3 origin, vec3 inv_dir, 
    vec3 center, float size, 
    float entry, float exit,
    out OctantIntersections intersections
) {
    vec3 delta = center - origin;
    vec3 plane_entries = delta * inv_dir;
    int order[3] = int[3](0, 1, 2);

    if (plane_entries.y < plane_entries.x) {
        order[0] = 1;
        order[1] = 0;
    }
    if (plane_entries.z < plane_entries[order[1]]) {
        order[2] = order[1];
        if (plane_entries.z < plane_entries[order[0]]) {
            order[1] = order[0];
            order[0] = 2;
        } else {
            order[1] = 2;
        }
    }

    vec3 entries = vec3(plane_entries[order[0]], plane_entries[order[1]], plane_entries[order[2]]);

    uint octant = (delta.x < 0 ? 4 : 0) | (delta.y < 0 ? 2 : 0) | (delta.z < 0 ? 1 : 0);
    float prev_time = entry;

    int count = 0;
    intersections.octants_and_count = 0;
    for (int i = 0; i < 3; i++) {
        // we only want intersections within the current node
        if (entries[i] < 0 || entries[i] >= exit) continue;

        if (entries[i] >= entry) {
            // store the current octant, with it's entry time
            insert_octant(intersections.octants_and_count, octant, count);
            intersections.entries[count] = prev_time;
            count++;

            // store the entry time for the next octant
            prev_time = entries[i];
        }

        // move to the next octant
        octant ^= 4 >> order[i];
    }

    // Store the current octant (we know there's always one)
    insert_octant(intersections.octants_and_count, octant, count);
    intersections.entries[count] = prev_time;
    count++;

    // store the count
    insert_count(intersections.octants_and_count, count);

    // Finally, store the time we enter the neighbouring node
    intersections.entries[count] = exit;
}

struct Frame {
    int node;
    int stage;
    vec3 center;
    float size;
    OctantIntersections intersections;
};

bool cast_ray(vec3 ray_origin, vec3 ray_dir, out float time, out vec3 color, out vec3 normal) {
    float root_entry, root_exit;
    int stack_size = 0;

    Frame stack[MAX_DEPTH];

    vec3 ray_inv_dir = 1.0 / ray_dir;

    bool intersect = ray_cube_intersection(
            ray_origin, ray_inv_dir, 
            root_center, 0.5 * root_size,
            root_entry, root_exit
        );

    if (intersect) {
        octant_intersections(
                ray_origin, ray_inv_dir,
                root_center, root_size,
                root_entry, root_exit,
                stack[0].intersections
            );

        stack[0].node = 0;
        stack[0].stage = 0;
        stack[0].center = root_center;
        stack[0].size = root_size;
        stack_size = 1;
    } else {
        return false;
    }

    int value = 0;
    int normal_plane;
    while (stack_size > 0) {
        // Advance to next octant of current node.
        int c = stack_size - 1;
        int i = stack[c].stage++;

        // This node is done.
        if (i >= get_count(stack[c].intersections.octants_and_count)) {
            // Pop from the stack.
            stack_size--;
            continue;
        }

        int node = stack[c].node;
        uint octant = get_octant(stack[c].intersections.octants_and_count, i);

        // Get the octant's value
        value = nodes[node + octant];

        // Is it a color value?
        if (value < 0) {
            // found a voxel
            time = stack[c].intersections.entries[i];
            vec3 point = ray_origin + ray_dir * time;
            vec3 child_center = octant_center(stack[c].center, stack[c].size, octant);
            vec3 delta = point - child_center;
            vec3 distances = abs(delta);
            float max_dist = max(max(distances.x, distances.y), distances.z);
            normal_plane = distances.x == max_dist ? 0 : (distances.y == max_dist ? 1 : 2);
            break;
        };

        // Is it empty?
        if (value == 0) continue;

        int child = value;
        vec3 child_center = octant_center(stack[c].center, stack[c].size, octant);
        float child_size = 0.5 * stack[c].size;

        // Find intersections with the octant, and push onto stack
        octant_intersections(
                ray_origin, ray_inv_dir,
                child_center, child_size,
                stack[c].intersections.entries[i], stack[c].intersections.entries[i+1],
                stack[stack_size].intersections
            );
        stack[stack_size].node = child;
        stack[stack_size].stage = 0;
        stack[stack_size].center = child_center;
        stack[stack_size].size = child_size;
        stack_size++;
    }

    if (stack_size == 0) {
        return false;
    }

    int r = (value >> 16) & 0xff;
    int g = (value >> 8) & 0xff;
    int b = value & 0xff;
    color = vec3(r, g, b) / 255.0;

    normal = vec3(0);
    normal[normal_plane] = -sign(ray_dir[normal_plane]);

    return true;
}

void main() {
    vec3 ray_origin = camera_origin.xyz;
    vec3 ray_dir = normalize(
        frag_coord.x * camera_right.xyz 
        + frag_coord.y * camera_up.xyz 
        + camera_forward.xyz
    );

    float time;
    vec3 color, normal;
    bool hit = cast_ray(ray_origin, ray_dir, time, color, normal);

    if (hit) {
        const vec3 LIGHT_DIR = normalize(vec3(1, -3, 2));

        float shadow_time;
        vec3 shadow_color, shadow_normal;
        vec3 hit_point = ray_origin + ray_dir * (0.9999 * time);
        bool shadow = cast_ray(hit_point, -LIGHT_DIR, shadow_time, shadow_color, shadow_normal);

        float diffuse = 0.8 * max(0.0, dot(-LIGHT_DIR, normal));
        float brightness = 0.2 + (shadow ? 0.3 * diffuse : diffuse);
        out_color = vec4(color * brightness, 1);
    } else {
        out_color = vec4(abs(ray_dir), 1);
    }
}
