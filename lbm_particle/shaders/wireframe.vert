#version 450
#extension GL_EXT_scalar_block_layout : require

#define Q 9

layout (binding = 0) uniform RenderingUBO {
    uint Nx;
    uint Ny;
    uint Nz;
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec3 fragColor;

vec3 particle_norm(vec3 pos, float factor) {
    return vec3((pos.x - ubo.Nx / 2.0f) / factor, (pos.y - ubo.Ny / 2.0f) / factor, (pos.z - ubo.Nz / 2.0f) / factor);
}

void main() {
    gl_PointSize = 2.0;
    float factor = max(ubo.Nx, max(ubo.Ny, ubo.Nz));
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(particle_norm(inPosition.xyz, factor), 1.0);
    fragColor = inColor.rgb;
}