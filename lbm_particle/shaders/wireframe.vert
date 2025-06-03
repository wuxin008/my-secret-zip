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

vec3 particle_norm(vec3 pos) {
    return vec3(pos.x / ubo.Nx - 0.5f, pos.y / ubo.Ny - 0.5f, pos.z / ubo.Nz - 0.5f);
}

void main() {
    gl_PointSize = 2.0;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(particle_norm(inPosition.xyz), 1.0);
    fragColor = inColor.rgb;
}