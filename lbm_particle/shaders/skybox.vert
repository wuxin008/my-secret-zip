#version 450

layout (location = 0) in vec3 inPos;

layout (binding = 0) uniform RenderingUBO {
    uint Nx;
    uint Ny;
    uint Nz;
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout (location = 0) out vec3 outUVW;

void main() 
{
	outUVW = inPos / 100;
	// Convert cubemap coordinates into Vulkan coordinate space
	outUVW.xyz *= -1.0;
	// Remove translation from view matrix
	mat4 viewMat = mat4(mat3(ubo.view * ubo.model));
	gl_Position = ubo.proj * viewMat * vec4(inPos.xyz / 100, 1.0);
}
