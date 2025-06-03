#version 450

layout (binding = 1) uniform samplerCube samplerCubeMap;

layout (location = 0) in vec3 inUVW;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	//outFragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
	//outFragColor = vec4(inUVW, 1.0f);
	outFragColor = texture(samplerCubeMap, inUVW);
}