#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "raycommon.glsl"
#include "wavefront.glsl"

layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(push_constant) uniform _PushConstantRay
{
    PushConstantRay pcRay;
};

void main()
{
    //prd.hitValue = pcRay.clearColor.xyz * 0.8 * prd.attenuation;
    prd.hitValue = pcRay.clearColor.xyz * 0.8;
    //prd.hitPosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * 10000.0; // if miss we set position as far plane
    prd.hitPosition = vec3(0);
}