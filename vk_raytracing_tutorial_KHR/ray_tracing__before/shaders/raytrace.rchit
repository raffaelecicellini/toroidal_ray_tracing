#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : enable

#include "raycommon.glsl"
#include "wavefront.glsl"

hitAttributeEXT vec3 attribs;

layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;
layout(set = 0, binding = eTlas) uniform accelerationStructureEXT topLevelAS;	// chit shader needs to be aware of TLAS to be able to shoot rays
layout(buffer_reference, scalar) buffer Vertices {Vertex v[];};	// Position of an object
layout(buffer_reference, scalar) buffer Indices {ivec3 i[];};	// Triangle indices
layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[];};	// Array of all materials of an object
layout(buffer_reference, scalar) buffer MatIndices {int i[];};	// Material ID for each triangle
layout(set = 1, binding = eObjDescs, scalar) buffer ObjDesc_ {ObjDesc i[];} objDesc;
layout(set = 1, binding = eTextures) uniform sampler2D textureSamplers[];	// binding to array of texture samplers

layout(push_constant) uniform _PushConstantRay { PushConstantRay pcRay; };

void main()
{
  // Object data
  ObjDesc objResource = objDesc.i[gl_InstanceCustomIndexEXT];	// index tells which object was hit
  MatIndices matIndices = MatIndices(objResource.materialIndexAddress);
  Materials materials = Materials(objResource.materialAddress);
  Indices indices = Indices(objResource.indexAddress);
  Vertices vertices = Vertices(objResource.vertexAddress);

  // Indices of the triangle
  ivec3 ind = indices.i[gl_PrimitiveID];	// find vertices of the triangle hit

  // Vertex of the triangle
  Vertex v0 = vertices.v[ind.x];
  Vertex v1 = vertices.v[ind.y];
  Vertex v2 = vertices.v[ind.z];

  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // This is not very precise if the hit point is very far
  //vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

  // Computing coordinates of hit position by interpolation
  const vec3 pos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
  const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));	// Transforming position to world space

  // Computing the normal at hit position
  const vec3 nrm = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
  const vec3 worldNrm = normalize(vec3(nrm * gl_WorldToObjectEXT));	// transforming normal to world space

  // Vector toward the light (simple diffuse lighting effect)
  vec3 L;
  float lightIntensity = pcRay.lightIntensity;
  float lightDistance = 100000.0;
  // Point light
  if(pcRay.lightType == 0)
  {
	vec3 lDir = pcRay.lightPosition - worldPos;
	lightDistance = length(lDir);
	lightIntensity = pcRay.lightIntensity / (lightDistance * lightDistance);
	L = normalize(lDir);
  }
  else
  {
	L = normalize(pcRay.lightPosition);
  }

  // Material of the object
  int matIdx = matIndices.i[gl_PrimitiveID];
  WaveFrontMaterial mat = materials.m[matIdx];

  // Diffuse
  vec3 diffuse = computeDiffuse(mat, L, worldNrm);
  if(mat.textureId >= 0)
  {
	uint txtId = mat.textureId + objDesc.i[gl_InstanceCustomIndexEXT].txtOffset;
	vec2 texCoord = v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
	diffuse *= texture(textureSamplers[nonuniformEXT(txtId)], texCoord).xyz;
  }

  vec3 specular = vec3(0);
  float attenuation = 1;

  // Tracing shadow ray only if the light is visible from the surface
  if(dot(worldNrm, L) > 0)
  {
	float tMin = 0.001;
	float tMax = lightDistance;
	vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	vec3 rayDir = L;
	uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
	isShadowed = true;
	traceRayEXT(topLevelAS,
				flags,
				0xFF,
				0,
				0,
				1,
				origin,
				tMin,
				rayDir,
				tMax,
				1
	);

	if(isShadowed)
	{
	  attenuation = 0.3;
	}
	else
	{
	  // Specular
	  specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, worldNrm);
	}
  }

  // If the material is reflective, we need to shoot a ray
  if(mat.illum == 3)
  {
	vec3 origin = worldPos;
	vec3 rayDir = reflect(gl_WorldRayDirectionEXT, worldNrm);
	prd.attenuation *= mat.specular;
	prd.done = 0;
	prd.rayOrigin = origin;
	prd.rayDir = rayDir;
  }

  prd.hitValue = vec3(attenuation * lightIntensity * (diffuse + specular));
  prd.hitPosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;	// save world pos of hit
}
