// Since it is common to different shaders, we define the payload in a shared file
struct hitPayload
{
  vec3 hitValue;
  int depth;	// ray depth
  vec3 attenuation;	// ray attenuation
  vec3 hitPosition;	// info about world pos of hit point
  // Additional info to shoot reflection rays if needed
  int done;
  vec3 rayOrigin;
  vec3 rayDir;
};
