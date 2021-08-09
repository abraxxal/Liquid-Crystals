#version 330 core

// Geometry data
layout (location = 0) in vec3 cube_vertex;

// Particle data
layout (location = 1) in vec3 position;
layout (location = 2) in vec3 direction;

// World space transforms
uniform mat4 view;
uniform mat4 proj;

// Orientation and zoom parameters
uniform vec3 front;
uniform float zoom;

out vec4 color;

mat4 lookAt(vec3 eye, vec3 at, vec3 up)
{
  vec3 zaxis = normalize(at - eye);    
  vec3 xaxis = normalize(cross(zaxis, up));
  vec3 yaxis = cross(xaxis, zaxis);

  zaxis *= -1;

  return transpose(mat4(
    vec4(xaxis, -dot(xaxis, eye)),
    vec4(yaxis, -dot(yaxis, eye)),
    vec4(zaxis, -dot(zaxis, eye)),
    vec4(0, 0, 0, 1)
  ));
}

mat4 scale(vec3 diag)
{
  return mat4(
    vec4(diag.x, 0, 0, 0),
    vec4(0, diag.y, 0, 0),
    vec4(0, 0, diag.z, 0),
    vec4(0, 0, 0, 1)
  );
}

mat4 translate(vec3 offset)
{
  return mat4(
    vec4(1, 0, 0, 0),
    vec4(0, 1, 0, 0),
    vec4(0, 0, 1, 0),
    vec4(offset, 1)
  );
}

void main()
{
  mat4 model = mat4(1);
  model *= lookAt(vec3(0), front, vec3(0, 1, 0));
  model *= scale(vec3(zoom));
  model *= translate(position);
  model *= lookAt(vec3(0), direction, vec3(0, 1, 0));
  model *= scale(vec3(0.01, 0.01, 0.05));

  gl_Position = proj * view * model * vec4(cube_vertex, 1);

  float z = direction.z;
  float pi = 3.141592;
  color = vec4(pow(sin(pi / 2 * z), 2), 0, pow(cos(pi / 2 * z), 2), 1);
}