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

// Model color to send to the fragment shader
out vec4 color;

// Implementation of glm's lookAt function, which orients a vector (located at "eye") to face towards a point in space ("at"). The "up" parameter is necessary for uniqueness; it represents where "eye" perceives the direction up to be.
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

// Scale matrix
mat4 scale(vec3 diag)
{
  return mat4(
    vec4(diag.x, 0, 0, 0),
    vec4(0, diag.y, 0, 0),
    vec4(0, 0, diag.z, 0),
    vec4(0, 0, 0, 1)
  );
}

// Translation matrix
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
  mat4 model = mat4(1); // Model transform matrix. Note the computations below should be read in reverse order
  model *= lookAt(vec3(0), front, vec3(0, 1, 0)); // Orient the entire vector field to match user input
  model *= scale(vec3(zoom)); // Now, the entire vector field by the zoom factor
  model *= translate(position); // Then move it to its location in the field
  model *= lookAt(vec3(0), direction, vec3(0, 1, 0)); // Then orient each model to face its specified direction
  model *= scale(vec3(0.01, 0.01, 0.05)); // First, scale down the cube model and shape it into a rectangular prism

  gl_Position = proj * view * model * vec4(cube_vertex, 1);

  float z = direction.z;
  float pi = 3.141592;
  // Specify colors (as seen in Graphics.py documentation)
  color = vec4(pow(sin(pi / 2 * z), 2), 0, pow(cos(pi / 2 * z), 2), 1);
}