#version 330 core

// Geometry data
layout (location = 0) in vec3 cube_vertex;
layout (location = 1) in vec3 cube_normal;

// Molecule data
layout (location = 2) in vec3 position;
layout (location = 3) in vec3 direction;
layout (location = 4) in vec3 momentum;

// World space transforms
uniform mat4 view;
uniform mat4 proj;

// Orientation and zoom parameters
uniform vec3 front;
uniform float zoom;

// A 1-0 vector representing whether to render each color channel (RGB)
uniform vec3 colorControls;

// Data to send to the fragment shader
out vec3 color;
out vec3 normal;
out vec3 fragPos;

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

// Implementation of glm's lookAt function, which orients a vector (located at "eye") to face towards a point in space 
// ("at"). The "up" parameter is necessary for uniqueness; it represents where "eye" perceives the direction up to be.
mat4 lookAt(vec3 eye, vec3 at)
{
  vec3 up = vec3(0, 1, 0);

  vec3 zaxis = normalize(at - eye);    
  vec3 xaxis = normalize(cross(zaxis, up));
  vec3 yaxis = cross(xaxis, zaxis);

  // In the special case that zaxis faces straight up (creates a singular matrix if not handled separately)
  if (zaxis.x == 0 && zaxis.z == 0)
  {
    if (zaxis.y < 0)
      return transpose(mat4(
        vec4(1, 0, 0, -dot(xaxis, eye)),
        vec4(0, 0, -1, -dot(yaxis, eye)),
        vec4(0, 1, 0, -dot(zaxis, eye)),
        vec4(0, 0, 0, 1)
      ));
    else
      return transpose(mat4(
        vec4(1, 0, 0, -dot(xaxis, eye)),
        vec4(0, 0, 1, -dot(yaxis, eye)),
        vec4(0, -1, 0, -dot(zaxis, eye)),
        vec4(0, 0, 0, 1)
      ));
  } else {
    return transpose(mat4(
      vec4(-xaxis, -dot(xaxis, eye)),
      vec4(yaxis, -dot(yaxis, eye)),
      vec4(zaxis, -dot(zaxis, eye)),
      vec4(0, 0, 0, 1)
    ));
  }
}

// Smush an unbounded value between 0 and 1
float regularize(float x) {
  return 1.0 - 1.0 / (1.0 + pow(0.1 * x, 4));
}

void main()
{
  mat4 model = mat4(1); // Model transform matrix. Note the computations below should be read in reverse order
  model *= lookAt(vec3(0), front); // Orient the entire vector field to match user input
  model *= scale(vec3(zoom)); // Now, the entire vector field by the zoom factor
  model *= translate(position); // Then move it to its location in the field
  model *= lookAt(vec3(0), direction); // Then orient each model to face its specified direction
  model *= scale(vec3(0.75, 0.75, 7.5) / 200); // First, scale down the cube model and shape it into a rectangular prism

  // Compute position and normal data
  gl_Position = proj * view * model * vec4(cube_vertex, 1);
  fragPos = vec3(model * vec4(cube_vertex, 1));
  normal = normalize(mat3(transpose(inverse(model))) * cube_normal);  

  // Specify colors (as seen in Graphics.py documentation)
  float z = direction.z;
  float pi = 3.141592;
  vec3 rgb = vec3(pow(sin(pi / 2 * z), 2) / 3, regularize(length(momentum)) / 3, pow(cos(pi / 2 * z), 2) / 3);
  color = rgb * colorControls;
}