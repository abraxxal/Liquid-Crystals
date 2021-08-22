#version 330 core

// Position of the light (and camera) in world coordinates
uniform vec3 lightPos;

// Input color from the vertex shader
in vec3 color;

// Input surface normal and fragment position from the vertex shader
in vec3 normal;
in vec3 fragPos;

// Final fragment color to be used in rendering
out vec4 fragColor;

void main() {
  vec3 lightColor = vec3(1, 1, 1);
  float ambientLightFactor = 0.2;
  float diffuseLightFactor = 1.0;
  float specularLightFactor = 1.0;

  // Compute ambient lighting
  float ambient = ambientLightFactor;

  // Compute diffuse lighting
  vec3 lightDir = normalize(lightPos - fragPos);
  float diffuse = max(dot(normal, lightDir), 0.0) * diffuseLightFactor;

  // Compute specular lighting
  vec3 reflectDir = reflect(-lightDir, normal);
  float specular = pow(max(dot(lightDir, reflectDir), 0.0), 64) * specularLightFactor;

  // Add the contributes of ambient, diffuse, and specular lighting for the final color
  fragColor = vec4((ambient + diffuse + specular) * lightColor * color, 1); 
}