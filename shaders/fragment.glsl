#version 330 core 

// Input color from the vertex shader
in vec4 color;

// Final fragment color to be used in rendering
out vec4 FragColor;

void main() { 
  // For now, just use the same color everywhere (supplied in the vertex shader)
  FragColor = color; 
}