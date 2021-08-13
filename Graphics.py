import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glm
import glfw

# Geometric data for a cube
cube_vertices = np.array([
  -0.5, -0.5, -0.5, 
  0.5, -0.5, -0.5,
  0.5,  0.5, -0.5,
  0.5,  0.5, -0.5,
  -0.5,  0.5, -0.5,
  -0.5, -0.5, -0.5,

  -0.5, -0.5,  0.5,
  0.5, -0.5,  0.5,
  0.5,  0.5,  0.5,
  0.5,  0.5,  0.5,
  -0.5,  0.5,  0.5,
  -0.5, -0.5,  0.5,

  -0.5,  0.5,  0.5,
  -0.5,  0.5, -0.5,
  -0.5, -0.5, -0.5,
  -0.5, -0.5, -0.5,
  -0.5, -0.5,  0.5,
  -0.5,  0.5,  0.5,

  0.5,  0.5,  0.5,
  0.5,  0.5, -0.5,
  0.5, -0.5, -0.5,
  0.5, -0.5, -0.5,
  0.5, -0.5,  0.5,
  0.5,  0.5,  0.5,

  -0.5, -0.5, -0.5,
  0.5, -0.5, -0.5,
  0.5, -0.5,  0.5,
  0.5, -0.5,  0.5,
  -0.5, -0.5,  0.5,
  -0.5, -0.5, -0.5,

  -0.5,  0.5, -0.5,
  0.5,  0.5, -0.5,
  0.5,  0.5,  0.5,
  0.5,  0.5,  0.5,
  -0.5,  0.5,  0.5,
  -0.5,  0.5, -0.5,
], dtype=np.float32)

# The Graphics class handles the creation of a window and the rendering of a static vector field. Functionality for
# animation is provided by dynamically updating the vector field. Basic user controls (zoom, pan, etc.) are also 
# provided, as well as colors encoding the state of the vector field.
#
# User Controls:
# Move the mouse to pan, scroll to rotate the model, and shift+scroll to zoom in and out.
#
# Color Key:
#  - Red encodes how upright the vector is (proximity to the z-axis)
#  - Blue encodes how lateral the vector is (proximity to the xy-plane)
#  - Green encodes the magnitude of the vector's angular momentum
class Graphics:
  def __mouse_callback(window, x, y):
    self = glfw.get_window_user_pointer(window)
    x, y = glfw.get_cursor_pos(window)
    width, height = glfw.get_window_size(window)

    # Invert horizontal axis for some reason
    x = -(x - width / 2) / width
    y = (y - height / 2) / height
    
    # Set camera location to cursor position and send the view matrix to the shader
    camera = glm.lookAt(np.array([x, y, 3.0]), np.array([x, y, 2.0]), np.array([0.0, 1.0, 0.0]))
    glUniformMatrix4fv(self.uniformLocs["view"], 1, GL_TRUE, np.array(camera).reshape(16))

  def __scroll_callback(window, x, y):
    self = glfw.get_window_user_pointer(window)

    if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
      # If holding shift, zoom in
      self.modelZoom += 0.05 * y
    else:
      # Otherwise, rotate the model accordingly
      self.modelPitch += -y
      self.modelYaw += x;

      if self.modelPitch > 89.0:
          self.modelPitch = 89.0
      if self.modelPitch < -89.0:
          self.modelPitch = -89.0

      front = np.zeros(3)
      front[0] = np.cos(glm.radians(self.modelYaw)) * np.cos(glm.radians(self.modelPitch));
      front[1] = np.sin(glm.radians(self.modelPitch));
      front[2] = np.sin(glm.radians(self.modelYaw)) * np.cos(glm.radians(self.modelPitch));
      self.modelFront = front / np.linalg.norm(front);

  def __init__(self, width, height, title):
    glfw.init()

    # Use OpenGL 3.3 core profile with forward compatibility
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    # Use 32 samples for multisampling (anti-aliasing)
    glfw.window_hint(glfw.SAMPLES, 32)

    # Window should be invisible until ready for rendering
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    # Create window
    self.window = glfw.create_window(width, height, title, None, None)
    if not self.window:
      print(glfw.get_error())
      exit()

    glfw.make_context_current(self.window)

    # Set global OpenGL state
    glViewport(0, 0, width, height)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE) # Enable multisampling

    # Set input callbacks
    glfw.set_window_user_pointer(self.window, self)
    glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED);
    glfw.set_cursor_pos_callback(self.window, Graphics.__mouse_callback)
    glfw.set_scroll_callback(self.window, Graphics.__scroll_callback)

  def __extract_instance_data(self, frame):
    # Extract director field data from current frame and augment it with positions
    nfield = frame[0]
    num_x, num_y, num_z, _ = np.shape(nfield)
    self.num_objects = num_x * num_y * num_z

    instance_data = []
    for i in range(0, num_x):
      for j in range(0, num_y):
        for k in range(0, num_z):
          direction = nfield[i,j,k]
          instance_data.extend(np.array([i - num_x / 2, j - num_y / 2, k - num_z / 2]) * 1/16)
          instance_data.extend(direction)

    return np.array(instance_data, dtype=np.float32).flatten()

  def start_rendering(self, initial_frame):
    # Populate cube buffer with vertices
    self.cube_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
    glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes, cube_vertices, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    instance_data = self.__extract_instance_data(initial_frame)

    # Populate instance buffer with frame data (position and direction)
    self.instance_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
    glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Initialize the vertex array to store all the state processed below
    self.vertex_array = glGenVertexArrays(1)
    glBindVertexArray(self.vertex_array)

    # Compile shaders
    vertFile = open("shaders/vertex.glsl", "r")
    fragFile = open("shaders/fragment.glsl", "r")
    vertShader = shaders.compileShader(vertFile.read(), GL_VERTEX_SHADER)
    fragShader = shaders.compileShader(fragFile.read(), GL_FRAGMENT_SHADER)
    self.shader = shaders.compileProgram(vertShader, fragShader)
    glUseProgram(self.shader)

    # Compute uniform locations for rendering
    self.uniformLocs = {}
    self.uniformLocs["proj"] = glGetUniformLocation(self.shader, "proj")
    self.uniformLocs["view"] = glGetUniformLocation(self.shader, "view")
    self.uniformLocs["front"] = glGetUniformLocation(self.shader, "front")
    self.uniformLocs["zoom"] = glGetUniformLocation(self.shader, "zoom")

    # Configure geometry vertex attribute
    float_size = np.dtype(np.float32).itemsize
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * float_size, ctypes.c_void_p(0))
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Configure position vertex attribute
    glEnableVertexAttribArray(1)
    glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * float_size, ctypes.c_void_p(0));
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glVertexAttribDivisor(1, 1)

    # Configure direction vertex attribute
    glEnableVertexAttribArray(2)
    glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 6 * float_size, ctypes.c_void_p(3 * float_size))
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glVertexAttribDivisor(2, 1)

    # Done setting vertex array state
    glBindVertexArray(0)

    # Initialize model orientation variables
    self.modelFront = np.array([0, 0, -1], dtype=np.float32)
    self.modelYaw = -90.0
    self.modelPitch = 0.0
    self.modelZoom = 1.0

    # Set camera location to initial cursor position
    x, y = glfw.get_cursor_pos(self.window)
    width, height = glfw.get_window_size(self.window)

    x = -(x - width / 2) / width
    y = (y - height / 2) / height
    camera = glm.lookAt(np.array([x, y, 3.0]), np.array([x, y, 2.0]), np.array([0.0, 1.0, 0.0]))
    glUniformMatrix4fv(self.uniformLocs["view"], 1, GL_TRUE, np.array(camera).reshape(16))

    glfw.show_window(self.window)

  def set_render_data(self, frame):
    glBindVertexArray(self.vertex_array)

    instance_data = self.__extract_instance_data(frame)

    glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
    glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    glBindVertexArray(0)

  def window_is_open(self):
    return not glfw.window_should_close(self.window)

  def set_window_title(self, title):
    glfw.set_window_title(self.window, title)

  def render(self):
    # Check for any user inputs
    glfw.poll_events()

    # Clear screen to background color
    glClearColor(0.1, 0.1, 0.1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Prepare shaders
    glUseProgram(self.shader)

    # Send 3D projection matrix to shader
    proj = glm.perspective(glm.radians(45), 1, 0.1, 100)
    glUniformMatrix4fv(self.uniformLocs["proj"], 1, GL_TRUE, np.array(proj).reshape(16))

    # Send model attributes to shader (rotation and zoom)
    glUniform3fv(self.uniformLocs["front"], 1, self.modelFront)
    glUniform1f(self.uniformLocs["zoom"], self.modelZoom)

    # Render all vectors at once (to the hidden framebuffer)
    glBindVertexArray(self.vertex_array)
    glDrawArraysInstanced(GL_TRIANGLES, 0, 36, self.num_objects)
    glBindVertexArray(0)

    # Swap the hidden framebuffer with the visible one, displaying the results of the above draw call
    glfw.swap_buffers(self.window)

  def stop_rendering(self):
    glDeleteVertexArrays(1, self.vertex_array)
    glDeleteProgram(self.shader)
    glfw.destroy_window(self.window)
