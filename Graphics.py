from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glm
import glfw
import time
import numpy as np
from tqdm import trange

# Geometric data for a cube (vertices and surface normals)
cube_vertices = np.array([
  -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
    0.5, -0.5, -0.5,  0.0,  0.0, -1.0, 
    0.5,  0.5, -0.5,  0.0,  0.0, -1.0, 
    0.5,  0.5, -0.5,  0.0,  0.0, -1.0, 
  -0.5,  0.5, -0.5,  0.0,  0.0, -1.0, 
  -0.5, -0.5, -0.5,  0.0,  0.0, -1.0, 

  -0.5, -0.5,  0.5,  0.0,  0.0, 1.0,
    0.5, -0.5,  0.5,  0.0,  0.0, 1.0,
    0.5,  0.5,  0.5,  0.0,  0.0, 1.0,
    0.5,  0.5,  0.5,  0.0,  0.0, 1.0,
  -0.5,  0.5,  0.5,  0.0,  0.0, 1.0,
  -0.5, -0.5,  0.5,  0.0,  0.0, 1.0,

  -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,
  -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,
  -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
  -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
  -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,
  -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,

    0.5,  0.5,  0.5,  1.0,  0.0,  0.0,
    0.5,  0.5, -0.5,  1.0,  0.0,  0.0,
    0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
    0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
    0.5, -0.5,  0.5,  1.0,  0.0,  0.0,
    0.5,  0.5,  0.5,  1.0,  0.0,  0.0,

  -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
    0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
    0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
    0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
  -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
  -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,

  -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
    0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
    0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
    0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
  -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
  -0.5,  0.5, -0.5,  0.0,  1.0,  0.0
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
  def __key_callback(window, key, _, action, mods):
    self = glfw.get_window_user_pointer(window)

    if key == glfw.KEY_ESCAPE and action == glfw.RELEASE:
      glfw.set_window_should_close(window, True)

    if key == glfw.KEY_SPACE and action == glfw.RELEASE:
      self.paused = not self.paused

    if self.paused:
      if key == glfw.KEY_RIGHT and (action == glfw.PRESS or action == glfw.REPEAT):
        self.total_time += (10 if mods & glfw.MOD_SHIFT else 1) * self.seconds_per_frame

      if key == glfw.KEY_LEFT and (action == glfw.PRESS or action == glfw.REPEAT):
        self.total_time -= (10 if mods & glfw.MOD_SHIFT else 1) * self.seconds_per_frame

    channel_changed = False

    if key == glfw.KEY_R and action == glfw.RELEASE:
      self.red_channel = not self.red_channel
      channel_changed = True

    if key == glfw.KEY_G and action == glfw.RELEASE:
      self.green_channel = not self.green_channel
      channel_changed = True

    if key == glfw.KEY_B and action == glfw.RELEASE:
      self.blue_channel = not self.blue_channel
      channel_changed = True

    if channel_changed:
      val = np.array([self.red_channel, self.green_channel, self.blue_channel], dtype=np.float32)
      glUniform3fv(self.uniformLocs["colorControls"], 1, val)

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
    glUniform3fv(self.uniformLocs["lightPos"], 1, np.array([x, y, 3.0]))

  def __scroll_callback(window, x, y):
    self = glfw.get_window_user_pointer(window)

    if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
      # If holding shift, zoom in
      self.modelZoom += 0.05 * y
    else:
      # Otherwise, rotate the model accordingly
      self.modelPitch += y
      self.modelYaw += x;

      if self.modelPitch > 89.5:
          self.modelPitch = 89.5
      if self.modelPitch < -89.5:
          self.modelPitch = -89.5

      front = np.zeros(3)
      front[0] = np.cos(glm.radians(self.modelYaw)) * np.cos(glm.radians(self.modelPitch));
      front[1] = np.sin(glm.radians(self.modelPitch));
      front[2] = np.sin(glm.radians(self.modelYaw)) * np.cos(glm.radians(self.modelPitch));
      self.modelFront = front / np.linalg.norm(front);

  def __init__(self, vfd_filepath, width, height, title, verbose=False):
    self.verbose_mode = verbose
    self.red_channel = True
    self.green_channel = True
    self.blue_channel = True

    # Before initializing graphics, read data from file
    file = open(vfd_filepath, 'r')
    lines = file.readlines()
    file.close()

    # Parses a space-separated string of floats into a NumPy array
    def parse_array(str):
      return np.array(str.split(), dtype=np.float32)

    self.positions = parse_array(lines[0])

    # Parse the input file into frame datas
    self.frames = []
    with trange(1, len(lines)) as progress_bar:
      progress_bar.set_description("Loading frames")
      for f in progress_bar:
        self.frames.append(parse_array(lines[f]))
      
    # Initialize window system
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

    # Create window and initialize OpenGL context
    if self.verbose_mode:
      print("Initializing window...")

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
    self.paused = False
    self.frame_offset = 0
    glfw.set_window_user_pointer(self.window, self)
    glfw.set_key_callback(self.window, Graphics.__key_callback)
    glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED);
    glfw.set_cursor_pos_callback(self.window, Graphics.__mouse_callback)
    glfw.set_scroll_callback(self.window, Graphics.__scroll_callback)

  def start_rendering(self, positions, initial_frame):
    self.pos_offset = positions.nbytes
    self.num_objects = int(len(positions) / 3)

    # Populate cube buffer with vertices
    self.cube_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
    glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes, cube_vertices, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Populate instance buffer with frame data (position, followed by interleaved direction and angular momentum)
    instance_data = np.array(list(positions) + list(initial_frame), dtype=np.float32)
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
    self.uniformLocs["colorControls"] = glGetUniformLocation(self.shader, "colorControls")
    self.uniformLocs["lightPos"] = glGetUniformLocation(self.shader, "lightPos")
    glUniform3fv(self.uniformLocs["colorControls"], 1, np.array([1,1,1], dtype=np.float32)) # Default value

    # Configure geometry attribute
    float_size = np.dtype(np.float32).itemsize
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * float_size, ctypes.c_void_p(0))
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Configure normals attribute
    glEnableVertexAttribArray(1)
    glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * float_size, ctypes.c_void_p(3 * float_size))
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Configure position attribute
    glEnableVertexAttribArray(2)
    glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * float_size, ctypes.c_void_p(0));
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glVertexAttribDivisor(2, 1)

    # Configure direction attribute
    glEnableVertexAttribArray(3)
    glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 6 * float_size, ctypes.c_void_p(self.pos_offset))
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glVertexAttribDivisor(3, 1)

    # Configure momentum attribute
    glEnableVertexAttribArray(4)
    glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 6 * float_size, ctypes.c_void_p(self.pos_offset + 3 * float_size))
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glVertexAttribDivisor(4, 1)

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
    glUniform3fv(self.uniformLocs["lightPos"], 1, np.array([x, y, 3.0]))

    glfw.show_window(self.window)

  def set_render_data(self, frame_data):
    glBindVertexArray(self.vertex_array)

    glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
    glBufferSubData(GL_ARRAY_BUFFER, self.pos_offset, frame_data.nbytes, frame_data)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    glBindVertexArray(0)

  def render(self):
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
    if self.verbose_mode:
      print("Freeing graphics resources...")

    glDeleteVertexArrays(1, self.vertex_array)
    glDeleteBuffers(1, self.cube_vbo)
    glDeleteBuffers(1, self.instance_vbo)
    glDeleteProgram(self.shader)
    glfw.destroy_window(self.window)

  def run(self, seconds_per_frame):
    self.seconds_per_frame = seconds_per_frame
    self.total_time = float(0)
    time_scale = 0.05
    time_scale_increment = 0.001

    if self.verbose_mode:
      print("Launching graphics...")

    self.start_rendering(self.positions, self.frames[0])

    prev_time = time.time()
    old_frame_index = 0
    while not glfw.window_should_close(self.window):
      # Check for any user inputs
      glfw.poll_events()

      # Handle key input for time dilation
      if glfw.get_key(self.window, glfw.KEY_DOWN):
        time_scale -= time_scale_increment

      if glfw.get_key(self.window, glfw.KEY_UP):
        time_scale += time_scale_increment
      
      # Update time info
      curr_time = time.time()
      delta_time = curr_time - prev_time
      prev_time = curr_time
      self.total_time += (not self.paused) * delta_time * time_scale
      self.total_time %= len(self.frames) * seconds_per_frame

      # Compute current frame index from current time
      i = int(self.total_time / self.seconds_per_frame)
      glfw.set_window_title(self.window, "Time: %.2f Seconds" % self.total_time)

      # Set the renderer to use the new frame, if necessary
      if i != old_frame_index:
        old_frame_index = i
        self.set_render_data(self.frames[i % len(self.frames)])

      self.render()

    self.stop_rendering()
    if self.verbose_mode:
      print("Done!")
