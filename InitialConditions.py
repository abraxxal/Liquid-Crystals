import numpy as np

def test(x, y, z):
  return (np.sin(10 * x), 0, 1)

def wave1(x, y, z):
  return (x * np.sin(10 * y) + 0.1, 0, y * np.sin(10 * x) + 0.1)

def wave2(x, y, z):
  r = x**2 + y**2
  return (200 * (x * y)**2 / np.exp(r), -200 * (x * y)**2 / np.exp(r), 1)

def test3d(x, y, z):
  return (x * np.sin(10 * y) + 0.1, np.sin(10 * z) - 0.1, y * np.sin(10 * x) + 0.1)