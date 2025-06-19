import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Equation for H
#This is scaled version, so m1 = m2 = 1 and so on
a_sp, b_sp, la_sp, lb_sp = sp.symbols('a_sp b_sp la_sp lb_sp')
H = (-2) * sp.cos(a_sp) - sp.cos(a_sp + b_sp) + (la_sp ** 2 - 2 * (1 + sp.cos(b_sp)) * la_sp * lb_sp + (3 + 2 * sp.cos(b_sp)) * lb_sp ** 2) / (3 - sp.cos(2 * b_sp))
#Partial dirivitives from H
dH_da = sp.diff(H, a_sp)
dH_db = sp.diff(H, b_sp)
dH_dla = sp.diff(H, la_sp)
dH_dlb = sp.diff(H, lb_sp)

#Hamilton equations for da/dt, etc
get_da_dt = sp.lambdify((a_sp, b_sp, la_sp, lb_sp), dH_dla, "numpy")
get_db_dt = sp.lambdify((a_sp, b_sp, la_sp, lb_sp), dH_dlb, "numpy")
get_dla_dt = sp.lambdify((a_sp, b_sp, la_sp, lb_sp), -dH_da, "numpy")
get_dlb_dt = sp.lambdify((a_sp, b_sp, la_sp, lb_sp), -dH_db, "numpy")

#Poincare section: track a = 0 during integration
def event_t(t, y):
  return  y[0]

#Setup for event_t
event_t.terminal = False
event_t.direction = 1

#a, b should not exceed pi
def normalize_angle(b):
    return np.mod(b + np.pi, 2 * np.pi) - np.pi

#Hamilton equations for integration
def the_dp(t, y):
  a, b, la, lb = y
  a = normalize_angle(a)
  b = normalize_angle(b)
  da_dt = get_da_dt(a, b, la, lb)
  db_dt = get_db_dt(a, b, la, lb)
  dla_dt = get_dla_dt(a, b, la, lb)
  dlb_dt = get_dlb_dt(a, b, la, lb)
  return [da_dt, db_dt, dla_dt, dlb_dt]

#color for visual
colors = ['red', 'blue', 'green', 'purple', 'orange',
    'cyan', 'magenta', 'yellow', 'black', 'gray',
    'pink', 'lime', 'teal', 'indigo', 'violet',
    'brown', 'olive', 'navy', 'maroon', 'turquoise',
    'coral', 'gold', 'salmon', 'crimson', 'darkgreen']

#Integration with solve_ivp
def integration(y0):
  the_t_span = (0, 5000)
  s = solve_ivp(the_dp, the_t_span, y0, method='RK45', events=event_t, rtol=1e-8, atol=1e-8)
  return s

#Visualisation
def visual(s, y0):
  a0, b0, la0, lb0 = y0
  y_events = s.y_events[0] if len(s.y_events) > 0 else np.array([])
  t_events = s.t_events[0] if len(s.t_events) > 0 else np.array([])
  b_events = y_events[:, 1] if len(y_events) > 0 else []
  for item in b_events:
     item = normalize_angle(item)
  lb_events = y_events[:, 3] if len(y_events) > 0 else []
  plt.figure(figsize=(8, 6))
  random_color = np.random.choice(colors)
  if len(b_events) > 0:
      plt.scatter(b_events, lb_events, color=random_color, marker='o', s=10)
  else:
      print("No Poincaré section points found.")
  plt.xlabel('b')
  plt.ylabel('lb')
  plt.legend()
  plt.grid(True)
  plt.title(f'Poincaré Section Points (a=0) for a={a0}, b={b0}, la={la0}, lb={lb0}')
  plt.show()

#do everything
def do_the_thing(a0, b0, la0, lb0):
  the_s = integration([a0, b0, la0, lb0])
  visual(the_s, [a0, b0, la0, lb0])
#run
a0 = float(input("a: "))
b0 = float(input("b: "))
la0 = float(input("la: "))
lb0 = float(input("lb: "))
do_the_thing(a0, b0, la0, lb0)


