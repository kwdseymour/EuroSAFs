import plotly.graph_objects as go

W_E = 8.1

E_F = 0.54
E_O = 7.56

C_F = 4.19

F_G = 0.3
F_K = 0.3
F_D = 0.3
F_O = 2
F_W = 1.73


source = [0,                    # W
          1, 1,                 # E
          2,                    # C
          3, 3, 3, 3, 3]        # F

target = [1,
          3, 4,
          3,
          5, 6, 7, 4, 0]

value = [W_E,
         E_F, E_O,
         C_F,
         F_G, F_K, F_D, F_O, F_W]

label = ['Water Supply', 'Electrolysis', 'DAC', 'FT Synthesis', 'Oxygen', 'Gasoline', 'Kerosene', 'Diesel']
color_link = ['rgba(16, 188, 188, 0.4)',
              'rgba(255, 192, 0,0.4)', 'rgba(158, 28, 76, 0.4)',
              'rgba(0, 62, 70, 0.4)',
              'rgba(204, 208, 34, 0.4)', 'rgba(204, 208, 34, 0.4)', 'rgba(204, 208, 34, 0.4)', 'rgba(158, 28, 76, 0.4)', 'rgba(16, 188, 188, 0.4)']

color_node = ['rgba(16, 188, 188, 1)',
              'rgba(255, 192, 0, 1)',
              'rgba(0, 62, 70, 1)',
              'rgba(158, 28, 76, 1)',
              'rgba(158, 28, 76, 1)',
              'rgba(10, 114, 10, 1)',
              'rgba(10, 114, 10, 1)',
              'rgba(10, 114, 10, 1)']

link = dict(source = source, target = target, value = value, color=color_link)
node = dict(label=label,
            pad=50,
            thickness=10,
            color=color_node,
            x=[ 0.05,  0.35,  0.05,  0.65,  0.95,  0.95,  0.95, 0.95],
            y=[0.308, 0.308, 0.591, 0.576, 0.338, 0.564, 0.577, 0.59])
data = go.Sankey(link = link, node=node)

fig = go.Figure(data)
fig.show()