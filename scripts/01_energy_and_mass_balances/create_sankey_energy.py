import plotly.graph_objects as go

# Hydrogen Electrolysis
# H_2 [kWh] = 1.667 Electricity [kWh] - 0.667 Energy [kWh] + 15 Water [kg]

# Direct Air Capture
# CO_2 [kg] = 0.4 Electricity [kWh] + 1.6 Heat [kWh] - 2 Energy [kWh]

# Fisher-Tropsch Synthesis
# HC [kWh] = 1.497   H_2 [kWh]                                       + 0.314   CO_2 [kg]                                                  + 0.134 Electricity [kWh] - 0.326 Heat [kWh] - 0.305 Energy [kWh]
# HC [kWh] = 1.497 * ( 1.667 Electricity [kWh] - 0.667 Energy [kWh]) + 0.314 * ( 0.4 Electricity [kWh] + 1.6 Heat [kWh] - 2 Energy [kWh]) + 0.134 Electricity [kWh] - 0.326 Heat [kWh] - 0.305 Energy [kWh]
# HC [kWh] = 2.495 Electricity [kWh] - 0.998 Energy [kWh]            + 0.126 Electricity [kWh] + 0.502 Heat [kWh] - 0.628 Energy [kWh])   + 0.134 Electricity [kWh] - 0.326 Heat [kWh] - 0.305 Energy [kWh]

E_H = 2.495
E_C = 0.126
E_F = 0.134
E_Q = 0.1764

H_F = 1.497
H_L = 0.998

C_L = 0.628

Q_C = 0.1764

F_C = 0.326
F_G = 0.31
F_K = 0.44
F_D = 0.25
F_L = 0.305

source = [0, 0, 0, 0,           # E
          1, 1,                 # H             5  # L
          2,                    # C
          3,                    # Q
          4, 4, 4, 4, 4]        # F

target = [1, 2, 4, 3,
          4, 5,
          5,
          2,
          2, 6, 7, 8, 5]

value = [E_H, E_C, E_F, E_Q,
         H_F, H_L,
         C_L,
         Q_C,
         F_C, F_G, F_K, F_D, F_L]

label = ['Energy Source', 'Electrolysis', 'DAC', 'Heat Source', 'FT Synthesis', 'Lost Energy', 'Gasoline', 'Kerosene', 'Diesel']
color_link = ['rgba(16, 188, 188, 0.4)', 'rgba(16, 188, 188, 0.4)', 'rgba(16, 188, 188, 0.4)', 'rgba(16, 188, 188, 0.4)',
              'rgba(255, 192, 0, 0.4)', 'rgba(0, 0, 0, 0.4)',
              'rgba(0, 0, 0, 0.4)',
              'rgba(158, 28, 76, 0.4)',
              'rgba(158, 28, 76, 0.4)', 'rgba(204, 208, 34, 0.4)', 'rgba(204, 208, 34, 0.4)', 'rgba(204, 208, 34, 0.4)', 'rgba(0, 0, 0, 0.4)']

color_node = ['rgba(16, 188, 188, 1)',
              'rgba(255, 192, 0, 1)',
              'rgba(0, 62, 70, 1)',
              'rgba(158, 28, 76, 1)',
              'rgba(204, 208, 34, 1)',
              'rgba(0, 0, 0, 1)',
              'rgba(10, 114, 10, 1)',
              'rgba(10, 114, 10, 1)',
              'rgba(10, 114, 10, 1)']

link = dict(source = source, target = target, value = value, color=color_link)
node = dict(label=label,
            pad=50,
            thickness=10,
            color=color_node,
            x=[  0,  0.25,  0.75,  0.25,   0.5,     1,     1,     1,     1],
            y=[0.5, 0.423, 0.757, 0.822, 0.267, 0.578, 0.085, 0.187, 0.281])
data = go.Sankey(link = link, node=node)

fig = go.Figure(data)
fig.show()

#----------------------------------------------------------------------------