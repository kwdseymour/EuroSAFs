import pandas as pd
import glob
import math
import matplotlib.pyplot as plt

extension = 'pkl'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

co2_data = {'lat': [], 'lon': [], 'CO2_wind': [], 'CO2_PV': [], 'CO2_bat': [], 'CO2_cap': [],'CO2_burn': [], 'total': [], 'fuel_mass': [], 'efficiency': [], 'CO2_fuel': []}

for filename in all_filenames:
    lat  = float(filename.split('_')[0])
    co2_data['lat'].append(lat)
    lon  = float(filename.split('_')[1])
    co2_data['lon'].append(lon)
    data = pd.read_pickle(filename)

    sum_wind = sum(data['wind_production_MWh'].values)*1e3 #kWh
    co2_wind = sum_wind * 0.0135 #kg
    co2_data['CO2_wind'].append(co2_wind)

    sum_PV = sum(data['PV_production_MWh'].values) * 1e3  # kWh
    co2_PV = sum_PV * 0.0638  # kg
    co2_data['CO2_PV'].append(co2_PV)

    cap_bat = math.ceil(max(data['battery_state_MWh'].values)*1e3)  # kWh
    weight_bat = cap_bat / 300 # kg
    co2_bat = weight_bat * 6.4827  # kg
    co2_data['CO2_bat'].append(co2_bat)

    sum_cap = sum(data['CO2_production_ton'].values) * 1e3 # kg
    co2_cap = sum_cap * 0.93
    co2_data['CO2_cap'].append(co2_cap)

    produced_fuel_energy = sum(data['fuel_production_MWh'])*1e3  # kWh

    efficiency = produced_fuel_energy / (sum_wind+sum_PV)
    co2_data['efficiency'].append(efficiency)

    produced_fuel_mass = produced_fuel_energy / (43000000 / 3600000)  # kg
    co2_data['fuel_mass'].append(produced_fuel_mass)

    co2_burn = produced_fuel_energy * 0.262
    co2_data['CO2_burn'].append(co2_burn)

    total = co2_wind + co2_PV + co2_bat - co2_cap + co2_burn
    co2_data['total'].append(total)

    co2_fuel = total / produced_fuel_mass # kg
    co2_data['CO2_fuel'].append(co2_fuel)


co2_balance = pd.DataFrame(co2_data)

co2_balance = co2_balance.loc[co2_balance['CO2_fuel']<0]

cost_data = pd.read_csv('Liechtenstein.csv')

co2_balance = co2_balance.merge(cost_data[['lat', 'lon', 'LCOF_liter']])

co2_balance = co2_balance.sort_values('LCOF_liter')

fig, ax = plt.subplots(1,1)
fig.suptitle('CO2 eq. [kg] Production of SAF - Netherlands (Onshore)')

ax.plot(co2_balance['LCOF_liter'], co2_balance['CO2_fuel'], 'o')
ax.set_ylabel('CO2 eq. [kg]')
ax.set_xlabel('LCOF_liter')

plt.show()

co2_balance = co2_balance.sort_values('CO2_fuel')
fig, ax = plt.subplots(3,2)
fig.suptitle('CO2 eq. [kg] Production of SAF - Netherlands (Onshore)')
row = 0
col = 0

for column in co2_balance.columns:
    if column not in ['lat', 'lon', 'CO2_fuel', 'CO2_burn', 'LCOF_liter', 'efficiency']:
        ax[row][col].plot(co2_balance['CO2_fuel'], co2_balance[column], 'o')
        ax[row][col].set_ylabel(column)
        ax[row][col].set_xlabel('CO2 eq. [kg]')
        if row < len(ax)-1 and col < len(ax[row])-1:
            col += 1
        elif row < len(ax)-1:
            row += 1
            col = 0
        elif col < len(ax[row])-1:
            col += 1

plt.show()