import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

specs = pd.read_excel('./data/plant_assumptions.xlsx',sheet_name='data')

columns = np.arange(0,1000)

scenarios = pd.DataFrame(columns=columns)

components = []
functions = {}

for idx, row in specs.iterrows():
    name = row['specification'].split('_')
    if name[0] not in components:
        components.append(name[0])
        functions[name[0]] = ['_'.join(name[1:])]
    else:
        functions[name[0]].append('_'.join(name[1:]))

names = []

for component in components:
    if 'max_CAPEX' in functions[component]:
        var_CAPEX = (specs.loc[specs['specification']==str(component+'_max_CAPEX'), 'value'].values - specs.loc[specs['specification']==str(component+'_min_CAPEX'), 'value'].values) / 2
        sig_CAPEX = var_CAPEX / 3
        mu_CAPEX = var_CAPEX + specs.loc[specs['specification']==str(component+'_min_CAPEX'), 'value'].values

        scenarios = scenarios.append([np.random.normal(mu_CAPEX, sig_CAPEX, 1001)], ignore_index=True)
        names.append(component+'_CAPEX')

    if 'max_OPEX' in functions[component]:
        var_OPEX = (specs.loc[specs['specification'] == str(component + '_max_OPEX'), 'value'].values - specs.loc[specs['specification'] == str(component + '_min_OPEX'), 'value'].values) / 2
        sig_OPEX = var_OPEX / 3
        mu_OPEX = var_OPEX + specs.loc[specs['specification'] == str(component + '_min_OPEX'), 'value'].values

        scenarios = scenarios.append([np.random.normal(mu_OPEX, sig_OPEX, 1001)], ignore_index=True)
        names.append(component + '_OPEX')

    if 'max_DECEX' in functions[component]:
        var_DECEX = (specs.loc[specs['specification'] == str(component + '_max_DECEX'), 'value'].values - specs.loc[specs['specification'] == str(component + '_min_DECEX'), 'value'].values) / 2
        sig_DECEX = var_DECEX / 3
        mu_DECEX = var_DECEX + specs.loc[specs['specification'] == str(component + '_min_DECEX'), 'value'].values

        scenarios = scenarios.append([np.random.normal(mu_DECEX, sig_DECEX, 1001)], ignore_index=True)
        names.append(component + '_DECEX')

    if 'max_lifetime' in functions[component]:
        var_lifetime = (specs.loc[specs['specification'] == str(component + '_max_lifetime'), 'value'].values - specs.loc[
            specs['specification'] == str(component + '_min_lifetime'), 'value'].values) / 2
        sig_lifetime = var_lifetime / 3
        mu_lifetime = var_lifetime + specs.loc[specs['specification'] == str(component + '_min_lifetime'), 'value'].values

        scenarios = scenarios.append([np.random.normal(mu_lifetime, sig_lifetime, 1001)], ignore_index=True)
        names.append(component + '_lifetime')

scenarios['specification'] = names

print(scenarios)