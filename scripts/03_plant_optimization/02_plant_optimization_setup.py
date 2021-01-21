import os
import plant_optimization as pop
import math

source = 'off_out'                          # 'on_cost', 'on_out', 'off_cost', 'off_out'

#countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech_Republic', 'Denmark','Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland','Italy', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands','Norway', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden','Switzerland', 'United_Kingdom']

countries = ['Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Denmark','Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Iceland', 'Ireland','Italy', 'Latvia', 'Lithuania', 'Malta', 'Netherlands','Norway', 'Poland', 'Portugal', 'Romania', 'Spain', 'Sweden', 'United_Kingdom']

for country in countries:
    results_pathes = {
                   'on_cost': os.path.join('.', 'results', '02_plant_optimization', 'optimal_cost', country),
                   'on_out': os.path.join('.', 'results', '02_plant_optimization', 'optimal_out', country),
                   'off_cost': os.path.join('.', 'results', '02_plant_optimization','optimal_cost', country, 'coast'),
                   'off_out': os.path.join('.', 'results', '02_plant_optimization', 'optimal_out', country, 'coast')
                  }

    results_path = results_pathes[source]
    points = pop.get_country_points(country, source)

    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    eval_points = []

    for point in points:
        try:
            file = open(os.path.join(results_path, country + str(point[0]) + str(point[1]) + '.csv'))
            file.close()
        except FileNotFoundError:
            eval_points.append(point)

    no_points = len(eval_points)
    ppr = 30
    rounds = math.ceil(no_points/ppr)

    for i in range(0,rounds):
        points_round = eval_points[i*ppr:(i+1)*ppr]
        exe = 'python ./scripts/03_plant_optimization/02_plant_optimization.py -c \'' + country + '\' -d \'.\' -s \'' + source + '\' -points \'' + str(points_round) +'\''
        print(exe)
        exit()
        exe = 'bsub -n 4 -W 24:00 -oo ./results/02_plant_optimization/lsf.' + country + '_round_0' + str(i) + '.txt python ./scripts/03_plant_optimization/02_plant_optimization.py -c \'' + country + '\' -d \'.\' -s \'' + source + '\' -points \'' + str(points_round) +'\'' + ' -m 0.01 -i 30 -v'
        os.system(exe)