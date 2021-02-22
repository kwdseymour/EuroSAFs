countries=("Austria" "Belgium" "Bulgaria" "Croatia" "Cyprus" "Czech_Republic" "Denmark" "Estonia" 
"Finland" "France" "Germany" "Greece" "Hungary" "Iceland" "Ireland" "Italy" "Latvia" "Liechtenstein"
"Lithuania" "Luxembourg" "Malta" "Netherlands" "Norway" "Poland" "Portugal" "Romania" "Slovakia" "Slovenia" 
"Spain" "Sweden" "Switzerland" "United_Kingdom")
# countries=("Spain" "Portugal" "United_Kingdom" "Ireland" "Iceland")

# loop through the countries list: 
for country in "${countries[@]}"
do 
    bsub -n 32 -W 30:00 -J \"S02_$country-1\" -oo $HOME/EuroSAFs/results/02_plant_optimization/lsf.$country-1.txt python $HOME/EuroSAFs/scripts/03_plant_optimization/02_plant_optimization.py -d $HOME/EuroSAFs -c $country -m 0.01 -i 30 -b 1 -v -s 
	sleep 0.01
done

countries=("Finland" "France" "Germany" "Greece" "Iceland" "Italy" "Norway" "Poland" "Romania" "Spain" "Sweden" "United_Kingdom")
for country in "${countries[@]}"
do 
    bsub -n 32 -W 30:00 -J \"S02_$country-2\" -oo $HOME/EuroSAFs/results/02_plant_optimization/lsf.$country-2.txt python $HOME/EuroSAFs/scripts/03_plant_optimization/02_plant_optimization.py -d $HOME/EuroSAFs -c $country -m 0.01 -i 30 -b 2 -v -s 
	sleep 0.01
done

countries=("Finland" "France" "Italy" "Norway" "Spain" "Sweden" "United_Kingdom")
for country in "${countries[@]}"
do 
    bsub -n 32 -W 30:00 -J \"S02_$country-3\" -oo $HOME/EuroSAFs/results/02_plant_optimization/lsf.$country-3.txt python $HOME/EuroSAFs/scripts/03_plant_optimization/02_plant_optimization.py -d $HOME/EuroSAFs -c $country -m 0.01 -i 30 -b 3 -v -s 
	sleep 0.01
done

countries=("Norway" "Sweden")
for country in "${countries[@]}"
do 
    bsub -n 32 -W 30:00 -J \"S02_$country-4\" -oo $HOME/EuroSAFs/results/02_plant_optimization/lsf.$country-4.txt python $HOME/EuroSAFs/scripts/03_plant_optimization/02_plant_optimization.py -d $HOME/EuroSAFs -c $country -m 0.01 -i 30 -b 4 -v -s 
	sleep 0.01
done