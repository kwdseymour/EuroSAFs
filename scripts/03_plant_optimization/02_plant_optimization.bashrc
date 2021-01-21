countries=("Austria" "Belgium" "Bulgaria" "Croatia" "Cyprus" "Czech_Republic" "Denmark" "Estonia" 
"Finland" "France" "Germany" "Greece" "Hungary" "Iceland" "Ireland" "Italy" "Latvia" "Liechtenstein"
"Lithuania" "Luxembourg" "Malta" "Netherlands" "Norway" "Poland" "Portugal" "Romania" "Slovakia" "Slovenia" 
"Spain" "Sweden" "Switzerland" "United_Kingdom")
# countries=("Malta")

# Run the setup script
python $HOME/SAFlogistics/scripts/02_plant_optimization/02_setup.py -d $HOME/SAFlogistics

# loop through the countries list: 
for country in "${countries[@]}"
do 
    bsub -n 32 -W 48:00 -J \"S02_$country\" -oo $HOME/SAFlogistics/results/02_plant_optimization/lsf.$country.txt python $HOME/SAFlogistics/scripts/02_plant_optimization/02_plant_optimization.py -d $HOME/SAFlogistics -c $country -m 0.01 -i 30 -v
	sleep 0.01
done
