# countries=("Austria" "Belgium" "Bulgaria" "Croatia" "Cyprus" "Czech_Republic" "Denmark" "Estonia" 
# "Finland" "France" "Germany" "Greece" "Hungary" "Iceland" "Ireland" "Italy" "Latvia" "Liechtenstein"
# "Lithuania" "Luxembourg" "Malta" "Netherlands" "Norway" "Poland" "Portugal" "Romania" "Slovakia" "Slovenia" 
# "Spain" "Sweden" "Switzerland" "United_Kingdom")
countries=("Switzerland" "Austria" "Germany" "Italy" "France")

# loop through the countries list: 
for country in "${countries[@]}"
do 
    bsub -n 32 -W 48:00 -J \"S02_$country\" -oo $HOME/EuroSAFs/results/02_plant_optimization/lsf.$country.txt python $HOME/EuroSAFs/scripts/03_plant_optimization/02_plant_optimization.py -d $HOME/EuroSAFs -c $country -m 0.01 -i 30 -v -s 
	sleep 0.01
done
