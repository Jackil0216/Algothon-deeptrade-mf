# Algothon 2023 README.md
- Team: DeepTrade<3

**Algorithm Purpose:** Automatically trade stocks by calling the function getMyPosition everyday with historical and current stock prices, the algotithm will then return the desired position for each stock. 

`requirements.txt` outlines all needed Python packages for this repository to run. 

To run the algorithm, run the file `eval.py` in terminal, which will output the value and return of each day, as well as the average earning per day and its standard deviation. 

**Notebook Description**
1. `EDA.ipynb` performs initial qualitative and quantatative assessment of dataset. 

2. `Position Plots.ipynb` plots the position of every stock across the period of trading, using our algorithm. 

3. `Plot.ipynb` plots the dates when holding is increased (green line) or decreased (red line), and when stop loss gets triggered (red dot for closed long position, green dot for closed short position) for every stock. 

4. `TuningHyperparameters.ipynb` tunes the hyper-parameters of the algorithm to find an optimal combination. The results are in `tuning` folder. 

