# process_documentation.md
## Data Source
The file "Ads_CTR_Optimisation.csv" contains the CTR data for multiple ads.

## Process Outline
1. Load the CSV data into a pandas DataFrame.  
2. Apply Thompson Sampling logic to select the best ad over many rounds.  
3. Track which ad is chosen, update success/failure counts.  
4. Run the script to see which ad yields the highest cumulative reward.

## Execution
1. Install dependencies (see "requirements.txt").  
2. Run the "thompson_sampling.py" script.  
3. Check the resulting console output or any generated plots.