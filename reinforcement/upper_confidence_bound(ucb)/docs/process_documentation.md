# process_documentation.md
This repository demonstrates the Upper Confidence Bound (UCB) algorithm on the Ads_CTR_Optimisation.csv dataset.

Process Outline:
1. Load the dataset (Ads_CTR_Optimisation.csv).
2. Use the UCB algorithm in upper_confidence_bound.py to select ads across multiple rounds.
3. Track cumulative rewards and ad selections.
4. Visualize or print the final results.

Refer to the main repository LICENSE for usage rights.

Key Points:
• The dataset is structured with each row representing a round of ad selection possibilities.  
• The script uses mathematical properties of confidence bounds to balance exploration and exploitation.  
• Results indicate which ad yields the highest click-through rate over many trials.