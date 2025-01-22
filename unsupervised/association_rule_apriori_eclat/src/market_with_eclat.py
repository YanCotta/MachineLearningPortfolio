"""
ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal) Implementation
-------------------------------------------------------------------------------
This script implements the ECLAT algorithm for market basket analysis.
ECLAT differs from Apriori by using a vertical database format and set intersections
to compute support values, making it more efficient for some datasets.

Key Concepts:
- Support: Frequency of itemset appearance in all transactions
- Vertical Data Format: Items stored with transaction IDs they appear in
- Set Intersection: Used to find common transactions between items

Author: Yan Cotta
Version: 1.0
"""

# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

class EclatAnalyzer:
    """Implementation of ECLAT algorithm for market basket analysis"""
    
    def __init__(self, min_support=0.003):
        """
        Initialize ECLAT analyzer
        
        Parameters:
            min_support (float): Minimum support threshold (0-1)
        """
        self.min_support = min_support
        self.transactions = None
        self.n_transactions = 0
        self.item_set = None
        
    def load_data(self, filepath):
        """
        Load and preprocess transaction data
        
        Parameters:
            filepath (str): Path to CSV file containing transactions
        """
        try:
            # Read data and convert to list of transactions
            dataset = pd.read_csv(filepath, header=None)
            self.transactions = []
            for i in range(len(dataset)):
                # Filter out nan values and convert to string
                transaction = [str(item) for item in dataset.values[i] if str(item) != 'nan']
                self.transactions.append(transaction)
            
            self.n_transactions = len(self.transactions)
            # Get unique items
            self.item_set = set(item for transaction in self.transactions 
                            for item in transaction)
            
            print(f"Loaded {self.n_transactions} transactions with {len(self.item_set)} unique items")
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def find_frequent_itemsets(self):
        """
        Find frequent itemsets using ECLAT approach
        Note: We use apyori implementation but with ECLAT-specific parameters
        """
        from apyori import apriori
        
        # ECLAT is essentially Apriori without confidence/lift metrics
        rules = apriori(
            transactions=self.transactions,
            min_support=self.min_support,
            min_confidence=0.0,  # Not used in ECLAT
            min_lift=0.0,       # Not used in ECLAT
            min_length=2,
            max_length=2
        )
        
        return list(rules)

    def get_results_df(self, rules):
        """Convert rules to pandas DataFrame"""
        def process_results(results):
            item1 = [tuple(result[2][0][0])[0] for result in results]
            item2 = [tuple(result[2][0][1])[0] for result in results]
            supports = [result[1] for result in results]
            return list(zip(item1, item2, supports))
            
        df = pd.DataFrame(
            process_results(rules),
            columns=['Item 1', 'Item 2', 'Support']
        )
        # Convert support to percentage for readability
        df['Support %'] = df['Support'].apply(lambda x: f"{x*100:.2f}%")
        return df

    def visualize_results(self, df, n_top=10, save_path="visualizations"):
        """
        Create visualizations of the results
        
        Parameters:
            df (DataFrame): Results DataFrame
            n_top (int): Number of top results to visualize
            save_path (str): Directory to save visualizations
        """
        Path(save_path).mkdir(exist_ok=True)
        
        # Plot top item pairs by support
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df.nlargest(n_top, 'Support'),
            x='Support',
            y='Item 1',
            palette='viridis'
        )
        plt.title(f'Top {n_top} Frequent Item Pairs')
        plt.tight_layout()
        plt.savefig(f'{save_path}/top_pairs.png')
        plt.close()

def main():
    """Main execution function"""
    try:
        # Initialize analyzer
        analyzer = EclatAnalyzer(min_support=0.003)
        
        # Load and process data
        if not analyzer.load_data('Market_Basket_Optimisation.csv'):
            return
            
        # Find frequent itemsets
        print("Finding frequent itemsets...")
        rules = analyzer.find_frequent_itemsets()
        
        # Convert to DataFrame and display results
        results_df = analyzer.get_results_df(rules)
        print("\nTop 10 Frequent Item Pairs:")
        print(results_df.nlargest(10, 'Support').to_string(index=False))
        
        # Create visualizations
        analyzer.visualize_results(results_df)
        
        # Save results
        results_df.to_csv('eclat_results.csv', index=False)
        print("\nResults saved to 'eclat_results.csv'")
        
    except Exception as e:
        print(f"Error in execution: {str(e)}")

if __name__ == "__main__":
    main()