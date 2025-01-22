"""
Market Basket Analysis using Apriori Algorithm
Author: Yan Cotta
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori
from pathlib import Path
from datetime import datetime

class MarketAnalyzer:
    def __init__(self, support=0.003, confidence=0.2, lift=3):
        self.params = {
            'min_support': support,
            'min_confidence': confidence,
            'min_lift': lift,
            'min_length': 2,
            'max_length': 2
        }
        self.rules = None
        self.transactions = None

    def load_data(self, filepath):
        try:
            df = pd.read_csv(filepath, header=None)
            self.transactions = [
                [str(item) for item in row if str(item) != 'nan']
                for row in df.values
            ]
            return len(self.transactions)
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def analyze(self):
        if not self.transactions:
            raise ValueError("No data loaded")
            
        self.rules = apriori(transactions=self.transactions, **self.params)
        results = self._format_results()
        self._create_visualizations(results)
        return results

    def _format_results(self):
        """Convert association rules to DataFrame"""
        rows = []
        for item in list(self.rules):
            pair = item[2][0]
            rows.append({
                'Left': list(pair[0])[0],
                'Right': list(pair[1])[0],
                'Support': item[1],
                'Confidence': pair[2],
                'Lift': pair[3]
            })
        return pd.DataFrame(rows)

    def _create_visualizations(self, df, top_n=10):
        """Generate visualization plots"""
        Path("viz").mkdir(exist_ok=True)
        top_rules = df.nlargest(top_n, 'Lift')

        # Lift plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_rules, x='Left', y='Lift')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top {top_n} Rules by Lift')
        plt.tight_layout()
        plt.savefig('viz/lift_rules.png')
        plt.close()

        # Support vs Confidence
        plt.figure(figsize=(8, 6))
        plt.scatter(df['Support'], df['Confidence'], alpha=0.5)
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Support vs Confidence')
        plt.tight_layout()
        plt.savefig('viz/support_confidence.png')
        plt.close()

def main():
    try:
        # Initialize and run analysis
        analyzer = MarketAnalyzer()
        n_transactions = analyzer.load_data('Market_Basket_Optimisation.csv')
        results = analyzer.analyze()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        results.to_csv(f'results_{timestamp}.csv', index=False)

        # Display summary
        print(f"\nAnalyzed {n_transactions} transactions")
        print("\nTop 5 Rules by Lift:")
        print(results.nlargest(5, 'Lift')[['Left', 'Right', 'Lift']].to_string(index=False))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()