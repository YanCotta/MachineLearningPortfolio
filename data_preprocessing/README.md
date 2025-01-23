# DataPreprocessingToolsForMLModels
A simple script demonstrating common data preprocessing tools:
- Reading a dataset
- Handling missing values
- Encoding categorical features
- Splitting into training/testing sets
- Scaling numeric features

## Requirements
- Python 3.7+
- NumPy
- pandas
- matplotlib (optional)
- scikit-learn

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your dataset in the script’s directory (adjust any file paths if needed).
2. Update the indexing for features/target according to your dataset.
3. Run the script

## Notes
- Adjust column indices in SimpleImputer and ColumnTransformer to suit your data.
- Set `test_size` and `random_state` as needed.

## License
See the root LICENSE for project licensing details.

## Testing
Run tests for this sub-project:
```
python -m unittest discover -s tests
```
