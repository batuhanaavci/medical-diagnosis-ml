# Breast Cancer Detection Using Logistic Regression and Random Forest

Contributors: Batuhan Avci, Kirill Zhukovsky

## Introduction
This project implements machine learning models to predict breast cancer diagnosis using the **Breast Cancer Wisconsin (Diagnostic) Dataset**. The dataset contains measurements from digitized images of fine needle aspirate (FNA) of breast mass, with features computed from the images describing characteristics of the cell nuclei.

## Dataset Description
- **Source**: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- **Size**: 569 samples
- **Classes**: Binary (Malignant/Benign)
- **Features**: 30 real-valued features computed from cell nuclei images
- **Feature Categories**:
  - Mean values
  - Standard error values
  - "Worst" values (mean of the three largest values)

### Feature Details
Each feature is computed for each cell nucleus, including:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

## Methodology

### Data Preprocessing
1. **Loading and Cleaning**: Removing unnecessary columns and handling missing values.
2. **Mapping Diagnosis Labels**: Converting labels (M=1, B=0) for binary classification.
3. **Feature Scaling**: Standardizing data using `StandardScaler` to normalize feature distributions.
4. **Feature Selection**: Removing highly correlated features to prevent redundancy.
5. **Splitting Data**: Using a **70-30 train-test split**, followed by **5-fold cross-validation** on the training set.

### Models Implemented

#### 1. Logistic Regression
- A widely used statistical model for binary classification.
- Uses **logistic loss function** for optimization.
- Provides interpretable results with feature importance.

#### 2. Random Forest Classifier
- An **ensemble learning method** that constructs multiple decision trees.
- Uses **Gini impurity** as a criterion for split quality.
- More robust to outliers and non-linear relationships.

## Implementation
The project is implemented in **Python** using the following libraries:
- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine learning algorithms
- `numpy` - Numerical computations
- `matplotlib` / `seaborn` - Data visualization

## Results
Both models performed well in classifying breast cancer samples. 

### Model Performance Comparison
| Model               | Test Accuracy | Cross-Validation Accuracy | False Positives | False Negatives |
|--------------------|--------------|--------------------------|----------------|----------------|
| **Logistic Regression** | **96.5%**       | 96.7%                    | 5              | **1**            |
| **Random Forest**      | 95.9%        | 95.5%                    | **3**          | 4              |

### Confusion Matrices
#### Logistic Regression
| Actual \ Predicted | Benign (0) | Malignant (1) |
|-------------------|------------|--------------|
| **Benign (0)**   | 103        | 5            |
| **Malignant (1)**| **1**      | 62           |

#### Random Forest
| Actual \ Predicted | Benign (0) | Malignant (1) |
|-------------------|------------|--------------|
| **Benign (0)**   | 105        | 3            |
| **Malignant (1)**| **4**      | 59           |

### Key Observations
- **Logistic Regression** had fewer **false negatives** (1 vs. 4), making it more reliable for detecting malignant cases.
- **Random Forest** had fewer **false positives** (3 vs. 5), meaning it reduced unnecessary alarms for benign cases.
- The close match between **cross-validation accuracy and test accuracy** suggests good generalization without overfitting.

## Requirements
Ensure you have the following dependencies installed:
- Python 3.x
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/breast-cancer-detection.git
   cd breast-cancer-detection
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook code/main.ipynb
   ```

## Future Improvements
- Implement additional machine learning algorithms (e.g., SVM, Neural Networks).
- Conduct hyperparameter tuning for improved accuracy.
- Apply **cross-validation with multiple metrics** to refine evaluation.
- Develop a **web interface** for real-time breast cancer diagnosis.

## Contributing
Contributions are welcome! If you wish to improve the project:
- Fork the repository.
- Create a feature branch.
- Submit a pull request.

## License
This project is licensed under the **MIT License** - see the LICENSE file for details.

## Acknowledgments
- **UCI Machine Learning Repository** for the Breast Cancer dataset.
- **scikit-learn** documentation and community.
- **Open source contributors** who maintain Python ML libraries.
