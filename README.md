# Credit Card Customer Segmentation using KMeans

This project performs **customer segmentation** on credit card data using **unsupervised learning (KMeans clustering)**. It includes data preprocessing, scaling, dimensionality reduction, clustering, evaluation, and visualization.

---

## 📌 Project Objective

To group credit card customers into meaningful segments based on their numerical attributes such as credit limit, balance, transactions, etc., for business insights and targeted marketing strategies.

---

## 🛠️ Tools & Libraries

* **Python**
* **Pandas** – data manipulation
* **NumPy** – numerical operations
* **Matplotlib** – plotting
* **Seaborn** – statistical visualization
* **Scikit-learn** – StandardScaler, PCA, KMeans, silhouette_score

---

## 📂 Dataset

**Input File:** `Credit Card Customer Data.csv`

### Key Columns

* `Customer Key` (dropped)
* `Sl_No` (dropped)
* Other numerical columns like:

  * `Credit_Limit`
  * `Balance`
  * `Total_Transactions`
  * `Avg_Transaction_Amount`

---

## 🔄 Project Workflow

### 1. Data Preprocessing

* Dropped identifier columns (`Sl_No`, `Customer Key`)
* Selected numerical features
* Standardized data using **StandardScaler**

---

### 2. KMeans Clustering

* Explored **k = 2 to 6 clusters**
* Calculated **Silhouette Score** for each k
* Selected **best k** based on highest silhouette score
* Assigned cluster labels to original data

---

### 3. Dimensionality Reduction

* Applied **PCA** to reduce features to 2D for visualization
* Plotted clusters in 2D scatter plot

---

### 4. Cluster Visualization

* **Scatter plot** of PCA components by cluster
* **Pairplot** for all numerical features colored by cluster
* **Countplot** for number of customers per cluster
* **Boxplot** for Credit Limit distribution by cluster (if available)
* **Heatmap** of cluster centers for numeric features

---

### 5. Output

* **Clustered dataset saved as:** `credit_card_clusters.csv`
* Cluster labels added for each customer

---

## 📈 Visual Outputs

* 2D scatter plot of clusters
* Pairplot of numerical features by cluster
* Countplot of customers per cluster
* Boxplot of Credit Limit by cluster
* Cluster centers heatmap

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Place `Credit Card Customer Data.csv` in the project directory
4. Run the Python script

---

## 📌 Use Cases

* Customer segmentation for targeted marketing
* Credit risk assessment
* Personalized offers for high-value customers
* Unsupervised learning portfolio project

---

## 👤 Author

**Khubaib**
Data Analyst | Data Scientist | Machine Learning Enthusiast

---

## ⭐ Notes

* Standardization is essential before KMeans
* Silhouette Score helps determine optimal number of clusters
* PCA is used only for visualization, not for clustering itself
* Segments can guide business strategy and promotions

---

If you like this project, feel free to ⭐ the repository and extend it with additional clustering techniques like Hierarchical Clustering or DBSCAN.
