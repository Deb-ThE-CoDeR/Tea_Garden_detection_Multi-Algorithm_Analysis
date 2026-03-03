A machine learning-based land cover classification system for detecting Tea, Forest, and Urban areas using Google Earth Engine (GEE) and Sentinel-2 imagery.

The project supports both:
🖥️ CLI-based workflow
🌐 Streamlit GUI application

🚀 Features

Sentinel-2 image composite generation
Balanced sampling for training
Train/Test split

#Multiple ML algorithms:

Random Forest
Gradient Boosting
CART

Model evaluation:

Overall Accuracy
Kappa coefficient
Confusion Matrix

Interactive map visualization (GUI)

Export classified raster to Google Drive

Modular and reusable architecture

📁 Project Structure
.
├── app.py                # Streamlit GUI
├── main.py               # CLI execution script
├── gee_utils.py          # Core GEE processing functions
├── requirements.txt
└── README.md
🛰️ Data Requirements

You must upload the following assets to Google Earth Engine:


AOI (Area of Interest)

Tea polygons
Forest polygons
Urban polygons

Each training feature must contain:

landmark = 1 (Tea)
landmark = 2 (Forest)
landmark = 3 (Urban)

⚙️ Installation

Create a virtual environment:
python -m venv venv

Activate it:
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Authenticate Earth Engine:

earthengine authenticate
🖥️ Running the CLI Version

Basic run:
python main.py

With parameters:
python main.py --algorithm "Random Forest" --year 2022 --trees 300 --export

🌐 Running the Streamlit GUI
streamlit run app.py

Features available in GUI:

Algorithm selection
Hyperparameter tuning
Confusion matrix display
Interactive map with legend
One-click export

🧠 Algorithms Used

Random Forest	Ensemble bagging method
Gradient Boost	Sequential boosting trees
CART	Decision tree classifier

📊 Evaluation Metrics

Overall Accuracy
Kappa Statistic
Confusion Matrix

Evaluation is performed using stratified sampling to avoid excessive GEE computation load.

📤 Export

Classified rasters are exported to:
Google Drive → Earth Engine Tasks

🛠️ Tech Stack

Google Earth Engine
Python
Streamlit

Folium

Pandas
