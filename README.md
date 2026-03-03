This project uses machine learning to classify land cover—specifically Tea, Forest, and Urban areas—by combining Google Earth Engine (GEE) with Sentinel-2 satellite images.

You can run it two ways: straight from the command line, or through a simple Streamlit web app.

What’s inside:

- Makes custom composites from Sentinel-2 images
- Picks balanced samples for training
- Splits your data into training and testing sets

Supports several ML algorithms:
- Random Forest
- Gradient Boosting
- CART (Classification and Regression Trees)

For evaluation, you get:
- Overall accuracy
- Kappa coefficient
- Confusion matrix

With the GUI, you can explore results on an interactive map. When you're done, export the classified raster straight to your Google Drive.

The code’s organized and easy to expand. Here’s the basic layout:

.
├── app.py            # Streamlit GUI
├── main.py           # CLI script
├── gee_utils.py      # GEE-related functions
├── requirements.txt
└── README.md

Data you’ll need:

Upload these to your GEE assets:
- AOI (your Area of Interest)
- Polygons for Tea, Forest, and Urban areas

Each training polygon must include:
- landmark = 1 (Tea)
- landmark = 2 (Forest)
- landmark = 3 (Urban)

How to set it up:

1. Make a virtual environment:
   python -m venv venv

2. Activate it:
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Authenticate Earth Engine:
   earthengine authenticate

To run from the command line:
- Basic: python main.py
- With custom settings:
  python main.py --algorithm "Random Forest" --year 2022 --trees 300 --export

To launch the Streamlit app:
- streamlit run app.py

The GUI lets you:
- Pick your algorithm
- Tweak hyperparameters
- See the confusion matrix
- Explore results on an interactive map (with a legend)
- Export with one click

Algorithms in the mix:
- Random Forest (bagging method)
- Gradient Boost (sequential boosting)
- CART (decision tree)

Evaluation metrics:
- Overall accuracy
- Kappa statistic
- Confusion matrix

Stratified sampling keeps GEE from getting bogged down with too much data.

Export results directly to your Google Drive under Earth Engine Tasks.

Tech stack:
- Google Earth Engine
- Python
- Streamlit
- Folium
- Pandas
