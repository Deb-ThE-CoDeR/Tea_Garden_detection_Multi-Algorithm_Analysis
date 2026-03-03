import streamlit as st
import folium
from streamlit_folium import st_folium
import gee_utils as gu
import ee
import pandas as pd

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(layout="wide")
st.title("🌿 Tea Garden Classification (GEE)")

if "models" not in st.session_state:
    st.session_state.models = {}

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("🧠 Algorithm Selection")

algorithm_choice = st.sidebar.selectbox(
    "Select Algorithm",
    ["Random Forest", "Gradient Boost", "CART"],
    index=0
)

# ==========================================================
# MODEL PARAMETERS
# ==========================================================

st.sidebar.header("⚙️ Model Parameters")
params = {}

# ----------------------------------------------------------
# Random Forest Parameters
# ----------------------------------------------------------
if algorithm_choice == "Random Forest":

    params["n_trees"] = st.sidebar.slider(
        "Number of Trees", 50, 500, 200)

    vars_split = st.sidebar.number_input(
        "Variables per Split (0 = auto)", value=0)

    params["variables_per_split"] = None if vars_split == 0 else vars_split

    params["min_leaf"] = st.sidebar.slider(
        "Min Leaf Population", 1, 20, 1)

    params["bag_fraction"] = st.sidebar.slider(
        "Bag Fraction", 0.1, 1.0, 0.7)

    max_nodes = st.sidebar.number_input(
        "Max Nodes (0 = unlimited)", value=0)

    params["max_nodes"] = None if max_nodes == 0 else max_nodes


# ----------------------------------------------------------
# Gradient Boost Parameters
# ----------------------------------------------------------
elif algorithm_choice == "Gradient Boost":

    params["n_trees"] = st.sidebar.slider(
        "Number of Trees", 50, 70, 50)

# ----------------------------------------------------------
# CART Parameters (No Hyperparameters in GEE)
# ----------------------------------------------------------
elif algorithm_choice == "CART":

    st.sidebar.info(
        "CART in Google Earth Engine uses default settings.\n"
        "No tunable hyperparameters available."
    )

    params = {}  # Explicitly empty


# ==========================================================
# DATA PARAMETERS
# ==========================================================

st.sidebar.header("📊 Data Parameters")

PROJECT_ID = st.sidebar.text_input(
    "GEE Project ID", "tea-garden-detection")

YEAR = st.sidebar.number_input(
    "Year", 2018, 2025, 2021)

TRAIN_FRACTION = st.sidebar.slider(
    "Train Fraction", 0.5, 0.9, 0.7)

SCALE = st.sidebar.selectbox(
    "Sampling Scale (m)", [10, 20, 30], index=0)

run_button = st.sidebar.button("🚀 Run Classification")

# ==========================================================
# FIXED ASSET PATHS
# ==========================================================

AOI_PATH = "users/debajyotitubai/AOI_asset"
TEA_PATH = "users/debajyotitubai/Tea_asset"
FOREST_PATH = "users/debajyotitubai/Forest_asset"
URBAN_PATH = "users/debajyotitubai/Urban_asset"
CLASS_FIELD = "landmark"

# ==========================================================
# WORKFLOW
# ==========================================================

if run_button:

    st.session_state.models = {}

    progress = st.progress(0)
    step = 0
    total_steps = 6

    # 1️⃣ Initialize GEE
    gu.initialize_gee(PROJECT_ID)
    step += 1; progress.progress(step / total_steps)

    # 2️⃣ Load Data
    aoi, polygons = gu.load_training_polygons(
        AOI_PATH, TEA_PATH, FOREST_PATH, URBAN_PATH, CLASS_FIELD
    )
    step += 1; progress.progress(step / total_steps)

    # 3️⃣ Train-Test Split
    train_polys, test_polys = gu.train_test_split(
        polygons, TRAIN_FRACTION)
    step += 1; progress.progress(step / total_steps)

    # 4️⃣ Create Composite
    img = gu.create_s2_composite(aoi, YEAR)
    step += 1; progress.progress(step / total_steps)

    # 5️⃣ Balanced Sampling
    training_data = gu.balanced_sampling(
        img, train_polys, CLASS_FIELD, SCALE
    )
    step += 1; progress.progress(step / total_steps)

    # 6️⃣ Train + Evaluate
    classifier = gu.train_classifier(
        img,
        training_data,
        CLASS_FIELD,
        algorithm_choice,
        params
    )

    results = gu.evaluate_model(
        img,
        test_polys,
        classifier,
        CLASS_FIELD,
        SCALE
    )

    classified = gu.classify_image(
        img, classifier, aoi)

    st.session_state.models = {
        algorithm_choice: {
            "results": results,
            "classified": classified,
            "aoi": aoi
        }
    }

    step += 1; progress.progress(step / total_steps)

    st.success("✅ Classification Completed")


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

for algo, model_data in st.session_state.models.items():

    st.header(f"📊 {algo}")

    results = model_data["results"]
    classified = model_data["classified"]
    aoi = model_data["aoi"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Overall Accuracy",
        round(results.get("overall_accuracy", 0), 3)
    )

    col2.metric(
        "Kappa",
        round(results.get("kappa", 0), 3)
    )

    col3.metric(
        "Test Samples",
        results.get("test_sample_count", 0)
        )
    cm = results["confusion_matrix"]

    st.write("### Confusion Matrix")

    labels = cm["labels"]
    matrix = cm["matrix"]

    df = pd.DataFrame(
        matrix,
        index=[f"Actual_{l}" for l in labels],
        columns=[f"Pred_{l}" for l in labels]
    )

    st.dataframe(df)
    # ------------------------------------------------------
    # Map Visualization
    # ------------------------------------------------------

    vis = {
        "min": 1,
        "max": 3,
        "palette": ["#14ED14", "#006400", "#FF0000"]
    }

    center = (
        aoi.geometry()
        .centroid()
        .coordinates()
        .getInfo()[::-1]
    )

    m = folium.Map(location=center, zoom_start=11)

    map_id = classified.getMapId(vis)

    folium.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        overlay=True
    ).add_to(m)

    folium.LayerControl().add_to(m)
    # ------------------------------------------------------
    # Legend
    # ------------------------------------------------------

    legend_html = """
    <div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    width: 200px;
    background-color: white;
    color: black;
    border: 2px solid black;
    border-radius: 6px;
    padding: 12px;
    font-size: 14px;
    box-shadow: 3px 3px 8px rgba(0,0,0,0.3);
    z-index: 9999;
    ">

    <div style="display:flex; align-items:center; margin-bottom:6px;">
        <div style="width:18px;height:18px;background:#14ED14;margin-right:8px;"></div>
        <span>Tea</span>
    </div>

    <div style="display:flex; align-items:center; margin-bottom:6px;">
        <div style="width:18px;height:18px;background:#006400;margin-right:8px;"></div>
        <span>Forest</span>
    </div>

    <div style="display:flex; align-items:center;">
        <div style="width:18px;height:18px;background:#FF0000;margin-right:8px;"></div>
        <span>Urban</span>
    </div>

    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=1100, height=600)

    # ------------------------------------------------------
    # Export
    # ------------------------------------------------------

    if st.button(f"📤 Export {algo}"):

        gu.export_to_drive(
            classified,
            aoi,
            description=f"{algo.replace(' ', '_')}_{YEAR}",
            scale=SCALE
        )

        st.success(
            "Export started. Check Google Earth Engine Tasks."
        ) 
