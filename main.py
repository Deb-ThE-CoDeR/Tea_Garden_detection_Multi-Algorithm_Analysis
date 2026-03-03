import gee_utils as gu

# ===========================================================
# USER PARAMETERS
# ===========================================================

AOI_PATH = "users/debajyotitubai/AOI_asset"
TEA_PATH = "users/debajyotitubai/Tea_asset"
FOREST_PATH = "users/debajyotitubai/Forest_asset"
URBAN_PATH = "users/debajyotitubai/Urban_asset"

CLASS_FIELD = "landmark"
TRAIN_FRACTION = 0.7
YEAR = 2021
TREES = 200
SCALE = 10
PROJECT_ID = "tea-garden-detection"

ALGORITHMS = ["Random Forest", "Gradient Boost", "CART"]

# ===========================================================
# WORKFLOW
# ===========================================================

gu.initialize_gee(PROJECT_ID)

aoi, polygons = gu.load_training_polygons(
    AOI_PATH, TEA_PATH, FOREST_PATH, URBAN_PATH, CLASS_FIELD
)

train_polys, test_polys = gu.train_test_split(polygons, TRAIN_FRACTION)

img = gu.create_s2_composite(aoi, YEAR)

training_data = gu.balanced_sampling(
    img, train_polys, CLASS_FIELD, SCALE
)

for algo in ALGORITHMS:

    print(f"\n===== {algo} =====")

    params = {}

    if algo in ["Random Forest", "Gradient Boost"]:
        params["n_trees"] = TREES

    classifier = gu.train_classifier(
    img, training_data, CLASS_FIELD, algo, params
    )

    results = gu.evaluate_model(
        img, test_polys, classifier, CLASS_FIELD, SCALE
    )

    print("Overall Accuracy:", results["overall_accuracy"])
    print("Kappa:", results["kappa"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    classified = gu.classify_image(img, classifier, aoi)

    gu.export_to_drive(
        classified,
        aoi,
        description=f"{algo.replace(' ', '_')}_Classification_{YEAR}",
        scale=SCALE
    )

print("\nAll exports started successfully.")
