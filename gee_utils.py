import ee
import pandas as pd

# ===========================================================
# INITIALIZATION
# ===========================================================

def initialize_gee(project_id):
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


# ===========================================================
# LOAD TRAINING ASSETS
# ===========================================================

def load_training_polygons(aoi_path, tea_path, forest_path, urban_path, class_field):

    aoi = ee.FeatureCollection(aoi_path)

    tea = ee.FeatureCollection(tea_path).map(
        lambda f: f.set(class_field, 1))

    forest = ee.FeatureCollection(forest_path).map(
        lambda f: f.set(class_field, 2))

    urban = ee.FeatureCollection(urban_path).map(
        lambda f: f.set(class_field, 3))

    polygons = tea.merge(forest).merge(urban)

    return aoi, polygons


# ===========================================================
# TRAIN TEST SPLIT
# ===========================================================

def train_test_split(polygons, train_fraction):

    polygons = polygons.randomColumn("rand")

    train = polygons.filter(ee.Filter.lt("rand", train_fraction))
    test = polygons.filter(ee.Filter.gte("rand", train_fraction))

    return train, test


# ===========================================================
# SENTINEL-2 COMPOSITE
# ===========================================================

def create_s2_composite(aoi, year, cloud_threshold=30):

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .median()
        .select(["B2", "B3", "B4", "B8", "B11", "B12"])
    )

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = s2.normalizedDifference(["B8", "B11"]).rename("NDWI")
    nbr  = s2.normalizedDifference(["B8", "B12"]).rename("NBR")

    evi = s2.expression(
        "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
        {
            "NIR": s2.select("B8"),
            "RED": s2.select("B4"),
            "BLUE": s2.select("B2")
        }
    ).rename("EVI")

    return s2.addBands([ndvi, ndwi, nbr, evi])


# ===========================================================
# BALANCED SAMPLING
# ===========================================================

def balanced_sampling(img, train_polys, class_field, scale=10, max_samples=5000):

    samples = img.sampleRegions(
        collection=train_polys,
        properties=[class_field],
        scale=scale,
        tileScale=4
    )

    c1 = samples.filter(ee.Filter.eq(class_field, 1)).limit(max_samples)
    c2 = samples.filter(ee.Filter.eq(class_field, 2)).limit(max_samples)
    c3 = samples.filter(ee.Filter.eq(class_field, 3)).limit(max_samples)

    return c1.merge(c2).merge(c3)


# ===========================================================
# TRAIN CLASSIFIER (MULTI-ALGORITHM)
# ===========================================================

def train_classifier(img, training_data, class_field, algorithm, params):

    bands = img.bandNames()

    if algorithm == "Random Forest":

        n_trees = params.get("n_trees", 200)
        variables_per_split = params.get("variables_per_split", None)
        min_leaf = params.get("min_leaf", 1)
        bag_fraction = params.get("bag_fraction", 0.7)
        max_nodes = params.get("max_nodes", None)

        classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=n_trees,
            variablesPerSplit=variables_per_split,
            minLeafPopulation=min_leaf,
            bagFraction=bag_fraction,
            maxNodes=max_nodes
        )

    elif algorithm == "Gradient Boost":

        n_trees = params.get("n_trees", 200)
        shrinkage = params.get("shrinkage", 0.1)
        sampling_rate = params.get("sampling_rate", 0.7)
        max_nodes = params.get("max_nodes", None)

        classifier = ee.Classifier.smileGradientTreeBoost(
            n_trees,
            shrinkage,
            sampling_rate,
            max_nodes,
            None
        )

    elif algorithm == "CART":
        classifier = ee.Classifier.smileCart()

    else:
        raise ValueError("Unsupported algorithm")

    trained = classifier.train(
        features=training_data,
        classProperty=class_field,
        inputProperties=bands
    )

    return trained

# ===========================================================
# MODEL EVALUATION
# ===========================================================

def evaluate_model(img, test_polys, classifier, class_field, sample_scale=10):

    test_pts = img.sampleRegions(
        collection=test_polys,
        properties=[class_field],
        scale=sample_scale,
        tileScale=4,
    ).limit(10000)

    classified_test = test_pts.classify(classifier)

    cm = classified_test.errorMatrix(class_field, "classification")

    # Confusion matrix array
    matrix = cm.getInfo()

    # Class labels (SAFE way in Python API)
    labels = cm.order().getInfo()

    results = {
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix
        },
        "overall_accuracy": cm.accuracy().getInfo(),
        "producer_accuracy": cm.producersAccuracy().getInfo(),
        "user_accuracy": cm.consumersAccuracy().getInfo(),
        "kappa": cm.kappa().getInfo(),
        "test_sample_count": test_pts.size().getInfo(),
    }

    return results

# ===========================================================
# CLASSIFICATION
# ===========================================================

def classify_image(img, classifier, aoi):
    return img.classify(classifier).clip(aoi)


# ===========================================================
# EXPORT
# ===========================================================

def export_to_drive(image, aoi, description, scale=10):

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        scale=scale,
        region=aoi.geometry(),
        fileFormat="GeoTIFF",
        maxPixels=1e13
    )

    task.start()
    return task