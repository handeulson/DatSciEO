def determine_dimensions(collection: dict):
    """
    Determines the spatial dimensions of an input geojson.
    """
    w, h, b = None, None, None
    for feature_ in collection["features"]:
        property_names = list(feature_["properties"].keys())
        for prop_name_ in property_names:
            rows = feature_["properties"][prop_name_]
            if rows is not None:
                b = len(feature_["properties"])
                h = len(feature_["properties"][prop_name_])        # number of rows
                w = len(feature_["properties"][prop_name_][0])     # number of columns (values per row)
                break
        if h is not None: break
    return w, h, b