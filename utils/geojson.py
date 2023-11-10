import json


def print_feature_collection(fc):
    with open(fc) as f: data = json.load(f)
    features = data["features"]
    print(f"{data['type']} (length {len(features)})")
    print("\tFeature")
    print(f"\t\tgeometry: {features[0]['geometry']}")
    print(f"\t\ttype: {features[0]['type']}")
    properties = features[0]['properties']
    print(f"\t\tproperties: {json.dumps(list(properties.keys()), indent=20)}")
    first_prop_name = list(properties.keys())[0]
    print(f"\t\t\tproperty '{first_prop_name}': \t\t{json.dumps(properties[first_prop_name], indent=15)}")

def filter_feature_collection(fc, indices):
    pass



if __name__ == "__main__":
    file = "data/Abies_alba_1102.geojson"
    print_feature_collection(file)