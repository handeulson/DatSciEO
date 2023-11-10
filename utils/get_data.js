/***        SETTINGS        ***/

// specify species names to filter for, a dataset identifier for this run
var dataset_identifier = "1102" // as per current date
var species_names = ["Abies alba", "Picea abies"];
var colors = ["#f3ff3c", "#ff0022"];
var leaf_type = 111; // Code 111 means closed needleleaf forest and 114 means closed broadleaf forest

// specify which bands to export
// index 1: summer, index 2: autumn, no index: spring
var Bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
'B2_1', 'B3_1', 'B4_1', 'B5_1', 'B6_1', 'B7_1', 'B8_1', 'B8A_1', 'B11_1', 'B12_1',
'B2_2', 'B3_2', 'B4_2', 'B5_2', 'B6_2', 'B7_2', 'B8_2', 'B8A_2', 'B11_2', 'B12_2']

// specify the size of the clipped patch around the "trees"
// resulting patch will be 2*patch_size+1 in edge length (e.g. patch_size=2 -> patches of 5x5)
var patch_size = 2;



/***Use forest mask to filter samples***/
// Use COPERNICUS Land cover product as forest mask
var forest_mask = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")
.select('discrete_classification');
Map.addLayer(forest_mask, null, 'forest mask')

var forest = forest_mask.reduceRegions({
  collection: samples,
  reducer: ee.Reducer.first(),
  scale: 10,
  crs: 'EPSG:3035'
});
print("first", forest.first())

/*
// Take "Abies alba" species as example
// Code 111 means closed needleleaf forest and 114 means closed broadleaf forest
var species = forest.filterMetadata('SPECIES NAME','equals', species_name);
print("after species filter", species);
var species = species.filterMetadata('first','equals', leaf_type);
print("after leaf type", species);
*/

/***Select the roi of Germany***/
Map.setCenter(16.35, 48.83, 4);
var dataset = Boundary.select("country_co");
var roi_germany = dataset.filter(ee.Filter.eq('country_co','GM'))
Map.addLayer(roi_germany, null, 'roi of Germany')


/***Remove clouds***/
function maskS2clouds(image) {
  var qa = image.select('QA60');
 
  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
 
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
 
  return image.updateMask(mask).divide(10000);
}


/***Select Sentinel-2 data***/
var s2_spring = ee.ImageCollection('COPERNICUS/S2_SR')
      .filterBounds(roi_germany)
      .filterDate('2022-03-01', '2022-05-31')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
      .map(function(image){
            return image.clip(roi_germany)
        })
      .map(maskS2clouds)
      .median();
      
var s2_summer = ee.ImageCollection('COPERNICUS/S2_SR')
      .filterBounds(roi_germany)
      .filterDate('2022-06-01', '2022-08-31')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
      .map(function(image){
            return image.clip(roi_germany)
        })
      .map(maskS2clouds)
      .median();
      
var s2_autumn = ee.ImageCollection('COPERNICUS/S2_SR')
      .filterBounds(roi_germany)
      .filterDate('2022-09-01', '2022-11-30')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
      .map(function(image){
            return image.clip(roi_germany)
        })
      .map(maskS2clouds)
      .median();
      
// Stack all bands
var s2 = s2_spring.addBands(s2_summer).addBands(s2_autumn);
print("s2", s2)
var vizParams = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 0.5,
  gamma: [0.95, 1.1, 1]
};
Map.addLayer(s2, vizParams, "s2")

// select bands from sentinel2 composite
var s2_select = s2.select(Bands)


/***Data export***/
// Set the patch size to 5 pixels
var kernel = ee.Kernel.rectangle(patch_size, patch_size)
var neighborImg = s2_select.neighborhoodToArray(kernel)
print("neighborImg", neighborImg)

//var day = String(new Date().toJSON().slice(0,10).replace(/-/g,'-'));
//print("day:", day)
//var day = "1102" // as per current date

var species_name;
for (var i = 0; i < species_names.length; i++) {
  species_name = species_names[i];
  print("current species name:", species_name)
  // Take "Abies alba" species as example
  // Code 111 means closed needleleaf forest and 114 means closed broadleaf forest
  var species = forest.filterMetadata('SPECIES NAME','equals', species_name);
  print("after species filter:", species.size())
  var species = species.filterMetadata('first','equals', leaf_type);
  print("after leaf type:", species.size());
  
  /***Visualize species samples***/
  //Map.addLayer(species, {color: '#f3ff3c'}, species_name);
  Map.addLayer(species, {color: colors[i], size: 1}, species_name);
  
  // Clip the patch
  //var species = "Abies_alba";
  var samples = neighborImg.reduceRegions({
    collection: species,
    reducer: ee.Reducer.first(),
    scale: 10,
    crs: 'EPSG:3035'
  });
  
  var folder_name = "DatSciEOData";
  // Export to Google Drive
  var filename = species_name.replace(" ", "_") + '_' + dataset_identifier;
  Export.table.toDrive({
      collection : samples,
      description : filename,
      //fileNamePrefix : species_name + '_' + day,
      fileNamePrefix : filename,
      fileFormat : 'GeoJSON',
      selectors : Bands,
      folder : folder_name
    });
}


// export the current dataset info
var info = ee.Dictionary({
    "species_name": species_name,
    "leaf_type": leaf_type,
    "bands": Bands,
    "patch_size": patch_size
});

var filename_info = "_info_" + dataset_identifier;
// Export to Google Drive
Export.table.toDrive({
    collection : ee.FeatureCollection(ee.Feature(null, info)),
    description : filename_info,
    //fileNamePrefix : species_name + '_' + day,
    fileNamePrefix : filename_info,
    fileFormat : 'GeoJSON',
    selectors : Bands,
    folder : folder_name
  });