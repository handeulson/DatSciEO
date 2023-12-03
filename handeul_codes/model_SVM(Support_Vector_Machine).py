import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Define the list of tree species
tree_species = ["Acer_pseudoplatanus-", "Betula_pendula-", "Carpinus_betulus-", "Fagus_sylvatica-", "Fraxinus_excelsior-",
                "Picea_abies_1123-", "Pinus_sylvestris-", "Quercus_petraea-", "Quercus_robur-", "Sorbus_aucuparia-"]
#Each species has different files numbers. In addition, some files are missing within a species. 
#For example, "Acer_pseudoplatanus_0", "Acer_pseudoplatanus_2","Acer_pseudoplatanus_3", ...
#We can't apply automatic search for end-file with while loop. Thus, I speficied the end num string below
tree_species_endnum = [745, 560, 881, 2276, 847,
                3760, 3166, 1168, 829, 243]
# Initialize empty lists to store data and labels
data = []
labels = []
# Specify data folder direction
data_dir = '/Users/handerson/Desktop/Codes/DatSciEO-main/data/1123_delete_nan_samples_nanmean_B2/'

for num, species in enumerate(tree_species):
    i = 0
    while True:
        file_path = f"{data_dir}{species}{i}.npy"
        if os.path.exists(file_path):
            # Load data from the file
            file_data = np.load(file_path)
            
            # Append data and corresponding label to the lists
            data.append(file_data)
            labels.append(tree_species.index(species))
            i += 1
        else:
            # Break the loop if the file is not found, unless the index exceeds a certain limit
            if i > tree_species_endnum[num]:  # Set a limit to avoid infinite loops
                break
            i += 1  # Move on to the next index and continue searching

# Convert lists to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize an SVM classifier
clf = SVC()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
