import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def pre_matt(file, scale=200):
    with open(file, 'rb') as fp:
        data = pickle.load(fp)
        fp.close()

    if not 'mask_keep' in data:
        mask_keep = np.arange(len(data['pts']))
    else:
        mask_keep = data['mask_keep']

    label = np.array(data['label'])[mask_keep] if 'label' in data else None
    name = np.array(data['name'])[mask_keep] if 'label' in data else None
    label = None
    pts = data['pts'][mask_keep, :] / 0.42

    # transform the pts so that it match my results.
    pts -= np.median(pts, axis=0)
    pts /= scale
    if 'side' in data and data['side'] == 0:
        pts[:, [0, 2]] *= -1
    if label is not None:
        label = label[:, np.newaxis]
        pts_out = np.hstack((pts, label))
    else:
        pts_out = pts
    output = dict()
    output['pts'] = pts_out
    output['color'] = data['fluo'][mask_keep, :] if 'fluo' in data else None
    output['label'] = label
    output['name'] = name
    output['f_name'] = file

    return output


def bin_matrix(data, bin_size):
    shape = (data.shape[0]//bin_size, bin_size, data.shape[1]//bin_size, bin_size)
    return data.reshape(shape).mean(axis=(1, 3))


#temp_f = '/Users/junangl/Documents/Princeton_projects/MultiColor_ID/fDLC_Neuron_ID/Data/Example/template.data'
#test_f = '/Users/junangl/Documents/Princeton_projects/MultiColor_ID/fDLC_Neuron_ID/Data/Example/test.data'
#temp = pre_matt(temp_f)  #load python dictinary from a pickle file.
#test = pre_matt(test_f)
# template worm contains pts, color(can be None) and label
#temp_pos = temp['pts']
#temp_label = temp['name']
#temp_color = temp['color']
# test worm contains pts, color(can be None)
#test_pos = test['pts']
#test_color = test['color']

save_csv = True
save_data = True
plot = False
z_plot = 1

# Define the file name
foldername = '/projects/LEIFER/Yagmur/data/multicolorworm_20230401_185401'
filename = foldername + '/brains.json'

# Load labeled data
with open(filename, 'r') as f:
    data = json.load(f)
    f.close()

# Load the laser info
f = open(foldername+"/moleculesSequence.txt","r")
ChannelName = f.read().splitlines()
nChannel = len(ChannelName)
f.close()

# Load the image data
nZ = len(np.loadtxt(foldername+"/piezoPosition.txt"))
nX = 2048
nY = 2048   
nFrames = nChannel*nZ
bin_size = 2
    
f = open(foldername+"/frames-"+str(nY)+"x"+str(nX)+".dat",'br') # 'b' tells the program to read it as binary    
im = np.fromfile(f, dtype=np.uint16, 
            count=nFrames*nX*nY).reshape(
            nZ,nChannel,
            nY,nX)                
f.close()

# Binning the image
im_binned = np.zeros((nZ,nChannel,nY//2,nX//2))
for i in range(nChannel):
    for j in range(nZ):
        im_binned[j,i,:,:] = bin_matrix(im[j,i,:,:], bin_size)

# Load the color data
raw_pos = np.array(data['coordZYX'])
color_array = np.zeros((len(raw_pos),nChannel))
for i in range(len(raw_pos)):
    color_array[i,:] = im_binned[raw_pos[i][0],:,raw_pos[i][2],raw_pos[i][1]]

if plot:
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2) 
    for i in range(nChannel):
        ax = fig.add_subplot(gs[i % 2, i // 2])  # Adjusting the indexing for 2x5 layout
        ax.imshow(im_binned[raw_pos[z_plot,0],i,:,:])
        ax.scatter(raw_pos[z_plot,2], raw_pos[z_plot,1], color='red')
        ax.set_ylabel('x' + str(i + 1))
        ax.set_xlabel('t')
    plt.show()
# print(data.keys())
# print(np.shape(frames))

fDLC_data = dict()
fDLC_data['pts'] = np.flip(raw_pos, axis=1) # Note that fDLC take data in the order of XYZ
fDLC_data['color'] = color_array
fDLC_data['label'] = None
fDLC_data['name'] = np.array(data['labels'][0])
fDLC_data['f_name'] = foldername + "/fDLC_train.data"

#Y: get a unique list of neuron labels from the data
unique_names = set(fDLC_data['name'])
#print(unique_names)

if save_data:
    with open(foldername + '/fDLC_train.data', 'wb') as f:
        pickle.dump(fDLC_data, f)
        f.close()


# Extract the "name" column from fDLC_data
num_entries = len(fDLC_data['name'])
#print("Number of entries in fDLC_data:", num_entries)

# Assuming fDLC_data['name'] is your 1-dimensional array
fDLC_name_2d = np.reshape(fDLC_data['name'], (-1, 1))

#combined_data = np.hstack((raw_pos, color_array))
combined_data = np.hstack((raw_pos, color_array,fDLC_name_2d))
combined_data_shape = combined_data.shape
print("Shape of combined_data:", combined_data_shape)

# Extract the "name" column from fDLC_data
num_entries = len(fDLC_data['name'])
print("Number of entries in fDLC_data:", num_entries)

#Create a panda df to store data

df = pd.DataFrame(combined_data, columns=['Z', 'Y', 'X', 'CC BPF', 'CC CyOFP', 'CC mCherry', 'CC mNeptune', 'Neuron Label-Human'])

# Add a new column 'Class' to the DataFrame with the first three letters from the 'Neuron Label-Human' column
df['Class'] = df['Neuron Label-Human'].str[:3]

## Enumeration of the label via giving class a unique random integer enumeration

# Create a copy of the DataFrame to avoid modifying the original
df_copy = df.copy()


# Define the list of all possible classes
possible_classes = [
    'ADA','ADE','ADF', 'ADL',
    'AFD', 'AIA', 'AIB', 'AIM', 
    'AIN', 'AIY','AIZ', 'ALA', 'AQR',
    'AS1', 'ASE', 'ASG','ASH', 'ASI',
    'ASJ', 'ASK','AUA','AVA',
    'AVB','AVD', 'AVE','AVF',
    'AVG', 'AVH','AVJ','AVK',
    'AVL', 'AWA', 'AWB','AWC','BAG',
    'CEP', 'DA1', 'DB1', 'DB2',
    'DD1', 'FLP', 'I1','I2', 'I3', 'I4',
    'I5', 'I6', 'IL1',
    'IL2', 'M1', 'M2',
    'M3', 'M4', 'M5', 'MCL',"MCR",'MI', 'NSM',
    'OLL',"OLQ",'RIA',
    'RIB','RIC','RID', 'RIF','RIG',
    'RIH', 'RIM','RIP', 'RIR', 'RIS', 'RIV', 'RIV',
    'RMD', 'RME',
    'RMF', 'RMG','RMH', 'SAA',
    'SAB', 'SIA','SIB', 'SMB','SMD','URA','URB','URX','URY',
     'VA1', 'VB1', 'VB2', 'VD1', 'VD2'
];

# Initialize an empty numpy array for one-hot encoding
one_hot_encoding = np.zeros((len(df), len(possible_classes)))

# Iterate over each entry in df['Class']
for i, cls in enumerate(df['Class']):
    # Find the index of cls in possible_classes
    if cls in possible_classes:
        index = possible_classes.index(cls)
        # Set the corresponding index in the one-hot encoding array to 1
        one_hot_encoding[i, index] = 1

# Add one-hot encoded columns to the DataFrame
for j, cls in enumerate(possible_classes):
    df[cls] = one_hot_encoding[:, j]


print(len(possible_classes))

###

# Initialize an empty array to store concatenated vectors
label_vectors = np.zeros((len(df), len(possible_classes)), dtype=np.int)

# Iterate over each entry in df['Class']
for i, cls in enumerate(df['Class']):
    # Find the index of cls in possible_classes
    if cls in possible_classes:
        index = possible_classes.index(cls)
        # Set the corresponding index in label_vectors to 1
        label_vectors[i, index] = 1

# Convert label_vectors to a list of arrays (for storing in DataFrame)
label_vector_list = [label_vectors[i, :] for i in range(len(df))]

# Add 'Label Vector' column to DataFrame df
df['Label Vector'] = label_vector_list


#Change data type of all numerics to float
df["Z"]=df["Z"].astype(float)

#Get exact Z positions with plane spacing
z_spacing=1.2 #microns
df["Z"]=df["Z"]*z_spacing

xy_spacing= 0.42 #microns per pixel

df["Y"]=df["Y"].astype(float)
df["Y"]=df["Y"]*xy_spacing

df["X"]=df["X"].astype(float)
df["X"]=df["X"]*xy_spacing
df["CC BPF"] = df["CC BPF"].astype(float)
df["CC CyOFP"] = df["CC CyOFP"].astype(float)
df["CC mCherry"] = df["CC mCherry"].astype(float)
df["CC mNeptune"] = df["CC mCherry"].astype(float)

### Normalized k-neighbor positional clustering algorithm : NUMBER OF CLUSTERS ARE EXTERNALLY SET!!!

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df_cut has been prepared with 'X', 'Y', 'Z' columns
X = df[['X', 'Y', 'Z']].values

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose k for k-nearest neighbors
k = 10

# Compute k-nearest neighbors
nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

# Apply clustering (KMeans clustering on distances can be one approach)
kmeans = KMeans(n_clusters=10, random_state=0).fit(distances)

# Assign cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to dataframe
df['Cluster'] = cluster_labels

# Visualize clusters (example 3D plot)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with cluster coloring
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    ax.scatter(cluster_data['X'], cluster_data['Y'], cluster_data['Z'], label=f'Cluster {cluster}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('3D Visualization of Positional Clusters')
plt.show()

#One-hot encoding for positional cluster
import numpy as np

# Assuming df contains your entire dataset, including 'Cluster' column
cluster_labels = df['Cluster'].values

# Get unique cluster labels
unique_clusters = np.unique(cluster_labels)

# Initialize an empty array to store concatenated vectors
cluster_vectors = np.zeros((len(df), len(unique_clusters)), dtype=np.int)

# Iterate over each entry in cluster_labels
for i, cluster in enumerate(cluster_labels):
    # Find the index of cluster in unique_clusters
    index = np.where(unique_clusters == cluster)[0][0]
    # Set the corresponding index in cluster_vectors to 1
    cluster_vectors[i, index] = 1

# Convert cluster_vectors to a list of arrays (for storing in DataFrame)
cluster_vector_list = [cluster_vectors[i, :] for i in range(len(df))]

# Add 'Cluster Vector' column to DataFrame df
df['Cluster Vector'] = cluster_vector_list

#Enumerate L-R and V-D similarity for available classes via one-hot encoding

# Create one-hot encoded columns for D, L, R, V

# Initialize an array filled with 0.5
ori_vectors = np.full((len(df), 4), 0.5, dtype=np.float)

# Iterate over each entry in df['Neuron Label-Human']
for i, label in enumerate(df['Neuron Label-Human']):
    # Check for 'D' in the last two characters
    if 'D' in label[-2:]:
        ori_vectors[i, 0] = 0.8
    # Check for 'L' in the last two characters
    if 'L' in label[-2:]:
        ori_vectors[i, 1] = 0.1
    # Check for 'R' in the last two characters
    if 'R' in label[-2:]:
        ori_vectors[i, 2] = 0.2
    # Check for 'V' in the last two characters
    if 'V' in label[-2:]:
        ori_vectors[i, 3] = 0.9

# Convert ori_vectors to a list of arrays (for storing in DataFrame)
ori_vector_list = [ori_vectors[i, :] for i in range(len(df))]

# Add 'Ori Vector' column to DataFrame df
df['Ori Vector'] = ori_vector_list


#Save CSV for visualization
if save_csv:
	# Save the DataFrame to a CSV file
	df.to_csv(foldername+'/combined_data.csv', index=False)


#Get necessary input features for analysis
# List of columns to include in df_cut
columns_to_include = ["Cluster",'CC BPF', 'CC CyOFP', 'CC mCherry', 'CC mNeptune', 'X', 'Y', 'Z',
                     'Label Vector', "Ori Vector","Cluster Vector","Class",'Neuron Label-Human']

# Create df_cut with selected columns from df

df_cut = df.loc[:, columns_to_include]

df_cut.to_csv(foldername+'/cut_data.csv', index=False)

#IMPLEMENT DECISION TREE

# Remove rows with all zeros in 'Label Vector' column since these are the unlabeled neurons that wont benefit us with training
df_fin = df_cut = df_cut[df_cut['Class'] != '']
#df_fin.to_csv(foldername+'/ML_data.csv', index=False)



# Concatenate Label Vector and Ori Vector to get the ID vector
# Adjusted approach to concatenate Label Vector and Ori Vector
def concatenate_vectors(row):
    label_vector = row['Label Vector']
    ori_vector = row['Ori Vector']
    # Concatenate label_vector and ori_vector element-wise
    return [label + ori for label, ori in zip(label_vector, ori_vector)]

# Apply the function row-wise to create 'Appended Vector' column
import numpy as np


# Initialize an empty list to store ID vectors
ID_vectors = []

# Iterate over each row in df_fin
for index, row in df_fin.iterrows():
    # Extract Label Vector and Ori Vector from the current row
    label_vector = row['Label Vector']
    ori_vector = row['Ori Vector']
    
    # Combine Label Vector and Ori Vector into ID_vector
    ID_vector = np.concatenate([label_vector, ori_vector])
    
    # Append ID_vector to the list
    ID_vectors.append(ID_vector)

# Convert ID_vectors list into numpy array
ID_vectors = np.array(ID_vectors)

# Add 'ID' column to df_fin
df_fin['ID'] = ID_vectors.tolist()  # Convert numpy array to list for pandas compatibility

# Display the updated DataFrame
#print(df_fin.head())

df_fin.to_csv(foldername+'/ML_data.csv', index=False)

###

#ML training- regression tree classifier with MSE minimization

#!!!!!YAĞMUR: Burada kullandığım  tree architectureından emin değilim, multiclass classification için one hot encoding yapılacak bilecek en simple lgoritma buydu. MSE hesaplarken vector label vs orientation label a verilen weighting ok mi: 1 vs 0.1 norm farkı? istediğimiz cluster sayısı?

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Function to normalize ID vector with sum of elements up to the last 4 elements equal to 1
def normalize_id_vector(id_vector):
    # Sum of elements up to the last 4 elements
    sum_up_to_last_4 = np.sum(id_vector[:-4])
    
    # Normalize elements up to the last 4 elements
    if sum_up_to_last_4 != 0:
        normalized_vector = id_vector[:-4] / sum_up_to_last_4
    else:
        normalized_vector = id_vector[:-4]  # If sum_up_to_last_4 is 0, leave unchanged
    
    # Last 4 elements remain unchanged
    normalized_vector = np.append(normalized_vector, id_vector[-4:])
    
    return normalized_vector

# Select relevant columns from df_fin
columns_to_select = ['Cluster Vector', 'CC BPF', 'CC CyOFP', 'CC mCherry', 'CC mNeptune', 'ID']
df_selected = df_fin[columns_to_select].copy()  

# Normalize CC columns using StandardScaler
scaler = StandardScaler()
df_selected.loc[:, ['CC BPF', 'CC CyOFP', 'CC mCherry', 'CC mNeptune']] = scaler.fit_transform(df_selected[['CC BPF', 'CC CyOFP', 'CC mCherry', 'CC mNeptune']])

# Split data into features (X) and target (y)
X = np.array(df_selected['Cluster Vector'].tolist())
y = np.array(df_selected['ID'].tolist())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Ridge Regression model
ridge = Ridge(alpha=1.0, random_state=42)  # Adjust alpha as needed for regularization strength
ridge.fit(X_train, y_train)

# Predict on test set
y_pred = ridge.predict(X_test)

# Normalize predicted ID vectors
normalized_y_pred = np.array([normalize_id_vector(id_vec) for id_vec in y_pred])

# Compute Mean Squared Error (MSE) as the cost function
mse = mean_squared_error(y_test, normalized_y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Print or use normalized_y_pred as needed
print(normalized_y_pred)

#Visualize the norm error squared for each test point for each ID vector point

# Compute squared differences
differences_squared = (normalized_y_pred - y_test) ** 2
# Plotting the differences for each test point seperately
fig, ax = plt.subplots()
for i in range(len(normalized_y_pred)):
    ax.plot(range(len(normalized_y_pred[i])), differences_squared[i], label=f'Test Point {i+1}')
ax.set_xlabel('Index of ID Vector')
ax.set_ylabel('Squared Difference')
ax.legend()
plt.title('Squared Differences between Normalized Predictions and Actual ID Vectors')
plt.show()

#Plot accumulated MSE for a random group of test points for each ID vector label

test_size=round(len(normalized_y_pred) * 0.3)
print(len(normalized_y_pred))

def plot_squared_differences(test_points, normalized_y_pred, y_test):
    plt.figure(figsize=(10, 6))
    for test_point in test_points:
        squared_diffs = (normalized_y_pred[test_point - 1] - y_test[test_point - 1]) ** 2
        plt.plot(range(len(squared_diffs)), squared_diffs, marker='o', linestyle='-', label=f'Test Point {test_point}')
    plt.xlabel('Index of ID Vector')
    plt.ylabel('Squared Difference')
    plt.title(f'Squared Differences for Randomly Selected Test Points')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_random_test_points(normalized_y_pred, y_test, num_points=30):
    all_test_points = list(range(1, len(normalized_y_pred) + 1))
    selected_test_points = np.random.choice(all_test_points, size=num_points, replace=False)
    plot_squared_differences(selected_test_points, normalized_y_pred, y_test)

plot_random_test_points(normalized_y_pred, y_test, num_points=test_size)


