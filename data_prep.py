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


# Initialize a dictionary to store mappings from neuron class to unique integers
class_dic = {}

# Iterate over each element in the 'Class' column
for index, label_prefix in enumerate(df_copy['Class']):
    # Check if the label prefix is already in the dictionary
    if label_prefix not in class_dic:
        # If it's not, generate a unique random integer and assign it to the label prefix
        random_int = np.random.randint(1, 1000)  # Adjust the range as needed
        # Ensure the generated integer is unique
        while random_int in class_dic.values():
            random_int = np.random.randint(1, 1000)
        # Store the mapping in the dictionary
        class_dic[label_prefix] = random_int
    else:
        # If the label prefix is in the dictionary, copy the value for that key
        random_int = class_dic[label_prefix]
    
    # Assign the integer to the label prefix in the DataFrame
    df_copy.at[index, 'Neuron Label-Integer'] = random_int

# Print the DataFrame to verify the changes
#print(df_copy["Neuron Label-Integer"])

# Add the 'Neuron Label-Integer' column to the original DataFrame df
df['Neuron Label-Integer'] = df_copy['Neuron Label-Integer']


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


#Enumerate L-R and V-D similarity for available classes via one-hot encoding

# Create one-hot encoded columns for D, L, R, V
df['Is_D'] = df['Neuron Label-Human'].apply(lambda x: 1 if len(x) >= 4 and x[3] == 'D' else 0)
df['Is_L'] = df['Neuron Label-Human'].apply(lambda x: 1 if len(x) >= 4 and x[3] == 'L' else 0)
df['Is_R'] = df['Neuron Label-Human'].apply(lambda x: 1 if len(x) >= 4 and x[3] == 'R' else 0)
df['Is_V'] = df['Neuron Label-Human'].apply(lambda x: 1 if len(x) >= 4 and x[3] == 'V' else 0)

# Create pair-wise relationships
df['Is_LR'] = df.apply(lambda row: 1 if row['Is_L'] == 1 or row['Is_R'] == 1 else 0, axis=1)
df['Is_DV'] = df.apply(lambda row: 1 if row['Is_D'] == 1 or row['Is_V'] == 1 else 0, axis=1)



#Implement K-nearest neighbor algorithm: Considering only positions

from sklearn.neighbors import KNeighborsClassifier
# Extract positional data
positional_data = df[['X', 'Y', 'Z']]
# Specify the number of clusters (k)
k = 10
# Initialize the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=k)
# Fit the model to your positional data
knn.fit(positional_data, df['Neuron Label-Integer'])
# Predict cluster labels for the positional data
cluster_labels = knn.predict(positional_data)
# Add cluster labels to the DataFrame
df['Cluster'] = cluster_labels
# Print the DataFrame to verify the changes
#print(df)
# Get unique cluster values
unique_clusters = sorted(df['Cluster'].unique())
# Print the number of unique clusters
num_unique_clusters = len(unique_clusters)
print("Number of unique clusters:", num_unique_clusters)
# Print the unique cluster values
print("Unique cluster values:", unique_clusters)




#Visualize clustering in the XY plane - 2d

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.lines as mlines

norm = Normalize(vmin=min(unique_clusters),vmax=max(unique_clusters))
cmap=plt.get_cmap("viridis")
# Plotting the clusters with boundaries
plt.figure(figsize=(10, 8))

# Define colormap for clusters
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
#cmap = ListedColormap(colors[:num_unique_clusters])

# Plot data points with different markers based on cluster label
handles=[]

for cluster_label in unique_clusters:
    cluster_data = df[df['Cluster'] == cluster_label]
    plt.scatter(cluster_data['X'], cluster_data['Y'], label=f'Cluster {cluster_label}', c=[cluster_label]*len(cluster_data), cmap=cmap, edgecolors='k',norm=norm)
    handles.append(mlines.Line2D([],[],mfc=cmap(norm(cluster_label)),linestyle="None",markersize=10,marker="o",mec="k"))


# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Plot of Clusters with Boundaries')

# Add legend
plt.colorbar(label='Cluster')
plt.grid(True)
plt.legend(handles,unique_clusters)
#plt.show()


#Visualization in the 3D plane



# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define colormap for clusters
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

#cmap = ListedColormap(colors[:num_unique_clusters])



# Plot data points with different markers based on cluster label
for cluster_label in unique_clusters:
    cluster_data = df[df['Cluster'] == cluster_label]
    ax.scatter(cluster_data['X'], cluster_data['Y'], cluster_data['Z'], label=f'Cluster {cluster_label}', c=[cluster_label]*len(cluster_data), cmap=cmap, edgecolors='k',norm=norm)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot of Clusters with Boundaries')

# Create a ScalarMappable for colorbar
sm = ScalarMappable(cmap=cmap,norm=norm)
sm.set_array(unique_clusters)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Cluster')

# Add legend
ax.legend()
plt.grid(True)
plt.show()


#First enumerate L/R, D/V info
import numpy as np
import pandas as pd

# Assuming `df` is your existing DataFrame created earlier in your script

# Initialize a new column 'Label-Num' with default value 0.5
df['Label-Num'] = 0.5

# Set values based on conditions
df.loc[df['Is_L'] == 1, 'Label-Num'] = 0.1
df.loc[df['Is_R'] == 1, 'Label-Num'] = 0.2
df.loc[df['Is_D'] == 1, 'Label-Num'] = 0.8
df.loc[df['Is_V'] == 1, 'Label-Num'] = 0.9


df["Label-Num"]=df["Neuron Label-Integer"]+df["Label-Num"]
df["Label-Num"]=df["Label-Num"].astype(float)
#print(df)

#Save CSV for visualization
if save_csv:
	# Save the DataFrame to a CSV file
	df.to_csv(foldername+'/combined_data.csv', index=False)


#Start tree implementation

#Remove rows with unlabeled human value
df = df[df['Neuron Label-Human'] != '']

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import graphviz




# Define features (Cluster and CC columns) and target variable (Label-Num)
features = ['Cluster', 'CC BPF', 'CC CyOFP', 'CC mCherry', 'CC mNeptune']
target = 'Label-Num'

X = df[features]
y = df[target]

#Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train data
# Initialize DecisionTreeRegressor
clf = DecisionTreeRegressor(random_state=42)

# Fit the model to the training data
clf.fit(X_train, y_train)

# Export the decision tree as a dot file
dot_data = export_graphviz(clf, out_file=None, feature_names=X_train.columns, filled=True, rounded=True, special_characters=True)

# Convert the dot file to a PDF using Graphviz
graph = graphviz.Source(dot_data)
graph.render("decision_tree_regression", format='pdf')

# Display the decision tree in Jupyter notebook or save it as a PDF
graph.view()

#Plot PDF for the test
# Make predictions on the test set
y_pred = clf.predict(X_test)


import matplotlib.pyplot as plt

# Plot histogram of actual target values (y_test)
plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=30, density=True, alpha=0.5, color='blue', label='Actual Distribution')

# Overlay histogram of predicted values (y_pred)
plt.hist(y_pred, bins=30, density=True, alpha=0.5, color='green', label='Predicted Distribution')

# Alternatively, plot KDE of predicted values
# kde = gaussian_kde(y_pred, bw_method='silverman')
# x_values = np.linspace(y_pred.min(), y_pred.max(), 1000)
# plt.plot(x_values, kde.evaluate(x_values), color='green', label='Predicted PDF')

plt.xlabel('Target Variable')
plt.ylabel('Density')
plt.title('Actual Distribution vs. Predicted Distribution')
plt.legend()
plt.grid(True)
plt.show()

from scipy.stats import gaussian_kde
import numpy as np

# Compute the kernel density estimate for predicted values
kde = gaussian_kde(y_pred, bw_method='silverman')

# Define a range of values for which to compute the PDF
x_values = np.linspace(min(y_pred.min(), y_test.min()), max(y_pred.max(), y_test.max()), 1000)

# Evaluate the KDE for the range of values
pdf_values = kde.evaluate(x_values)

# Plot the estimated PDF
plt.figure(figsize=(8, 6))
plt.plot(x_values, pdf_values, label='Predicted PDF', color='red')
plt.xlabel('Predicted Values')
plt.ylabel('Density')
plt.title('Actual Distribution vs. Predicted PDF')
plt.legend()
plt.grid(True)
plt.show()


