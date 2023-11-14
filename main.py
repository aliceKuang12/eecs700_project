import patoolib
import cv2
import math
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import statistics
from shapely.geometry import LineString
import seaborn as sns
sns.set_theme()

# PART I ==========================================================================
patoolib.extract_archive("GallerySet.rar", outdir="GallerySet")
patoolib.extract_archive("ProbeSet.rar", outdir="ProbeSet")

# Given formula from appendix; use as similarity measure. 
def normalizedCorrelationCoefficient(img1: np.ndarray, img2: np.ndarray):
    x = img1.reshape((-1,1))
    y = img2.reshape((-1,1))
    xn = x - np.mean(x)
    yn = y - np.mean(y)
    r = (np.sum(xn * yn)) / (np.sqrt(np.sum(xn**2)) * np.sqrt(np.sum(yn**2)))
    return r

score_matrix = np.zeros([200,100])

# record scores to correlation matrix
pi = 2
for i in range(0, 200):
    if i%2==0:
        pi = 2 
    else: 
        pi = 3 # probe idx
    img1_path = "ProbeSet/subject" + str(math.ceil((i+1) / 2)) + "_img" + str(pi) + ".pgm"
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    for j in range(0, 100):
        img2_path = "GallerySet/subject" + str(j+1) + "_img1.pgm"
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        score_matrix[i][j] = normalizedCorrelationCoefficient(img1, img2)

print('10x10 matrix')
for i in range(10):
    print('[', end = '')
    for j in range(10):
        print(score_matrix[i][j], end = ' ')
    print(']')


# filter by idx i = j to get arrays containing two distributions.
def separate_distributions(arr, genuine, imposter):
    for i in range(200):
        for j in range(100):
            if math.floor(i/2) == j:
                genuine.append(arr[i][j])
            else:
                imposter.append(arr[i][j])

# plot w/ Seaborn distplot
def plotNormalizedDist(genuine, imposter, title):
    sns.histplot(data=genuine, bins=35, kde=True, stat='density')
    sns.histplot(data=imposter, bins=25, kde=True, stat='density')
    plt.xlabel('Threshold')
    plt.ylabel('Probability Density')
    plt.title(title + ' Normalized Distribution')
# initialize two arrays to hold genuine and imposter scores
genuine=[]
imposter=[]
separate_distributions(score_matrix, genuine, imposter)
plotNormalizedDist(genuine, imposter, 'Face Matching Score Distribution')

#(c)	Report the decidability index value. (10 points) 
def d_value_calc(genuine, imposter):
    m1 = statistics.mean(genuine)
    s1 = statistics.stdev(genuine)    
    m0 = statistics.mean(imposter)
    s0 = statistics.stdev(imposter)
    return math.sqrt(2) * abs(m1 - m0) / math.sqrt(s1*s1 + s0*s0)

print('Decidability value: ', d_value_calc(genuine, imposter))

#(d) Plot the Receiver Operating Curve (FAR vs. FRR). 
# FAR
dimensions = np.shape(imposter)
imp_rows = dimensions[0]
FAR=np.zeros([11])
for i in range(imp_rows):
    score = imposter[i]
    for j in range(11):
        if score >= j/10: # threshold       
            FAR[j] +=1
   
for i in range(11):
    FAR[i] = 100 * FAR[i] / imp_rows

# FRR
dimensions = np.shape(genuine)
gen_rows = dimensions[0]
FRR=np.zeros(11)
for i in range(gen_rows):
    score = genuine[i]
    for j in range(11):
        if score < j/10: # threshold     
            FRR[j] +=1
   
for i in range(11):
    FRR[i] = 100 * FRR[i] / gen_rows
    
# ROC curve
plt.plot(FAR, FRR)
plt.title("ROC Curve")
plt.xlabel("FAR (%)")
plt.ylabel("FRR (%)")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.rcParams["figure.autolayout"] = True


#(e) Threshold for EER and value of EER at threshold
threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
plt.plot(threshold, FAR, 'r')
plt.plot(threshold, FRR, 'b')
plt.title("Finding Threshold from FAR and FRR curves")
plt.xlabel("Threshold")
plt.ylabel("Identification Rate (%)")
plt.legend(["FAR", "FRR"], loc=0, frameon=True)

#get intersection
line1 = LineString(np.column_stack((threshold, FAR)))
line2 = LineString(np.column_stack((threshold, FRR)))
intersection = line1.intersection(line2)
x, y = intersection.xy
plt.plot(*intersection.xy, 'o')
plt.show()

print("Threshold value: ", x[0])
print("Equal Error Rate: ", y[0])

# ===============================================================================