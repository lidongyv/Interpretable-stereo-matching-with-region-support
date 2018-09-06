PSSM
pixel-set stereo matching
log:
2018-6-19
init the PSSM
prepare the training data for multi-task pyramid cacscade network MTPCN
determine the pixel-set from semantic segmentation and disparity map
P1:
discriminative pixel detecton
representive selection
P2:
instance segmentation
P3:
related areas
depth prediction
P4:
variance direction
gradient detection

Stereo Matching:
feature extraction:
network1: receiptive field r7, five k3 conv
network2: receiptive field all, pyramid network
candidate generation:
from P2 to generate the region-level disparity:D,Dmin,Dmax
shift matching network
for each pixel in P1, the initial D is Dmin,Dmax, so the candidate is the pixels at the same line and same object
The max and min limits the search area
The candidate is both decided by the D, the same object
We search on the matching region and limits the pixels by D
cosine corrlation is applied to form the sparse cost volume
sparse-to-dense cost aggregation:
with P3
smooth with P3 on the matching costs
fill the gap with P3, mask the related region, use the convolutional layer on the cost volume with both rgb and P3
sparse soft-argmin:
soft-argmin, remove the pixel have close values more than 6
sparse-to-dense refinement:
directional interplation to fill the gaps
through cnn to refine and remove the outliers
back to interplation


