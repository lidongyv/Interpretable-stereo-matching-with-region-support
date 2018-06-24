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

MTPCN:
P1:
groud truth:
edges on the disparity map, discriminative and representative depth edges
inside the objects, colorful edges
sample between the edges
network:
pixel-wise classification
P2:
groud truth:
object segmentation
network:
detection and 2-class classification
multi-class classification
multi two-class classification
mask rcnn for object detection
P3:
groud truth:
resegment the P2 with depth
segment with the P1 and P3
network:
re-segmentation from P2
P4:
groud truth:
pair of pixels with direction
from P1 and P3
network:
the pixels among each P3 to show the direction of the variance
classification



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


