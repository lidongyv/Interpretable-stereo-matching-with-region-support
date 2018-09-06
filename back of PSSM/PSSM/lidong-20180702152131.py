P1=edges
P2=object
depth=disparity
segmentation=object
P3=np.zeros([segmentation.shape[0],segmentation.shape[1]])
for i in range(max(segmentation)):
	region=np.where(segmentation==i,disparity,0)
    max=np.floor(np.max(region))
    region=np.where(region==0,max+1,region)
    min=np.floor(np.min(region))
    i_region=0
    for j in range(min,max+1):
    	i_region+=1
    	P3+=np.where(np.logical_and(region>=j,region<j+1),i_region,0)
