###############################################################################
# Project CS230: Train/Validation/Test Dataset Creation
# Authors: Matias, Negin, Alex

# date: March 24, 2020

## Final Version
###############################################################################

import json
import numpy as np
from shapely.geometry import Polygon
import pandas as pd
from pyproj import Proj, transform
from sklearn.cluster import KMeans
import numpy.ma as ma
from scipy import stats
import random
from sklearn.utils import shuffle


def merge_segment_to_crash(data_crash,data_video):
    
    '''
    This function merges road segments with corresponding hotspots. Polygon geometries
    are defined in "data_video" and "data_crash" for road segments and car accident
    hotspots respectively. We use shapely module to determine whether two polygons
    intercept.

    Args
    - data_crash: geojson file for crash hotspots
    - data_video: geojson file for road videos
    
    Output
    -merge_matrix: len_video x len_crash matrix where merge_matrix[i,j]=1
    signifies road segment i was matched to crash hotspot j
    '''

    
    #Loading Coordinate Systems. Since geometry is expressed in longitude and latitude,
    #we need to convert those coordinates into meters so that we can use the state-plane
    #representation to work with Polygons. 
    
    inProj = Proj(init='epsg:4326') #The code for lon/lat coordinates
    outProj = Proj(init='epsg:21036') #The code for meters (Kenya system)
    
    len_crash=len(data_crash['features'])
    len_video = len(data_video['features'])
    
    matrix_merge = np.zeros((len_video,len_crash))
    matrix_distance = np.zeros((len_video,len_crash))

    segment_area = np.zeros((len_video,1))
    
    for i in range(len_video):
        if i%100==0:
            print("Iteration"+" "+ str(i))
        polygon_road=[transform(inProj,outProj,k[0],k[1]) for k in data_video['features'][i]['geometry']['coordinates']]
        polygon_road = Polygon(polygon_road)
        segment_area[i,0] = polygon_road.area
        #polygon_road = polygon_road.buffer(0)
        #polygon_road = polygon_road.convex_hull
        for j in range(len_crash):
            polygon_crash =[transform(inProj,outProj,k[0],k[1]) for k in data_crash['features'][j]['geometry']['coordinates'][0]]
            polygon_crash = Polygon(polygon_crash)
            #polygon_crash = polygon_crash.buffer(0)
            #polygon_crash = polygon_crash.convex_hull
            try:
                intersection=polygon_crash.intersects(polygon_road)
                
                #Previously, I was getting geometry shape errors (eg,self-intersection)
                #when I tried to construct the intersection polygon rather than just check
                #whether the road and hotspot intersected. After checking some of the polygon
                #ids, I saw this is because the road segments are sometimes not clear segments
                #but thin triangles that are squashed such that there is a "self-intersection"
                #point which shapely does not like.
                #One solution that I found is use .buffer(0) method that "cleans" out the polygons
                #in the sense that it smooths out the self intersection points such that there is some area.
                #This actually gave me fewer errors, but still gave me some errors.
                #The second solution is .convex_hull, which convexifies the entire polygon (eg,fills out
                #all spots in the case of a concave polygon). This actually worked and the results
                #were essentially the same.
                #In the end, I just use the boolean intersects, which gives yes/no if road and crash intersect 
                #or not without any errors. The downside is that we do not have information on the area of the 
                #intersection, but we should discuss whether this is a problem. The convexification
                #solution does give us the area, but the convex polygon is an overestimate of the true polygon.
    
                #intersection=polygon_crash.intersection(polygon_road)
            except:
                print(i,j)
            matrix_merge[i,j] = intersection
            
            if intersection==True:
                matrix_distance[i,j] = 0.00001
                
            elif intersection==False:
                matrix_distance[i,j] = polygon_crash.distance(polygon_road)
                if matrix_distance[i,j]<=130:
                    matrix_merge[i,j] = True
            #matrix_merge[i,j] = intersection.area/polygon_crash.area
    return matrix_merge,segment_area,matrix_distance 

def create_feature_matrix(data_crash,matrix_merge,matrix_distance):
    
    '''
    This function merges road segments to hotspots and creates number of crashes per
    road segment.
    
    Args
    - data_crash: geojson file for crash hotspots
    - matrix_merge: len_video x len_crash matrix where merge_matrix[i,j]=1
    signifies road segment i was matched to crash hotspot j
    - matrix_distance: len_video x len_crash matrix where merge_matrix[i,j] gives the
    distance between road segment i and hotspot j
    
    Output
    -features_matrix: len_video matrix where features_matrix[i]
    gives total number of crashes between 2015 and 2018 associated to road i 
    '''

        
    len_crash=len(data_crash['features'])
    
    num_crash_1518 = np.zeros(len_crash)
    
    for i in range(len(num_crash_1518)):
        num_crash_1518[i] = data_crash['features'][i]['properties']['N_crashes_2015']+data_crash['features'][i]['properties']['N_crashes_2016'] + data_crash['features'][i]['properties']['N_crashes_2017'] + data_crash['features'][i]['properties']['N_crashes_2018']
    
    data_crash_feat = np.zeros((len_crash,len(data_crash['features'][0]['properties'])-1))

    for i in range(data_crash_feat.shape[0]):
        for j in range(data_crash_feat.shape[1]):
            data_crash_feat[i,j] = list(data_crash['features'][i]['properties'].values())[j+1]        

            
            
            
    num_crash = data_crash_feat[:,0]
            
    mask_matrix_dist = ma.masked_array(matrix_distance, mask=(matrix_merge-1)*(-1))
    weight_matrix_dist = 1/mask_matrix_dist
    weight_sum = np.sum(weight_matrix_dist,axis=1,keepdims=True)
    
    feature_matrix_unnorm = np.sum(weight_matrix_dist*num_crash_1518[None,:],axis=1)
    feature_matrix = feature_matrix_unnorm/weight_sum.flatten()
    
    #feature_matrix=np.dot(matrix_merge,data_crash_feat)
    
    feature_matrix_unnorm_tot = np.sum(weight_matrix_dist*num_crash[None,:],axis=1)
    feature_matrix_tot = feature_matrix_unnorm_tot/weight_sum.flatten()
        
    return feature_matrix_tot,num_crash,num_crash_1518,data_crash_feat,feature_matrix,mask_matrix_dist, weight_matrix_dist,weight_sum,feature_matrix_unnorm


def create_crash_labels(data_crash,num_clusters):
    
    '''
    This function defines road labels using k-means.

    Args
    - data_crash: geojson file for crash hotspots
    - num_clusters: number of desired clusters
    
    Output
    -crash_labels: road danger labels
    -kmeans_crash_centers: mean of crashes for each centroid
    '''
    
    len_crash=len(data_crash['features'])
    data_crash_feat = np.zeros((len_crash,len(data_crash['features'][0]['properties'])-1))

    for i in range(data_crash_feat.shape[0]):
        for j in range(data_crash_feat.shape[1]):
            data_crash_feat[i,j] = list(data_crash['features'][i]['properties'].values())[j+1]        
    
    cluster_features = data_crash_feat[:,0:14]
    
    kmeans_crash = KMeans(n_clusters=num_clusters, random_state=10).fit(cluster_features[:,0][:,None])
    kmeans_crash_labels = kmeans_crash.labels_
    kmeans_crash_centers = kmeans_crash.cluster_centers_
    crash_labels = kmeans_crash_labels+1
    
    return crash_labels,kmeans_crash_centers


def create_road_labels(data_video,num_clusters,feature_matrix):
    
    '''
    Not used
    '''
    
    kmeans_road = KMeans(n_clusters=num_clusters, random_state=10).fit(feature_matrix[:,None])
    kmeans_road_labels = kmeans_road.labels_
    kmeans_road_centers = kmeans_road.cluster_centers_
    road_labels = kmeans_road_labels
    
    return road_labels,kmeans_road_centers

def create_data_frame(data_video,crash_labels,matrix_distance):
    
    '''
    Not used
    '''
    
    id_video = []

    id1_crash = []
    id2_crash = []
    id3_crash = []
    id4_crash = []
    id5_crash = []
    id6_crash = []

    id1_label = []
    id2_label = []
    id3_label = []
    id4_label = []
    id5_label = []
    id6_label = []

    id1_dist = []
    id2_dist = []
    id3_dist = []
    id4_dist = []
    id5_dist = []
    id6_dist = []

    crash_list = [id1_crash,id2_crash,id3_crash,id4_crash,id5_crash,id6_crash]
    label_list = [id1_label,id2_label,id3_label,id4_label,id5_label,id6_label]
    dist_list = [id1_dist,id2_dist,id3_dist,id4_dist,id5_dist,id6_dist]


    final_label = []

#id_final_label = []
    for i in range(len(data_video['features'])):
        id_video.append(data_video['features'][i]['properties']['video_name'])
        if np.sum(matrix_merge[i,:]>0)>0:
            matched_crash_index = np.where(matrix_merge[i,:]>0)[0]
        
            dist = np.zeros(len(matched_crash_index))
            label = np.zeros(len(matched_crash_index))
        
            for index in range(len(crash_list)):
                if index<len(matched_crash_index):
                    crash_list[index].append(data_crash['features'][matched_crash_index[index]]['properties']['id'])
                    label_list[index].append(crash_labels[crash_list[index][i]-1])
                    label[index] = crash_labels[crash_list[index][i]-1]
                    dist_list[index].append(matrix_distance[i,matched_crash_index[index]])
                    dist[index] = matrix_distance[i,matched_crash_index[index]]
                #print(dist[index])
                else:
                    crash_list[index].append(0)
                    label_list[index].append(0)
                    dist_list[index].append(1000)
            #print(dist)      
            if 0 in dist:
            #print(dist)
                index = np.where(dist==0)[0][0]
                final_label.append(crash_labels[crash_list[index][i]-1])
            else:
                sum_weight = np.sum(1/dist)
                weights = (1/dist)/sum_weight
                average_label = np.sum(weights*label)
                final_label.append(average_label)
        
        else:
            for index in range(len(crash_list)):
                crash_list[index].append(0)
                label_list[index].append(0)
                dist_list[index].append(1000)
            final_label.append(0)
        
        
    data_final_table = {'Video ID':id_video,'Final Label':final_label,'1st Hotspot Id':id1_crash,'1st Hotspot Label':id1_label,'1st Hotspot Dist':id1_dist,'2nd Hotspot Id':id2_crash,'2nd Hotspot Label':id2_label,'2nd Hotspot Dist':id2_dist,'3rd Hotspot Id':id3_crash,'3rd Hotspot Label':id3_label,'3rd Hotspot Dist':id3_dist,'4th Hotspot Id':id4_crash,'4th Hotspot Label':id4_label,'4th Hotspot Dist':id4_dist,'5th Hotspot Id':id5_crash,'5th Hotspot Label':id5_label,'5th Hotspot Dist':id5_dist,'6th Hotspot Id':id6_crash,'6th Hotspot Label':id6_label,'6th Hotspot Dist':id6_dist}
    df_merged = pd.DataFrame(data_final_table)
    
    return df_merged

def split_data(table,num_clusters, test_set=False):
    
    '''
    This function splits data into train, validation and test.
    
    Args:
    -table: pandas table that includes each road segment and associate crash number
    -num_clusters: number of danger categories
    -test_set: whether to include a test set split or not
    '''
    
    
    random.seed(100)
    
    if test_set==True:
        for i in range(num_clusters):
            lab=table[table.Label==i]
    
            num_train = int(0.8*len(lab))
            num_valid = int(0.1*len(lab))
            num_test = len(lab)-num_train-num_valid
    
            lab = shuffle(lab)
            lab=lab.reset_index(drop=False)
    
            if i==0:
                train= lab.loc[0:num_train]
                valid =  lab.loc[num_train:(num_train+num_valid)]
                test = lab.loc[(num_train+num_valid):]
        
            else:
                train= train.append(lab.loc[0:num_train])
                valid= valid.append(lab.loc[num_train:(num_train+num_valid)])
                test = test.append(lab.loc[(num_train+num_valid):])
            
        train.to_csv('train1.csv')
        valid.to_csv('valid1.csv')
        test.to_csv('test1.csv')
    
    else:
        for i in range(num_clusters):
            lab=table[table.Label==i]
    
            num_train = int(0.8*len(lab))
            num_valid = len(lab)-num_train
    
            lab = shuffle(lab)
            lab=lab.reset_index(drop=False)
    
            if i==0:
                train= lab.loc[0:num_train]
                valid =  lab.loc[num_train:]
        
            else:
                train= train.append(lab.loc[0:num_train])
                valid= valid.append(lab.loc[num_train:])
            
        train.to_csv('train2.csv')
        valid.to_csv('valid2.csv')

    

if __name__ == '__main__':
    
    #Loading crash and video data
    with open('crash_clusters.geojson') as d_crash:
        data_crash = json.load(d_crash)

    len_crash=len(data_crash['features'])

    with open('video_counts_1.geojson') as d_video:
        data_video = json.load(d_video)
        
        
    print('Video Length before removing duplicates is {}'.format(len(data_video['features'])))
    #Removing duplicate video names
    video_names= []
    for i in range(len(data_video['features'])):
        video_names.append(data_video['features'][i]['properties']['video_name'])
    
    index_dup = pd.Series(video_names)[pd.Series(video_names).duplicated()].index
    
  
                
    dict_test = {}
    dict_test['features'] = []
    for i in range(len(data_video['features'])):
        if i not in index_dup:
            dict_test['features'].append(data_video['features'][i])
    
    data_video = dict_test
    len_video = len(data_video['features'])
    print('Video Length after removing duplicates is {}'.format(len_video))
    

    print('Merging road segments with car accident hotspots')
    matrix_merge, segment_road, matrix_distance = merge_segment_to_crash(data_crash,data_video)
    
    print("There are {} road segments that have not been matched to any hotspot".format(np.sum(np.sum(matrix_merge>0.0, axis=1)==0)))
    print("There are {} road segments that have been matched to at least one hotspot".format(np.sum(np.sum(matrix_merge>0.0, axis=1)>0)))
    print("There are {} road segments that have been matched to exactly one hotspot".format(np.sum(np.sum(matrix_merge>0.0, axis=1)==1)))
    print("There are {} road segments that have been matched to exactly two hotspots".format(np.sum(np.sum(matrix_merge>0.0, axis=1)==2)))
    print("There are {} road segments that have been matched to exactly three hotspots".format(np.sum(np.sum(matrix_merge>0.0, axis=1)==3)))
    print("There are {} road segments that have been matched to exactly four hotspots".format(np.sum(np.sum(matrix_merge>0.0, axis=1)==4)))
    print("There are {} road segments that have been matched to exactly five hotspots".format(np.sum(np.sum(matrix_merge>0.0, axis=1)==5)))
    print("There are {} road segments that have been matched to exactly six hotspots".format(np.sum(np.sum(matrix_merge>0.0, axis=1)==6)))
    print("There are {} road segments that have been matched to more than six hotspots".format(np.sum(np.sum(matrix_merge>0.0, axis=1)>7)))

    
    
    print('The mean road segment is of area {}'.format(np.mean(segment_road)))
    print('The standard deviation of road segment area are is {}'.format(np.std(segment_road)))
        
    print('Creating Crashes per Road')
    feature_matrix_tot,num_crash,num_crash_1518,data_crash_feat,feature_matrix,mask_matrix_dist, weight_matrix_dist,weight_sum,feature_matrix_unnorm= create_feature_matrix(data_crash,matrix_merge,matrix_distance)
        
    print('Labelling Roads')
    num_clusters = 4
    road_labels,kmeans_road_centers=create_road_labels(data_video,num_clusters,feature_matrix)
    
    sort_ind = np.argsort(kmeans_road_centers.flatten())

        
    print("The means correspond to the following percentiles in the distribution of road crashes:")
    
    print(stats.percentileofscore(feature_matrix, kmeans_road_centers[0]),
      stats.percentileofscore(feature_matrix, kmeans_road_centers[1]),
      stats.percentileofscore(feature_matrix, kmeans_road_centers[2]),
      stats.percentileofscore(feature_matrix, kmeans_road_centers[3]))
        
    road_labels_new= np.zeros(len(road_labels))   
    for i in range(len(kmeans_road_centers.flatten())):
        #print(crash_labels[crash_labels==(sort_ind[i]+1)])
        road_labels_new[road_labels==(sort_ind[i])] = i
    road_labels = road_labels_new
    
    for i in range(num_clusters):
        print("There are {} road segments in category {}".format(np.sum(road_labels==i),i))
        print("Its mean is {}".format(kmeans_road_centers[sort_ind[i]]))
    
    print('Creating Road-Label Table')
    
    id_video=[]
    
    for i in range(len(data_video['features'])):
        id_video.append(data_video['features'][i]['properties']['video_name'])
    
    data_final_table = {'Video ID':id_video,'Label':road_labels}
    table=pd.DataFrame(data_final_table)
    table.to_csv('merged_road_labels2.csv')
    
    
    split_data(table,num_clusters, test_set=True)
    split_data(table,num_clusters, test_set=False)
        
    #print('Creating Merged Table')
    
    #df_merged = create_data_frame(data_video,crash_labels,matrix_distance)
    #df_merged.to_csv('merged_videos_labels.csv')
    
    
    
    
    
    



