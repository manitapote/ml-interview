
import copy

def initialize_centroids(k, points):
  return points[:k]

def euclidean_distance(points, centroids):
  #sqrt(sum(x_1 - x_2)^2)
  #points(6,2), centroids (2, 2)
  #(6, 2)
  all_dis = [[] for _ in range(len(points))]
  for i, x in enumerate(points):
    for y in centroids:
      dis = (sum((x_ - y_)**2 for x_, y_ in zip(x, y)))**(1/2)
      all_dis[i].append(dis)

  return all_dis

def min_id(x):
  return x.index(min(x))

def mean(x):
  # return [sum(y)/len(y) for y ]
  return [round(sum(y)/len(y), 2) for y in zip(*x)]

def mean_centroid(x, ids, k):
  new_centroid = []
  for i in range(k):
    p = [x[id] for id, label in enumerate(ids) if label == i]

    if len(p) == 0:
      new_centroid.append(x[i])
      continue

    new_centroid.append(mean(p))

  return new_centroid

def k_means(points, k, max_iters):
  # k as first k points
  centroids = initialize_centroids(k, points)
  past_centroid = copy.deepcopy(centroids)

  # assign points to cetnroid based on distance euclidean
  for _ in range(max_iters):
    dis = euclidean_distance(points, centroids)
   
    ids = [min_id(x) for x in dis]

    centroids = mean_centroid(points, ids, k)
    
    if past_centroid == centroids:
      break
    
    past_centroid = copy.deepcopy(centroids)


  return (centroids, ids)
    

points = [[1, 1], [1, 2], [2, 1], [8, 8], [9, 8], [8, 9]]
k = 2
max_iters = 10
k_means(points, k, max_iters)
# output = ([[1.33, 1.33], [8.33, 8.33]], [0, 0, 0, 1, 1, 1])
