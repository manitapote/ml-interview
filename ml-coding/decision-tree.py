def gini_impurity(ids, y_train):
  positive = [id for id in ids if y_train[id] == 1]
  negative = [id for id in ids if y_train[id] == 0]
  #gini purity = p_1^2, p_2^2
  return 1 - ((len(positive)/len(ids))**2 + (len(negative)/len(ids))**2)

def total_gini(left_ids, right_ids, n_left, n_right, n, y_train):
  if n == 0:
    return 0

  gini_left = gini_impurity(left_ids, y_train)
  gini_right = gini_impurity(right_ids, y_train)

  #minimize this
  return (n_left/n)*gini_left + (n_right/n)*gini_right

def split(ids, features, X_train, y_train):
  best_feat = float('inf')
  best_gini = float('inf')
  best_right = []
  best_left = []
  x = sorted(set(features))
  for a, b in zip(x, x[1:]):
    feat = (a+b)/2
    left = [ids[i] for i, x in enumerate(features) if x <= feat]
    right = [ids[i] for i,  x in enumerate(features) if x > feat]

    if len(left) == 0 or len(right) == 0:
      continue

    gini = total_gini(left, right, len(left), len(right), len(X_train), y_train)

    if gini < best_gini:
      best_feat = feat
      best_gini = gini
      best_left = left
      best_right = right

  if best_gini is float('inf'):
    return None, None, None, None

  return best_gini, best_feat, best_left, best_right

def majority_class(ids, y_train):
  positive = sum(1 for id in ids if y_train[id] == 1)

  return 1 if positive > len(ids) - positive else 0

def predict(tree, x):
  if tree['leaf']:
    return tree['class']

  if x[tree['feat_idx']] <= tree['threshold']:
    return predict(tree['left'], x)
  else:
    return predict(tree['right'], x)

def build_tree(ids, 
               X_train,
               y_train, 
               depth, 
               min_samples_split, 
               max_depth
               ):
  if len(ids) == 1:
    return {'leaf': True, 'class': y_train[ids[0]]}

  if len(ids) < min_samples_split:
    return {'leaf': True, 'class': majority_class(ids, y_train)}

  if depth >= max_depth:
    return {'leaf': True, 'class': majority_class(ids, y_train)}
  
  n_feat = len(X_train[0])
  best_feat_overall = None
  best_left = []
  best_right = []
  best_gini = float('inf')
  best_feat_id = None

  for id_feat in range(n_feat):
    features = [X_train[i][id_feat] for i in ids]
    gini, best_feat, left, right = split(ids, features, X_train, y_train)

    if gini < best_gini - 1e-6:
      best_gini = gini
      best_feat_overall = best_feat
      best_left = left
      best_right = right
      best_feat_id = id_feat

  if best_feat_id is None:
    return {'leaf': True, 'class': majority_class(ids, y_train)}

  return  {
      'leaf'      : False,
      'threshold': best_feat_overall,
      'feat_idx': best_feat_id,
      'left': build_tree(best_left, X_train, y_train, depth+1, min_samples_split, max_depth),
      'right': build_tree(best_right, X_train, y_train, depth+1, min_samples_split, max_depth)
  }
  
def decision_tree_predict(X_train,
                            y_train, 
                            X_test, 
                            max_depth, 
                            min_samples_split
                          ):
  ids = [i for i in range(len(X_train))]
  tree = build_tree(ids, X_train, y_train, 0, min_samples_split, max_depth)
  
  return [predict(tree, x) for x in X_test]

X_train = [[0, 0], [1, 1], [2, 0], [3, 1]]
y_train = [0, 0, 1, 1]
X_test = [[0, 1], [2.5, 0]]
max_depth = 2
min_samples_split = 2

X_test= [[0], [3]]
X_train= [[0], [1], [2], [3]]
y_train =[0, 0, 1, 1]
max_depth = 0
min_samples_split= 2

X_test = [[1, 1], [1, 0], [0, 1]]
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 0, 0, 1]
max_depth = 2
min_samples_split = 2

X_test= [[0], [3]]
X_train= [[0], [1], [2], [3]]
y_train= [0, 0, 1, 1]
max_depth= 0
min_samples_split= 2

X_test = [[-1.5], [0], [1.5]]
X_train= [[-2], [-1], [1], [2]]
y_train= [0, 0, 1, 1]
max_depth = 3
min_samples_split= 2

X_test = [[5], [6]]
X_train= [[5], [5], [5]]
y_train= [0, 1, 1]
max_depth= 5
min_samples_split=2

print(decision_tree_predict(X_train,
                            y_train, 
                            X_test, 
                            max_depth, 
                            min_samples_split
                            ))