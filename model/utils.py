import numpy as np


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois] #list of user session lengths
    len_max = max(us_lens) #maximum user session length in the datasets (train/test)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)] #Pad 0's till session length is max length - inputs
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens] #List of 1 and 0, where size = max_session_length and 1 if there is an item click
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def pad_features(features):
    features[0] = [0] * len(features[1])
    return features

def get_feature_vectors(node_size, features_data):
    features = []
    features.append([features_data[i] for i in range(1, node_size)])
    return features

class Data():
    def __init__(self, data, shuffle=False, graph=None, features=None):
        inputs = data[0] #Sequence of Lists [[1,2,3],[1]]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1]) #Next click in the sequence
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph
        self.features = pad_features(features)

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        
        items, n_node, A, alias_inputs, features = [], [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        
        for u_input in inputs:
            #u_input = [947 821 839 425 424   0   0   0   0   0   0   0   0   0   0   0   0   0  0]
            #mask = [1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            node = np.unique(u_input) #[0 424 425 821 839 947]
            padded_nodes = node.tolist() + (max_n_node - len(node)) * [0]
            items.append(padded_nodes)
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) #the index in items [5, 3, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            features.append([self.features[i] for i in padded_nodes]) #Same order as the unique items/inputs - alias_inputs
            
        return alias_inputs, A, items, mask, targets, features
