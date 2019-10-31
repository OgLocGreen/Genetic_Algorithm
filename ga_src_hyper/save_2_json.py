
import json
import gc

class test(object):
    def __init__(self):
        self.gene = [1,2,3,4,5]
        self.var_acc = 1
        self.var_loss = 1
        self.variables = -1

def save_init_data(save_file):
    try:
        with open(save_file, "r") as f:
            data = json.load(f)
    except:
        data = {}

    configurations = { "config": {
        "pop_size": 50,
        "generations" :5,
        "dataset" : "mnist_fashion",
        "knn_size" : "small",
        "small_dataset" : False,
        "gpu" : False}
    }
    generation = {"generation": {}}
    round_time = {"round_time": {}}
    data.update(configurations)
    data.update(generation)
    data.update(round_time)
    with open(save_file, "w") as outfile:
        json.dump(data, outfile, indent=2)
    

def save_gens(save_file,individuals,generations):
    try:
        with open(save_file, "r") as f:
            data = json.load(f)
    except:
        data = {}
    i = 0
    family_tree = {generations: {}}
    for x in individuals:
        generation = {
            "name": i,
            "learningrate": str(x.gene[0]),
            "dropout": str(x.gene[1]),
            "epoch": str(x.gene[2]),
            "batchsize": str(x.gene[3]),
            "optimizer": str(x.gene[4]),
            "acc": str(x.var_acc),
            "loss": str(x.var_loss),
            "variables" : str(x.variables)
        }
        family_tree[generations][i] = generation
        i += 1
    del i
    data["generation"].update(family_tree)
    with open(save_file, "w") as outfile:
        json.dump(data, outfile, indent=2)
    print("saved population gens into {}".format(save_file))
    print(generation)
    del data
    del family_tree
    gc.collect()

def save_end_data(save_file, round_time, fitness_history):
    try:
        with open(save_file, "r") as f:
            data = json.load(f)
    except:
        data = {}
    data["round_time"] = round_time
    data["fitness_history"] = fitness_history
    with open(save_file, "w") as outfile:
        json.dump(data, outfile, indent=2)


individuals = []
times = [1,2,3,4,5]
fitntess = [5,4,3,2,1]
for i in range(0,10):
    individuals.append(test())
save_file = "test.json"
save_init_data(save_file)
save_gens(save_file,individuals,0)
save_gens(save_file,individuals,1)
save_gens(save_file,individuals,2)

save_end_data(save_file,times,fitntess)

