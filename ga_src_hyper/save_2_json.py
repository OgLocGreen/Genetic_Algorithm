
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
    generation = {"generation":{}}
    round_time = {"round_time":{}}
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

def save_results(save_file,round_time):
    try:
        with open(save_file, "r") as f:
            data = json.load(f)
    except:
        data = {}
    i = 0
    family_tree = { "round_time": {}}
    for x in round_time:
        round = {
            
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

def save_end_data(save_file, round_time):
    try:
        with open(save_file, "r") as f:
            data = json.load(f)
    except:
        data = {}
    data["round_time"] = round_time
    data["fitness_history"] = self.fitness_history
    with open(save_file, "w") as outfile:
        json.dump(data, outfile, indent=2)

"""
def save_gens_winner(self,multiprocessing_var=2):
    with open(self.save_file, "r") as f:
        data = json.load(f)
    self.grade_multi(multiprocessing_var=multiprocessing_var)  # damit alle induviduals noch auf fittnes überprüft werden
    self.individuals = list(sorted(self.individuals, key=lambda x: x.var_acc, reverse=True))
    i = 0
    family_tree = {"Winner": {}}
    for x in self.individuals:
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
        family_tree["Winner"][i] = generation
        i += 1
    del i
    data.update(family_tree)
    with open(self.save_file, "w") as outfile:
        json.dump(data, outfile, indent=2)
    print("saved winnerpopulation gens into",self.save_file)
    del data
    del family_tree
    gc.collect()

def log_file_beginn(self, multiprocessing_var):
    file = open(self.save_file_log,"w")
    file.write("GA\n")
    file.write("Datenset: " + str(self.dataset) +"\n")
    file.write("Knn größe: "+ str(self.knn_size) + "\n")
    file.write("Population Size: "+ str(self.pop_size) +"\n")
    file.write("Generations: " + str(self.generations)+"\n")
    file.write("Muationrate: "+ str(self.mutate_prob)+"\n")
    file.write("Retain: " + str(self.retain)+"\n")
    file.write("Jasonfile: " + str(self.save_file)+"\n")
    file.write("PC-name: "+ str(socket.gethostname()+"\n"))
    file.write("Multiprocess: "+ str(multiprocessing_var)+"\n")
    file.write("GPU: "+ str(self.gpu)+ "\n")
    file.close()


    file = open(save_file_log,"w")
    i = 0
    all_time= 0
    for i in range(0,len(round_time)):
        file.write("Round: " + str(i) +" Time: " + str(round_time[i])+"\n")
        all_time += round_time
    file.write("Time for all: "+str(all_time)+"\n")
    for i in range(0,len(self.fitness_history)):
        file.write("Generation: "+ str(i) +" Fitness: "+ str(self.fitness_history[i])+"\n")
    file.close()
"""

individuals = []
times = [1,2,3,4,5]
for i in range(0,10):
    individuals.append(test())
save_file = "test.json"
save_init_data(save_file)
save_gens(save_file,individuals,0)
save_gens(save_file,individuals,1)
save_gens(save_file,individuals,2)

save_end_data(save_file,times)

