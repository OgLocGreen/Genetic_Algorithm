

def log_file_beginn(self,round_time,multiprocessing_var):
        file = open(self.save_file_log,"w")
        file.write("GA\n")
        file.write("Population Size: "+ str(self.pop_size)+"\n")
        file.write("Generations: " + str(self.generations)+"\n")
        file.write("Muationrate: "+ str(self.mutate_prob)+"\n")
        file.write("Retain: " + str(self.retain)+"\n")
        file.write("Jasonfile: " + str(self.save_file)+"\n")
        file.write("PC-name: "+ str(socket.gethostname()+"\n"))
        file.write("Multiprocess: "+ str(multiprocessing_var)+"\n")
        file.close()

def logfile_end(round_time):
    i = 0
    all_time= 0
    for i in range(0,len(round_time)):
        file.write("Round: " + str(i) +" Time: " + str(round_time[i])+"\n")
        all_time += round_time
    file.write("Time for all: "+str(all_time)+"\n")
    for i in range(0,len(self.fitness_history)):
        file.write("Generation: "+ str(i) +" Fitness: "+ str(self.fitness_history[i])+"\n")
    file.close()

if __name__ == "__main__":
    save_file_log = "test.txt"
    pop_size = 50
    round_time = [1,2,3,4,5]
    file = open(save_file_log,"w")
    file.write(str("Population Size"+ str(pop_size)+"\n"))
    tmp = 0
    for i in range(0,len(round_time)):
        file.write("Round: "+ str(i) + " Time: "+ str(round_time[i] - tmp)+"\n")
        tmp = round_time[i]
    file.close()