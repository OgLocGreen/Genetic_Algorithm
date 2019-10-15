

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