
from KNN import train_and_evalu_CNN
from plotting import plot_winner, plot_all, plot_histogram_all, scatterplot
import KNN
import crossover
import mutation
import individual
import population




learningrate = 0.07
dropout = 0.2
epoch = 80
batchsize = 40
optimizer = 1


test_loss, test_acc = train_and_evalu_CNN(learningrate,dropout,epoch,batchsize,optimizer)

print("loss: ", test_loss,"acc: ",test_acc)